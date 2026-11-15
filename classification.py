# @title Load the imports

# import keras  # only used by the commented-out experiments below
# import ml_edu.experiment
# import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import yaml
import argparse
import json
import joblib
import time
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    GroupKFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer


# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statements.")


def progress(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def data_loading(yaml_path='strain_names_and_paths.yaml', include_validation=False):
    progress(f"Loading dataset configuration: {yaml_path}")
    with open(yaml_path) as f:
        strain_names_and_paths = yaml.safe_load(f)
    paths = strain_names_and_paths['Paths']
    dataframes = []
    skipped_missing = []

    def load_group(path_map, split_role):
        for strain, strain_paths in path_map.items():
            unique_paths = list(dict.fromkeys(strain_paths))
            if len(unique_paths) != len(strain_paths):
                print(f"Warning: duplicate paths for strain '{strain}' in {split_role} set were ignored")
            strain_paths = unique_paths
            progress(f"Reading {split_role} files for strain '{strain}' ({len(strain_paths)} paths)")
            for path in strain_paths:
                path_obj = Path(path)
                if not path_obj.is_file():
                    skipped_missing.append(path)
                    print(f"Warning: Missing file skipped: {path}")
                    continue
                df = pd.read_csv(path)
                df['strain'] = strain
                df['source_path'] = str(path_obj)
                df['split_role'] = split_role
                dataframes.append(df)
            progress(f"Completed strain '{strain}' for split '{split_role}'")

    load_group(paths, split_role='train')

    if include_validation:
        validation_paths = strain_names_and_paths.get('ValidationPaths', {})
        if not validation_paths:
            raise RuntimeError(
                "Manual mode selected but no 'ValidationPaths' section found in strain_names_and_paths.yaml"
            )
        load_group(validation_paths, split_role='validation')

    if skipped_missing:
        print(f"Skipped {len(skipped_missing)} missing file(s) listed in YAML.")

    if not dataframes:
        raise RuntimeError("No readable CSV files were found from YAML path groups")

    dataset = pd.concat(dataframes, ignore_index=True)
    progress(f"Dataset loaded successfully with {len(dataset)} rows")
    return dataset, strain_names_and_paths


def train_test_by_manual_paths(dataset):
    if 'split_role' not in dataset.columns:
        raise RuntimeError("Manual split requires split_role column from data loading")

    test_df = dataset[dataset['split_role'] == 'validation'].copy()
    train_df = dataset[dataset['split_role'] != 'validation'].copy()

    if train_df.empty:
        raise RuntimeError("Manual split produced an empty training set. Check Paths entries.")
    if test_df.empty:
        raise RuntimeError("Manual split produced an empty validation set. Check ValidationPaths entries.")

    return train_df, test_df


def parse_args():
    parser = argparse.ArgumentParser(description='Train strain classifier.')
    parser.add_argument(
        '--split-mode',
        choices=['random', 'manual'],
        default='manual',
        help='manual (default): use ValidationPaths from YAML, random: sklearn train_test_split'
    )
    parser.add_argument(
        '--yaml-path',
        default='strain_names_and_paths.yaml',
        help='Path to YAML config containing Paths and optional ValidationPaths'
    )
    parser.add_argument('--test-size', type=float, default=0.2, help='Test fraction for random split mode')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for random split mode')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds for hyperparameter search')
    parser.add_argument('--search-iters', type=int, default=30, help='Number of RandomizedSearch iterations')
    parser.add_argument('--no-group-split', action='store_true', help='Disable grouped split by source_path in random split mode')
    parser.add_argument('--no-extra-trees', action='store_true', help='Skip ExtraTrees baseline comparison')
    parser.add_argument(
        '--model',
        choices=['rf', 'hgb'],
        default='rf',
        help='rf (default): RandomForest, hgb: HistGradientBoosting (much faster on large datasets)'
    )
    parser.add_argument(
        '--no-per-sample-norm',
        action='store_true',
        help='Disable per-sample robust scaling of features (scaling removes sample/day batch shifts)'
    )
    parser.add_argument('--plots', action='store_true', help='Show interactive 3D scatter plots')
    parser.add_argument('--export-csv', action='store_true', help='Write normalized_dataset.csv (large file)')
    parser.add_argument(
        '--include-shape-spatial',
        action='store_true',
        help='Also use particle shape and spacing features (excluded by default: they track sample prep, not strain)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Folder for saved results (default: classification_results/<date-time>_<model>)'
    )
    parser.add_argument(
        '--save-particle-predictions',
        action='store_true',
        help='Also save one row per test particle with predicted probabilities (large file)'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Also save the trained model as model.joblib (a large RandomForest can take several GB)'
    )
    return parser.parse_args()

args = parse_args()

dataset, split_config = data_loading(
    args.yaml_path,
    include_validation=(args.split_mode == 'manual')
)
print(dataset.head())

def add_engineered_features(df):
    if {'max_value_low', 'min_value_low'}.issubset(df.columns):
        df['range_low'] = df['max_value_low'] - df['min_value_low']
    if {'max_value_high', 'min_value_high'}.issubset(df.columns):
        df['range_high'] = df['max_value_high'] - df['min_value_high']
    # Intensity-invariant features: less sensitive to illumination/day differences.
    eps = 1e-9
    if {'mean_value_high', 'mean_value_low'}.issubset(df.columns):
        df['ratio_mean'] = df['mean_value_high'] / (df['mean_value_low'] + eps)
    if {'max_value_high', 'max_value_low'}.issubset(df.columns):
        df['ratio_max'] = df['max_value_high'] / (df['max_value_low'] + eps)
    for channel in ('high', 'low', 'divided'):
        std_col, mean_col = f'stdev_{channel}', f'mean_value_{channel}'
        if {std_col, mean_col}.issubset(df.columns):
            df[f'cv_{channel}'] = df[std_col] / (df[mean_col] + eps)
    if 'area_high' in df.columns:
        df['log_area'] = np.log1p(df['area_high'])
    return df


# Columns that are numeric but are identifiers, not particle measurements.
NON_FEATURE_COLS = {'particle_id', 'centroid_x', 'centroid_y', 'touches_border'}

# Extracted by Imageanalysis_ml.py but excluded by default. In leave-one-sample-out
# tests they lowered accuracy (shape 0.42 -> 0.38, spacing -> 0.33): they mostly
# encode sample preparation/density rather than strain.
SHAPE_SPATIAL_COLS = {
    'perimeter', 'circularity', 'solidity', 'extent', 'equiv_diameter',
    'major_axis', 'minor_axis', 'aspect_ratio', 'eccentricity',
    'nn_distance', 'n_neighbors_50px',
}


def per_sample_robust_scale(df, feature_cols, group_col='source_path'):
    """Scale each feature by its own sample's median and IQR.

    Uses no labels, so it is also valid on held-out samples. It removes
    sample/day-level shifts in brightness and particle size, leaving the
    within-sample distribution shape that the classifier learns from.
    """
    grouped = df.groupby(group_col)[feature_cols]
    median = grouped.transform('median')
    iqr = grouped.transform(lambda c: c.quantile(0.75) - c.quantile(0.25))
    df[feature_cols] = (df[feature_cols] - median) / (iqr + 1e-9)
    return df


def split_dataset(raw_df, split_mode, test_size, random_state, grouped_split=True):
    if split_mode == 'manual':
        train_df, test_df = train_test_by_manual_paths(raw_df)
        print(
            f"Manual split selected (ValidationPaths from YAML). "
            f"Train samples: {len(train_df)}, Validation samples: {len(test_df)}"
        )
        return train_df, test_df

    if grouped_split and 'source_path' in raw_df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(raw_df, raw_df['strain'], groups=raw_df['source_path']))
        train_df = raw_df.iloc[train_idx].copy()
        test_df = raw_df.iloc[test_idx].copy()
        print(
            f"Random grouped split selected (test_size={test_size}, random_state={random_state}). "
            f"Train samples: {len(train_df)}, Test samples: {len(test_df)}"
        )
        return train_df, test_df

    train_df, test_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df['strain']
    )
    print(
        f"Random stratified split selected (test_size={test_size}, random_state={random_state}). "
        f"Train samples: {len(train_df)}, Test samples: {len(test_df)}"
    )
    return train_df.copy(), test_df.copy()


def fit_global_iqr_thresholds(df, feature_cols, multiplier=1.5):
    thresholds = {}
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        thresholds[col] = (lower, upper)
    return thresholds


def apply_iqr_filter(df, thresholds):
    mask = pd.Series(True, index=df.index)
    for col, (lower, upper) in thresholds.items():
        mask = mask & df[col].between(lower, upper, inclusive='both')
    return df[mask].copy()


def build_feature_matrices(train_df, test_df, per_sample_norm=True, include_shape_spatial=False):
    train_df = add_engineered_features(train_df.copy())
    test_df = add_engineered_features(test_df.copy())

    numeric_cols_train = set(train_df.select_dtypes(include=[np.number]).columns)
    numeric_cols_test = set(test_df.select_dtypes(include=[np.number]).columns)
    excluded = NON_FEATURE_COLS if include_shape_spatial else NON_FEATURE_COLS | SHAPE_SPATIAL_COLS
    feature_cols = sorted(numeric_cols_train.intersection(numeric_cols_test) - excluded)

    if not feature_cols:
        raise RuntimeError("No shared numeric feature columns were found between train and test sets")

    if per_sample_norm and 'source_path' in train_df.columns:
        train_df = per_sample_robust_scale(train_df, feature_cols)
        test_df = per_sample_robust_scale(test_df, feature_cols)
        print("Per-sample robust scaling applied (median/IQR per source_path).")

    print(f"Outlier removal disabled. Training rows kept: {len(train_df)}")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df['strain'].copy()
    y_test = test_df['strain'].copy()

    # Median imputation learned from train only.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test, y_train, y_test, train_df, test_df, feature_cols


def tune_model(X_train, y_train, groups, random_state, cv_folds, search_iters, model_kind='rf'):
    if model_kind == 'hgb':
        estimator = HistGradientBoostingClassifier(random_state=random_state)
        param_distributions = {
            'max_iter': [200, 400, 800],
            'learning_rate': [0.03, 0.05, 0.1, 0.2],
            'max_leaf_nodes': [15, 31, 63, 127],
            'min_samples_leaf': [20, 50, 100, 200],
            'l2_regularization': [0.0, 0.1, 1.0],
            'class_weight': [None, 'balanced'],
        }
    else:
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        param_distributions = {
            'n_estimators': [300, 500, 800, 1200],
            'max_depth': [None, 8, 12, 20, 30, 40],
            'min_samples_split': [2, 4, 8, 12, 20],
            'min_samples_leaf': [1, 2, 4, 8, 12],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'bootstrap': [True],
        }

    scorer = make_scorer(f1_score, average='macro')
    if groups is not None and len(np.unique(groups)) >= 2:
        n_splits = min(cv_folds, len(np.unique(groups)))
        cv = GroupKFold(n_splits=max(2, n_splits))
        fit_kwargs = {'groups': groups}
        print(f"Using GroupKFold CV with {max(2, n_splits)} folds.")
    else:
        cv = StratifiedKFold(n_splits=max(2, cv_folds), shuffle=True, random_state=random_state)
        fit_kwargs = {}
        print(f"Using StratifiedKFold CV with {max(2, cv_folds)} folds.")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=search_iters,
        scoring={'accuracy': 'accuracy', 'f1_macro': scorer},
        refit='f1_macro',
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    total_fits = search_iters * cv.get_n_splits()
    progress(
        f"Starting RandomizedSearchCV ({search_iters} candidates x {cv.get_n_splits()} folds = {total_fits} fits)"
    )
    t0 = time.time()
    search.fit(X_train, y_train, **fit_kwargs)
    elapsed_min = (time.time() - t0) / 60
    progress(f"RandomizedSearchCV finished in {elapsed_min:.1f} minutes")
    return search

# for x_axis_data, y_axis_data in [
#     ('mean_value_high', 'strain'),
#     ('mean_value_low', 'strain'),
#     ('area_high', 'strain')]:
#     px.scatter(dataset, x=x_axis_data, y=y_axis_data, color='strain').show()

# px.scatter_3d(
#     dataset,
#     x='mean_value_high',
#     y='mean_value_low',
#     z='area_high',
#     color='strain',
#     ).show()

# Calculate the Z-scores of each numerical column in the raw data and write
# them into a new DataFrame named df_norm.

train_df, test_df = split_dataset(
    dataset,
    args.split_mode,
    test_size=args.test_size,
    random_state=args.random_state,
    grouped_split=not args.no_group_split,
)

X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = build_feature_matrices(
    train_df,
    test_df,
    per_sample_norm=not args.no_per_sample_norm,
    include_shape_spatial=args.include_shape_spatial,
)
print(f"Using {len(feature_cols)} features: {feature_cols}")

if args.export_csv or args.plots:
    train_export = train_df.copy()
    train_export['split_role'] = 'train'
    test_export = test_df.copy()
    test_export['split_role'] = 'test'
    normalized_dataset = pd.concat([train_export, test_export], ignore_index=True)

if args.export_csv:
    normalized_dataset.to_csv("normalized_dataset.csv", index=False)

if args.plots:
    for x, y, z in [
        ('max_value_low', 'max_value_high', 'pearson_corr'),
        ('center_edge_ratio_low', 'center_edge_ratio_high', 'stdev_divided'),
        ('min_value_low', 'min_value_high', 'area_divided'),
    ]:
        px.scatter_3d(normalized_dataset, x=x, y=y, z=z, color='strain').show()


def report_aggregated(proba, df, y_true, level_cols, label):
    """Average particle probabilities within each image/sample and report accuracy.

    Returns one row per group: the grouping columns, true strain, predicted strain,
    number of particles and the mean probability of each strain.
    """
    grouped = proba.groupby([df[c] for c in level_cols])
    mean_proba = grouped.mean()
    table = mean_proba.add_prefix('prob_')
    table.insert(0, 'n_particles', grouped.size())
    table.insert(0, 'predicted', mean_proba.idxmax(axis=1))
    table.insert(0, 'true', y_true.groupby([df[c] for c in level_cols]).first())
    acc = accuracy_score(table['true'], table['predicted'])
    f1 = f1_score(table['true'], table['predicted'], average='macro')
    print(f"{label}-level ({len(table)} {label.lower()}s): accuracy={acc:.4f}, macro-F1={f1:.4f}")
    return table.reset_index(), {'n': int(len(table)), 'accuracy': acc, 'macro_f1': f1}


def evaluate(model, name):
    """Print test results for a model and save them to the output folder."""
    prefix = name.lower().replace(' ', '_')
    proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_, index=test_df.index)
    y_pred = proba.idxmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{name} particle-level test accuracy: {acc:.4f}")
    print(f"{name} particle-level test macro-F1: {f1:.4f}")
    print(f"{name} classification report:\n", classification_report(y_test, y_pred))
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"{name} confusion matrix (rows = true, columns = predicted, order {labels}):\n", cm)

    metrics = {'particle': {'n': int(len(y_test)), 'accuracy': acc, 'macro_f1': f1}}
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.to_csv(
        output_dir / f'{prefix}_classification_report.csv')
    pd.DataFrame(cm, index=pd.Index(labels, name='true'), columns=pd.Index(labels, name='predicted')).to_csv(
        output_dir / f'{prefix}_confusion_matrix.csv')

    if 'image' in test_df.columns and 'source_path' in test_df.columns:
        images, metrics['image'] = report_aggregated(proba, test_df, y_test, ['source_path', 'image'], 'Image')
        images.to_csv(output_dir / f'{prefix}_image_predictions.csv', index=False)
    if 'source_path' in test_df.columns:
        samples, metrics['sample'] = report_aggregated(proba, test_df, y_test, ['source_path'], 'Sample')
        samples.to_csv(output_dir / f'{prefix}_sample_predictions.csv', index=False)
        print(samples[['source_path', 'true', 'predicted']].to_string(index=False))

    if args.save_particle_predictions:
        keep = [c for c in ('source_path', 'image', 'particle_id') if c in test_df.columns]
        particles = test_df[keep].copy()
        particles['true'] = y_test
        particles['predicted'] = y_pred
        particles = particles.join(proba.add_prefix('prob_'))
        particles.to_csv(output_dir / f'{prefix}_particle_predictions.csv', index=False)
    return metrics


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v))


run_started = datetime.now()
output_dir = Path(args.output_dir) if args.output_dir else (
    Path('classification_results') / f"{run_started.strftime('%Y%m%d-%H%M%S')}_{args.model}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving results to: {output_dir}")
save_json(output_dir / 'run_config.json', {
    'started': run_started.isoformat(timespec='seconds'),
    'arguments': vars(args),
    'features': feature_cols,
    'train_particles': int(len(train_df)),
    'test_particles': int(len(test_df)),
    'train_samples': sorted(train_df['source_path'].unique()) if 'source_path' in train_df.columns else None,
    'test_samples': sorted(test_df['source_path'].unique()) if 'source_path' in test_df.columns else None,
})


model_name = 'HistGradientBoosting' if args.model == 'hgb' else 'Random Forest'
print(f"\nTuning {model_name} classifier for strain prediction...")
train_groups = train_df['source_path'].values if 'source_path' in train_df.columns else None
search = tune_model(
    X_train,
    y_train,
    groups=train_groups,
    random_state=args.random_state,
    cv_folds=args.cv_folds,
    search_iters=args.search_iters,
    model_kind=args.model,
)
best_model = search.best_estimator_

print(f"Best {model_name} CV macro-F1: {search.best_score_:.4f}")
print(f"Best {model_name} params: {search.best_params_}")
pd.DataFrame(search.cv_results_).sort_values('rank_test_f1_macro').to_csv(output_dir / 'cv_results.csv', index=False)
summary = {
    'model': model_name,
    'cv_macro_f1': search.best_score_,
    'best_params': search.best_params_,
    'test': {model_name: evaluate(best_model, model_name)},
}

if args.model == 'rf' and not args.no_extra_trees:
    print("\nTraining ExtraTrees baseline for comparison...")
    et_model = ExtraTreesClassifier(
        n_estimators=best_model.n_estimators,
        max_depth=best_model.max_depth,
        min_samples_split=best_model.min_samples_split,
        min_samples_leaf=best_model.min_samples_leaf,
        max_features=best_model.max_features,
        class_weight=best_model.class_weight,
        random_state=args.random_state,
        n_jobs=-1,
    )
    et_model.fit(X_train, y_train)
    summary['test']['ExtraTrees'] = evaluate(et_model, 'ExtraTrees')

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 15 {model_name} feature importances:")
    print(feature_importance.head(15))
    feature_importance.rename_axis('feature').rename('importance').to_csv(output_dir / 'feature_importances.csv')

if args.save_model:
    joblib.dump({'model': best_model, 'features': feature_cols}, output_dir / 'model.joblib')

summary['finished'] = datetime.now().isoformat(timespec='seconds')
save_json(output_dir / 'summary.json', summary)
print(f"\nResults saved to: {output_dir}")


# # Filter out RML strain
# filtered_dataset = normalized_dataset[normalized_dataset['strain'] != 'ME7']
    


# px.scatter_3d(
#     normalized_dataset,
#     x='mean_value_low',
#     y='range_low',
#     z='range_high',
#     color='strain',
#     ).show()

# # Copy the strain column to the normalized dataset
# normalized_dataset['strain'] = dataset['strain']

# keras.utils.set_random_seed(42)

# # Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
# # then show 10 randomly selected rows.
# normalized_dataset['Class_Bool'] = (
#     # Returns true if class is Cammeo, and false if class is Osmancik
#     normalized_dataset['Class'] == 'Cammeo'
# ).astype(int)


# # Create indices at the 80th and 90th percentiles
# number_samples = len(normalized_dataset)
# index_80th = round(number_samples * 0.8)
# index_90th = index_80th + round(number_samples * 0.1)

# # Randomize order and split into train, validation, and test with a .8, .1, .1 split
# shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
# train_data = shuffled_dataset.iloc[0:index_80th]
# validation_data = shuffled_dataset.iloc[index_80th:index_90th]
# test_data = shuffled_dataset.iloc[index_90th:]

# label_columns = ['Class', 'Class_Bool']

# train_features = train_data.drop(columns=label_columns)
# train_labels = train_data['Class_Bool'].to_numpy()
# validation_features = validation_data.drop(columns=label_columns)
# validation_labels = validation_data['Class_Bool'].to_numpy()
# test_features = test_data.drop(columns=label_columns)
# test_labels = test_data['Class_Bool'].to_numpy()

# # Name of the features we'll train our model on.
# input_features = [
#     'Eccentricity',
#     'Major_Axis_Length',
#     'Area',
# ]

# # @title Define the functions that create and train a model.


# def create_model(
#     settings: ml_edu.experiment.ExperimentSettings,
#     metrics: list[keras.metrics.Metric],
# ) -> keras.Model:
#   """Create and compile a simple classification model."""
#   model_inputs = [
#       keras.Input(name=feature, shape=(1,))
#       for feature in settings.input_features
#   ]
#   # Use a Concatenate layer to assemble the different inputs into a single
#   # tensor which will be given as input to the Dense layer.
#   # For example: [input_1[0][0], input_2[0][0]]

#   concatenated_inputs = keras.layers.Concatenate()(model_inputs)
#   model_output = keras.layers.Dense(
#       units=1, name='dense_layer', activation=keras.activations.sigmoid
#   )(concatenated_inputs)
#   model = keras.Model(inputs=model_inputs, outputs=model_output)
#   # Call the compile method to transform the layers into a model that
#   # Keras can execute.  Notice that we're using a different loss
#   # function for classification than for regression.
#   model.compile(
#       optimizer=keras.optimizers.RMSprop(
#           settings.learning_rate
#       ),
#       loss=keras.losses.BinaryCrossentropy(),
#       metrics=metrics,
#   )
#   return model


# def train_model(
#     experiment_name: str,
#     model: keras.Model,
#     dataset: pd.DataFrame,
#     labels: np.ndarray,
#     settings: ml_edu.experiment.ExperimentSettings,
# ) -> ml_edu.experiment.Experiment:
#   """Feed a dataset into the model in order to train it."""

#   # The x parameter of keras.Model.fit can be a list of arrays, where
#   # each array contains the data for one feature.
#   features = {
#       feature_name: np.array(dataset[feature_name])
#       for feature_name in settings.input_features
#   }

#   history = model.fit(
#       x=features,
#       y=labels,
#       batch_size=settings.batch_size,
#       epochs=settings.number_epochs,
#   )

#   return ml_edu.experiment.Experiment(
#       name=experiment_name,
#       settings=settings,
#       model=model,
#       epochs=history.epoch,
#       metrics_history=pd.DataFrame(history.history),
#   )


# print('Defined the create_model and train_model functions.')

# # Let's define our first experiment settings.
# settings = ml_edu.experiment.ExperimentSettings(
#     learning_rate=0.001,
#     number_epochs=60,
#     batch_size=100,
#     classification_threshold=0.35,
#     input_features=input_features,
# )

# metrics = [
#     keras.metrics.BinaryAccuracy(
#         name='accuracy', threshold=settings.classification_threshold
#     ),
#     keras.metrics.Precision(
#         name='precision', thresholds=settings.classification_threshold
#     ),
#     keras.metrics.Recall(
#         name='recall', thresholds=settings.classification_threshold
#     ),
#     keras.metrics.AUC(num_thresholds=100, name='auc'),
# ]

# # Establish the model's topography.
# model = create_model(settings, metrics)

# # Train the model on the training set.
# experiment = train_model(
#     'baseline', model, train_features, train_labels, settings
# )

# # Plot metrics vs. epochs
# ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
# ml_edu.results.plot_experiment_metrics(experiment, ['auc'])

# plt.show()

# def compare_train_validation(experiment: ml_edu.experiment.Experiment, validation_metrics: dict[str, float]):
#   print('Comparing metrics between train and validation:')
#   for metric, validation_value in validation_metrics.items():
#     print('------')
#     print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
#     print(f'Validation {metric}:  {validation_value:.4f}')


# # Evaluate validation metrics
# validation_metrics = experiment.evaluate(validation_features, validation_labels)
# compare_train_validation(experiment, validation_metrics)
