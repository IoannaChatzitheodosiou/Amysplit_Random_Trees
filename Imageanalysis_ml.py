
#Importing Relevant Libraries
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pylab as plt
import os
from read_roi import read_roi_zip
from pystackreg import StackReg
import tifffile
import sys
import csv
import yaml
from pathlib import Path

def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "no", "n", "off"}:
        return False
    return default


def image_listing(datapath):  #function that reads your images from the folder
    derived_suffixes = (
        "_low.tif",
        "_high.tif",
        "_stack.tif",
        "_divided.tif",
        "_divided_non_aligned.tif",
        "_contoured.tif",
    )
    # Match .tif/.tiff files regardless of upper/lowercase extension.
    imagelist = [
        str(p)
        for p in Path(datapath).iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and not p.name.startswith("._")
            and not p.name.startswith(".")
            and not p.name.lower().endswith(derived_suffixes)
        )
    ]
    open_images = {} #dictionary for open images
    # Iterating over the list of images and opening them
    for image in imagelist:
        # Get the filename (without the full path)
        filename = os.path.basename(image)
        # Read the image and store it in the dictionary
        try:
            open_images[filename] = tifffile.imread(image)
        except Exception as e:
            print(f"Skipping unreadable image: {filename} ({e})")
    return (open_images)

def roi_selection(roipath): #function that reads the rois
    rois = read_roi_zip(roipath)
    roi_high = rois['high'] #higher wavelength roi
    roi_low = rois['low']  #low wavelength roi
    return (roi_high, roi_low)

def apply_crop(roi, image): #function that crops the image into two, according to the rois
    img_h, img_w = image.shape[:2]
    top_left_y = int(roi['top'])
    top_left_x = int(roi['left'])
    roi_h = int(roi['height'])
    roi_w = int(roi['width'])

    bottom_right_y = top_left_y + roi_h
    bottom_right_x = top_left_x + roi_w

    clipped_y1 = max(0, top_left_y)
    clipped_x1 = max(0, top_left_x)
    clipped_y2 = min(img_h, bottom_right_y)
    clipped_x2 = min(img_w, bottom_right_x)

    if clipped_y1 >= clipped_y2 or clipped_x1 >= clipped_x2:
        raise ValueError(
            "ROI crop is outside image bounds after clipping "
            f"(roi top={top_left_y}, left={top_left_x}, height={roi_h}, width={roi_w}; "
            f"image shape={image.shape})"
        )

    if (clipped_y1, clipped_x1, clipped_y2, clipped_x2) != (top_left_y, top_left_x, bottom_right_y, bottom_right_x):
        print(
            "Warning: ROI exceeds image bounds; clipping crop to valid area "
            f"(roi top={top_left_y}, left={top_left_x}, height={roi_h}, width={roi_w}; "
            f"image shape={image.shape})"
        )

    return image[clipped_y1:clipped_y2, clipped_x1:clipped_x2]

def crop_divided(divided):
    #define crop area
    height, width = divided.shape[:2]
    center_x = width // 2
    center_y = height // 2
    # Compute half-size crop bounds
    half_w = width // 4
    half_h = height // 4
    x1 = center_x - half_w
    x2 = center_x + half_w
    y1 = center_y - half_h
    y2 = center_y + half_h
    # Crop the image
    cropped = divided[y1:y2, x1:x2]
    return cropped

def match_common_shape(img_a, img_b):
    """Trim two images to their largest shared top-left rectangle."""
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(
            f"Invalid crop dimensions after matching: {img_a.shape} vs {img_b.shape}"
        )
    return img_a[:h, :w], img_b[:h, :w]


def image_stacking(cropped_high, cropped_low, image_name=None, crop_divided_center=True):
    if cropped_high.shape != cropped_low.shape:
        print(
            f"Warning: ROI crop shape mismatch"
            f"{f' for {image_name}' if image_name else ''}: "
            f"high={cropped_high.shape}, low={cropped_low.shape}. "
            f"Trimming to common shape."
        )
        cropped_high, cropped_low = match_common_shape(cropped_high, cropped_low)

    sr = StackReg(StackReg.AFFINE)
    aligned_high = sr.register_transform(cropped_low, cropped_high)
    # Combine the two crops into a 2-channel grayscale image
    try:
        combined_image = np.stack((aligned_high, cropped_low), axis=-1)  # Use np.stack to maintain grayscale
    except:
        print(cropped_low.shape)
        print(aligned_high.shape)

    split1 = combined_image[:, :, 0]
    split2 = combined_image[:, :, 1]

    combined = np.array([split1,split2], dtype=np.uint16)

    combined = combined.astype(np.uint16)

    aligned_high[aligned_high==0]= np.median (aligned_high)
    divided =   aligned_high*1500 / cropped_low
    divided = divided.astype(np.uint16)
    if crop_divided_center:
        divided = crop_divided(divided)
    divided2 =  cropped_high*1500 /cropped_low 
    divided2 = divided2.astype(np.uint16)
    # Return the (possibly trimmed) low image so callers measure on the same
    # pixel grid that the divided image was computed from.
    return combined, divided, divided2, aligned_high, cropped_low


def process_tif(image_name, image, outputpath, roi_high, roi_low, crop_divided_center=True):
    image_folder = image_name.replace(".tif","")
    if not os.path.exists(outputpath + "/" + image_folder):
        os.makedirs(outputpath + "/" + image_name.replace(".tif",""))
    image_low = apply_crop(roi_low, image)
    image_high = apply_crop(roi_high, image)
    lowname = outputpath + "/" +  image_folder + '/' + image_folder  + "_low.tif"
    highname = outputpath + "/" + image_folder + '/' + image_folder + "_high.tif"
    stackname = outputpath + "/" + image_folder + '/' + image_folder + "_stack.tif"
    divname = outputpath + "/" + image_folder + '/' + image_folder + "_divided.tif"
    divname2 = outputpath + "/" + image_folder + '/' + image_folder + "_divided_non_aligned.tif"
    tifffile.imwrite(lowname, image_low, imagej=True)
    tifffile.imwrite(highname, image_high, imagej=True)
    stacked_image, divided, divided2, aligned_high, matched_low = image_stacking(
        image_high,
        image_low,
        image_name=image_name,
        crop_divided_center=crop_divided_center,
    )
    tifffile.imwrite(stackname, stacked_image, imagej=True)
    tifffile.imwrite(divname, divided, imagej=True)
    tifffile.imwrite(divname2, divided2, imagej=True)
    return aligned_high, matched_low, divided

# Per-particle feature extraction settings
LOCAL_PAD = 10          # pixels around each particle's bounding box for local masks
RING_INNER = 2          # local background ring starts this many pixels outside the particle
RING_OUTER = 8          # ... and ends here
NEIGHBOR_RADIUS = 50    # pixels, for the local particle density feature


def _safe_div(a, b):
    return a / b if b not in (0, 0.0) and np.isfinite(b) else np.nan


def _skewness(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return np.nan
    sd = x.std()
    return float(np.mean((x - x.mean()) ** 3) / sd ** 3) if sd > 0 else 0.0


def _center_edge_ratio(values, center_mask, edge_mask):
    center, edge = values[center_mask], values[edge_mask]
    if center.size == 0 or edge.size == 0:
        return np.nan
    return _safe_div(np.mean(center), np.mean(edge))


def shape_features(contour, area):
    """Morphology descriptors that do not depend on intensity."""
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    _, _, w, h = cv2.boundingRect(contour)
    m = cv2.moments(contour)
    if m['m00'] > 0:
        mu20, mu02, mu11 = m['mu20'] / m['m00'], m['mu02'] / m['m00'], m['mu11'] / m['m00']
        common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
        lam1 = (mu20 + mu02) / 2 + common
        lam2 = max((mu20 + mu02) / 2 - common, 0.0)
        major_axis, minor_axis = 4 * np.sqrt(lam1), 4 * np.sqrt(lam2)
        eccentricity = np.sqrt(1 - lam2 / lam1) if lam1 > 0 else np.nan
    else:
        major_axis = minor_axis = eccentricity = np.nan
    return {
        "perimeter": perimeter,
        "circularity": _safe_div(4 * np.pi * area, perimeter ** 2),
        "solidity": _safe_div(area, hull_area),
        "extent": _safe_div(area, w * h),
        "equiv_diameter": np.sqrt(4 * area / np.pi),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "aspect_ratio": _safe_div(major_axis, minor_axis),
        "eccentricity": eccentricity,
    }


def neighbor_features(centroids, radius=NEIGHBOR_RADIUS):
    """Nearest-neighbour distance and local density for each particle."""
    n = len(centroids)
    if n < 2:
        return np.full(n, np.nan), np.zeros(n, dtype=int)
    pts = np.asarray(centroids, dtype=np.float64)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1), (d <= radius).sum(axis=1)


def calculate_mean_value_and_areas(divided_image, high_image, low_image, image_name, output_path, contour_threshold):
    areas_high = []
    areas_low = []
    image_folder = image_name.replace(".tif","")
    if not os.path.exists(output_path + "/" + image_folder):
        os.makedirs(output_path + "/" + image_name.replace(".tif",""))
    
    contourname = output_path + "/" +  image_folder + '/' + image_folder  + "_contoured.tif"
    
    norm_img = cv2.normalize(divided_image, None, 0, 255, cv2.NORM_MINMAX)
    divided_normalized = norm_img.astype('uint8')

    # Segmentation is performed on divided_image, which may be center-cropped.
    # Measure high/low intensities in the same spatial window.
    if high_image.shape[:2] != divided_normalized.shape[:2]:
        h_full, w_full = high_image.shape[:2]
        h_div, w_div = divided_normalized.shape[:2]
        y1 = max(0, (h_full - h_div) // 2)
        x1 = max(0, (w_full - w_div) // 2)
        y2 = min(h_full, y1 + h_div)
        x2 = min(w_full, x1 + w_div)
        analysis_high = high_image[y1:y2, x1:x2]
        analysis_low = low_image[y1:y2, x1:x2]
        if analysis_high.shape[:2] != divided_normalized.shape[:2]:
            common_h = min(analysis_high.shape[0], divided_normalized.shape[0])
            common_w = min(analysis_high.shape[1], divided_normalized.shape[1])
            analysis_high = analysis_high[:common_h, :common_w]
            analysis_low = analysis_low[:common_h, :common_w]
            divided_normalized = divided_normalized[:common_h, :common_w]
    else:
        analysis_high = high_image
        analysis_low = low_image
    #Adaptive thresholding
    thresh = cv2.adaptiveThreshold(divided_normalized,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=51,  # must be odd and tailored to feature size
                                    C=5)
    
    kernel = np.ones((3, 3), np.uint8) 
    eroded = cv2.erode(thresh, kernel, iterations=1) #morphological erosion to decrease particle size
    kernel = np.ones((4, 4), np.uint8) 
    expanded = cv2.dilate(eroded, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mask of every detected object (before size filtering), used to keep other
    # particles out of the background estimates.
    all_contours_mask = np.zeros(analysis_high.shape[:2], dtype='uint8')
    cv2.drawContours(all_contours_mask, contours, -1, 255, thickness=cv2.FILLED)
    global_bg_mask = all_contours_mask == 0
    background_high = np.mean(analysis_high[global_bg_mask]) if np.any(global_bg_mask) else np.nan
    background_low = np.mean(analysis_low[global_bg_mask]) if np.any(global_bg_mask) else np.nan

    contours = [cnt for cnt in contours if len(cnt) > contour_threshold]

    # Draw contours on a 3-channel image for visualization
    color_image = cv2.cvtColor(divided_normalized, cv2.COLOR_GRAY2BGR)
    thecontours = cv2.drawContours(color_image, contours, -1, (0,255,0), 2)

    img_h, img_w = analysis_high.shape[:2]
    ring_outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_OUTER + 1, 2 * RING_OUTER + 1))
    ring_inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_INNER + 1, 2 * RING_INNER + 1))

    records = []
    centroids = []
    mean_values_divided = []
    particle_id = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        areas_high.append(area)
        areas_low.append(area)

        # Work in a padded window around the particle instead of the full image.
        bx, by, bw, bh = cv2.boundingRect(contour)
        y0, x0 = max(0, by - LOCAL_PAD), max(0, bx - LOCAL_PAD)
        y1w, x1w = min(img_h, by + bh + LOCAL_PAD), min(img_w, bx + bw + LOCAL_PAD)
        local_mask = np.zeros((y1w - y0, x1w - x0), dtype='uint8')
        cv2.drawContours(local_mask, [contour], -1, 255, thickness=cv2.FILLED, offset=(-x0, -y0))
        inside = local_mask == 255
        hi = analysis_high[y0:y1w, x0:x1w]
        lo = analysis_low[y0:y1w, x0:x1w]
        dv = divided_normalized[y0:y1w, x0:x1w]

        # Pixels within the contour, from the original (not normalized) images
        pixels_high = hi[inside]
        pixels_low = lo[inside]
        pixels_divided = dv[inside]
        if pixels_high.size == 0 or pixels_low.size == 0 or pixels_divided.size == 0:
            continue

        mean_value_high = np.mean(pixels_high)
        mean_value_low = np.mean(pixels_low)
        mean_value_divided = np.mean(pixels_divided)
        if np.isnan(mean_value_high) or np.isnan(mean_value_low) or np.isnan(mean_value_divided):
            continue

        # Center vs edge of the particle (distance from the particle boundary).
        # Center = inner 50% of max distance, Edge = outer 20%, both inside the particle.
        dist_map = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
        max_dist = dist_map.max()
        if max_dist > 0:
            center_mask = inside & (dist_map >= 0.5 * max_dist)
            edge_mask = inside & (dist_map <= 0.2 * max_dist)
        else:
            center_mask = edge_mask = np.zeros_like(inside)

        # Pixel-wise high/low ratio on raw intensities (independent of per-image normalization)
        hi_f = hi.astype(np.float64)
        lo_f = lo.astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_img = np.where(lo_f > 0, hi_f / lo_f, np.nan)
        ratio_px = ratio_img[inside]
        ratio_px = ratio_px[np.isfinite(ratio_px)]

        # Local background: ring around the particle, excluding other particles
        dil_outer = cv2.dilate(local_mask, ring_outer_kernel)
        dil_inner = cv2.dilate(local_mask, ring_inner_kernel)
        ring = (dil_outer > 0) & (dil_inner == 0) & (all_contours_mask[y0:y1w, x0:x1w] == 0)
        if ring.sum() >= 5:
            bg_local_high = float(np.mean(hi[ring]))
            bg_local_low = float(np.mean(lo[ring]))
            bg_local_std_high = float(np.std(hi[ring]))
        else:
            bg_local_high, bg_local_low, bg_local_std_high = background_high, background_low, np.nan

        # Pearson correlation between high and low pixels within the contour
        if pixels_high.size >= 2 and np.std(pixels_high) > 0 and np.std(pixels_low) > 0:
            pearson_corr = np.corrcoef(pixels_high, pixels_low)[0, 1]
        else:
            pearson_corr = np.nan

        cx, cy = bx + bw / 2.0, by + bh / 2.0
        m = cv2.moments(contour)
        if m['m00'] > 0:
            cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
        centroids.append((cx, cy))
        touches_border = int(bx <= 0 or by <= 0 or bx + bw >= img_w or by + bh >= img_h)

        record = {
            "image": image_name,
            "particle_id": particle_id,
            # --- original columns (same names/meaning as before) ---
            "mean_value_high": mean_value_high,
            "area_high": area,
            "max_value_high": np.max(pixels_high),
            "min_value_high": np.min(pixels_high),
            "stdev_high": np.std(pixels_high),
            "pearson_corr": pearson_corr,
            "center_edge_ratio_high": _center_edge_ratio(hi, center_mask, edge_mask) if max_dist > 0 else np.nan,
            "mean_value_low": mean_value_low,
            "area_low": area,
            "max_value_low": np.max(pixels_low),
            "min_value_low": np.min(pixels_low),
            "stdev_low": np.std(pixels_low),
            "center_edge_ratio_low": _center_edge_ratio(lo, center_mask, edge_mask) if max_dist > 0 else np.nan,
            "mean_value_divided": mean_value_divided,
            "area_divided": area,
            "max_value_divided": np.max(pixels_divided),
            "min_value_divided": np.min(pixels_divided),
            "stdev_divided": np.std(pixels_divided),
            # --- location / QC (not classification features) ---
            "centroid_x": cx,
            "centroid_y": cy,
            "touches_border": touches_border,
        }
        # --- shape ---
        record.update(shape_features(contour, area))
        # --- raw pixel-wise ratio (high/low) statistics ---
        if ratio_px.size > 0:
            record.update({
                "ratio_px_mean": float(np.mean(ratio_px)),
                "ratio_px_median": float(np.median(ratio_px)),
                "ratio_px_std": float(np.std(ratio_px)),
                "ratio_px_p10": float(np.percentile(ratio_px, 10)),
                "ratio_px_p90": float(np.percentile(ratio_px, 90)),
                "ratio_px_skew": _skewness(ratio_px),
            })
        record["center_edge_ratio_ratio"] = (
            _center_edge_ratio(np.nan_to_num(ratio_img, nan=0.0), center_mask & np.isfinite(ratio_img),
                               edge_mask & np.isfinite(ratio_img)) if max_dist > 0 else np.nan
        )
        # --- intensity distribution / integrated intensity ---
        record.update({
            "integrated_high": float(np.sum(pixels_high, dtype=np.float64)),
            "integrated_low": float(np.sum(pixels_low, dtype=np.float64)),
            "skew_high": _skewness(pixels_high),
            "skew_low": _skewness(pixels_low),
        })
        # --- local background and contrast ---
        record.update({
            "bg_local_high": bg_local_high,
            "bg_local_low": bg_local_low,
            "contrast_high": _safe_div(mean_value_high, bg_local_high),
            "contrast_low": _safe_div(mean_value_low, bg_local_low),
            "snr_high": _safe_div(mean_value_high - bg_local_high, bg_local_std_high),
        })
        records.append(record)
        mean_values_divided.append(mean_value_divided)
        particle_id += 1

    # --- spatial context ---
    nn_dist, n_neighbors = neighbor_features(centroids)
    for rec, d, k in zip(records, nn_dist, n_neighbors):
        rec["nn_distance"] = d
        rec["n_neighbors_50px"] = int(k)

    tifffile.imwrite(contourname, thecontours, imagej=True)

    mean_values_df = pd.DataFrame(records)
    return mean_values_df, areas_high, areas_low, mean_values_divided


def main():
    with open ("settings.yaml") as settingfile:
        settings = yaml.safe_load(settingfile)
        particle_size_threshold = settings['particle_size_threshold']
        crop_divided_center = parse_bool(settings.get("crop_divided_center", True), default=True)

    env_crop_divided = os.environ.get("IMAGEANALYSIS_CROP_DIVIDED")
    if env_crop_divided is not None:
        crop_divided_center = parse_bool(env_crop_divided, default=crop_divided_center)

    print(f"Center crop on divided image enabled: {crop_divided_center}")
    if len(sys.argv)!=4:
        print(f'Number of arguments expected: 4. Number of arguments provided {len(sys.argv)}')
        images_path = input('Images path: ')
        roipath = input('ROI path: ')
        output_path = input('Output path: ')
    else:
        images_path = sys.argv[1]
        roipath = sys.argv[2]
        output_path = sys.argv[3]
    roi_high, roi_low = roi_selection(roipath)
    images = image_listing(images_path)
    particle_data = []
    mean_values_from_divided = []
    areas = []
    for image_name, image in images.items():
        try:
            aligned_high, cropped_low, divided = process_tif(
                image_name,
                image,
                output_path,
                roi_high,
                roi_low,
                crop_divided_center=crop_divided_center,
            )
            mean_merged, areas_high, areas_low, mean_divided = calculate_mean_value_and_areas(
                divided,
                aligned_high,
                cropped_low,
                image_name,
                output_path,
                particle_size_threshold,
            )
            particle_data.append(mean_merged)
            mean_values_from_divided.extend(mean_divided)
            areas.extend(areas_high)
        except (ValueError, RuntimeError, cv2.error) as e:
            print(f"Skipping image {image_name}: {e}")
    avg_area = float(np.mean(areas)) if areas else float('nan')
    print(f'Areas calculated: {len(areas)}, {avg_area}')
    
    # Save particle data (high/low measurements)
    if particle_data:
        particle_data_df = pd.concat(particle_data, ignore_index=True)
    else:
        particle_data_df = pd.DataFrame(
            columns=[
                "image",
                "particle_id",
                "mean_value_high",
                "area_high",
                "mean_value_low",
                "area_low",
            ]
        )
    particle_data_df.to_csv(f'{output_path}/particle_data.csv', index=False)
    
    # Save mean values from divided image
    with open(f'{output_path}/mean_values.csv', 'w') as f:
        for value in mean_values_from_divided:
            f.write(str(value))
            f.write(';')
    with open(f'{output_path}/areas.csv', 'w') as f:
        for value in areas:
            f.write(str(value))
            f.write(';')
                           

if __name__ == "__main__":
    main()