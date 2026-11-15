#!/bin/bash
# Full pipeline: image analysis on every folder under DATA_ROOT, then classifier training.
# Runs the same way on an AWS instance and locally.
#
# Usage:
#   bash run_pipeline.sh DATA_ROOT RESULTS_DIR [classification.py options...]
# Example:
#   JOBS=8 bash run_pipeline.sh /Volumes/Untitled/Preps_Repeat_Machine_Learning ./results --model hgb
#
# strain_names_and_paths.yaml lists absolute paths on the Mac. Every occurrence of
# LOCAL_DATA_ROOT (default below) is rewritten to DATA_ROOT for this run.
# Set SKIP_IMAGE_ANALYSIS=1 to reuse existing out_ml_full/particle_data.csv files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:?Usage: run_pipeline.sh DATA_ROOT RESULTS_DIR [classifier options]}"
RESULTS_DIR="${2:?Usage: run_pipeline.sh DATA_ROOT RESULTS_DIR [classifier options]}"
shift 2
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/Volumes/Untitled/Preps_Repeat_Machine_Learning}"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

if command -v nproc >/dev/null 2>&1; then CPUS="$(nproc)"; else CPUS="$(sysctl -n hw.ncpu)"; fi
export JOBS="${JOBS:-$CPUS}"

if [[ -x "$SCRIPT_DIR/.venv311/bin/python" ]]; then PY="$SCRIPT_DIR/.venv311/bin/python"; else PY=python3; fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ "${SKIP_IMAGE_ANALYSIS:-0}" != "1" ]]; then
    log "Image analysis: $DATA_ROOT ($JOBS folders in parallel)"
    bash "$SCRIPT_DIR/run_imageanalysis.sh" "$DATA_ROOT" "$DATA_ROOT/RoiSet.zip" 2>&1 \
        | { grep -v "^Warning: ROI exceeds" || true; } | tee "$RESULTS_DIR/imageanalysis.log"
fi

log "Writing dataset config with paths under $DATA_ROOT"
sed "s#${LOCAL_DATA_ROOT}#${DATA_ROOT}#g" "$SCRIPT_DIR/strain_names_and_paths.yaml" > "$RESULTS_DIR/strain_names_and_paths.yaml"

# Keep a copy of every particle table next to the results.
(cd "$DATA_ROOT" && find . -path "*/out_ml_full/particle_data.csv" -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$RESULTS_DIR/particle_data/$(dirname "$f")"
    cp "$f" "$RESULTS_DIR/particle_data/$f"
done)

log "Training classifier: classification.py $*"
cd "$SCRIPT_DIR"
"$PY" classification.py --yaml-path "$RESULTS_DIR/strain_names_and_paths.yaml" \
    --output-dir "$RESULTS_DIR/classification" "$@" 2>&1 \
    | tee "$RESULTS_DIR/classification.log"
log "Done. Results in $RESULTS_DIR"
