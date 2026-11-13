#!/bin/bash

# Usage:
#   bash run_imageanalysis.sh [ROOT_DIR] [MAIN_ROI_ZIP] [CROP_DIVIDED_CENTER]
# Example:
#   bash run_imageanalysis.sh /Volumes/Untitled/Preps_Repeat_Machine_Learning
# If MAIN_ROI_ZIP is omitted, the script looks for RoiSet.zip in ROOT_DIR.
# CROP_DIVIDED_CENTER optional values: true/false (overrides settings.yaml for this run).
# Set JOBS=<n> to process n folders in parallel (default 1), e.g.
#   JOBS=8 bash run_imageanalysis.sh /path/to/data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv311/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON_BIN="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

ROOT_DIR="${1:-/Volumes/Untitled/Preps_Repeat_Machine_Learning}"
MAIN_ROI_ZIP="${2:-}"
CROP_DIVIDED_CENTER="${3:-}"
JOBS="${JOBS:-1}"

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Root directory does not exist: $ROOT_DIR"
    exit 1
fi

if [[ -z "$MAIN_ROI_ZIP" ]]; then
    MAIN_ROI_ZIP="$(find "$ROOT_DIR" -maxdepth 1 -type f \( -iname "RoiSet.zip" -o -iname "*_RoiSet.zip" \) -print -quit)"
fi

if [[ -z "$MAIN_ROI_ZIP" || ! -f "$MAIN_ROI_ZIP" ]]; then
    echo "Could not find main ROI zip."
    echo "Pass it explicitly as second argument, e.g.:"
    echo "  bash run_imageanalysis.sh \"$ROOT_DIR\" \"$ROOT_DIR/RoiSet.zip\""
    exit 1
fi

if [[ -n "$CROP_DIVIDED_CENTER" ]]; then
    # tr instead of ${VAR,,}: works with macOS's bash 3.2 as well as Linux bash.
    case "$(printf '%s' "$CROP_DIVIDED_CENTER" | tr '[:upper:]' '[:lower:]')" in
        true|false|1|0|yes|no|on|off|y|n) ;;
        *)
            echo "Invalid CROP_DIVIDED_CENTER value: $CROP_DIVIDED_CENTER"
            echo "Use true/false (or 1/0, yes/no, on/off)."
            exit 1
            ;;
    esac
fi

if ! "$PYTHON_BIN" -c "import cv2, numpy, pandas, tifffile, read_roi, pystackreg" >/dev/null 2>&1; then
    echo "Missing Python dependencies in: $PYTHON_BIN"
    echo "Install into this interpreter, then rerun."
    exit 1
fi

# Imageanalysis_ml.py reads settings.yaml from the current directory.
cd "$SCRIPT_DIR" || exit 1

process_folder() {
    local folder="$1"
    local ROI_PATH
    ROI_PATH="$(find "$folder" -maxdepth 1 -type f \( -iname "RoiSet.zip" -o -iname "*_RoiSet.zip" \) -print -quit)"
    if [[ -z "$ROI_PATH" ]]; then
        ROI_PATH="$MAIN_ROI_ZIP"
    fi

    local OUT_PATH="$folder/out_ml_full"
    mkdir -p "$OUT_PATH"
    echo "Processing folder: $folder"
    echo "Using ROI: $ROI_PATH"
    if [[ -n "$CROP_DIVIDED_CENTER" ]]; then
        echo "Override center crop on divided image: $CROP_DIVIDED_CENTER"
        IMAGEANALYSIS_CROP_DIVIDED="$CROP_DIVIDED_CENTER" \
            "$PYTHON_BIN" "$SCRIPT_DIR/Imageanalysis_ml.py" "$folder" "$ROI_PATH" "$OUT_PATH"
    else
        "$PYTHON_BIN" "$SCRIPT_DIR/Imageanalysis_ml.py" "$folder" "$ROI_PATH" "$OUT_PATH"
    fi
    echo "Finished folder: $folder (exit $?)"
}
export -f process_folder
export PYTHON_BIN SCRIPT_DIR MAIN_ROI_ZIP CROP_DIVIDED_CENTER

# The folder list is complete (sort waits for find) before any processing starts,
# so newly written output folders are never picked up as inputs.
find "$ROOT_DIR" \
    -mindepth 2 \
    -type f \
    \( -iname "*.tif" -o -iname "*.tiff" \) \
    ! -name "._*" \
    ! -path "*/out_ml_nob/*" \
    ! -path "*/out_ml/*" \
    ! -path "*/out_ml_full/*" \
    ! -iname "*_low.tif" \
    ! -iname "*_high.tif" \
    ! -iname "*_stack.tif" \
    ! -iname "*_divided.tif" \
    ! -iname "*_divided_non_aligned.tif" \
    ! -iname "*_contoured.tif" \
    -exec dirname {} \; | sort -u | tr '\n' '\0' | \
    xargs -0 -n 1 -P "$JOBS" bash -c 'process_folder "$1"' _
