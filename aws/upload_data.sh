#!/bin/bash
# Upload the raw TIFFs and RoiSet.zip to S3. Existing analysis outputs (out_ml*) and
# macOS metadata files are skipped. Re-running only uploads new or changed files.
#
# Usage: bash aws/upload_data.sh [LOCAL_DATA_ROOT]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SRC="${1:-$LOCAL_DATA_ROOT}"
if [[ ! -d "$SRC" ]]; then
    echo "Data folder not found: $SRC (is the drive mounted?)" >&2
    exit 1
fi

echo "Uploading $SRC -> $S3/data/"
aws s3 sync "$SRC" "$S3/data/" \
    --exclude "*" \
    --include "*.tif" --include "*.tiff" --include "*.TIF" --include "*.TIFF" --include "*RoiSet.zip" \
    --exclude "*/out_ml*/*" --exclude "._*" --exclude "*/._*" \
    --exclude "*_low.tif" --exclude "*_high.tif" --exclude "*_stack.tif" \
    --exclude "*_divided.tif" --exclude "*_divided_non_aligned.tif" --exclude "*_contoured.tif"

echo "Uploaded. Size in S3:"
aws s3 ls "$S3/data/" --recursive --summarize --human-readable | tail -2
