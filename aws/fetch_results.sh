#!/bin/bash
# Download a run's results (classification log, particle tables, logs) to aws_results/RUN_ID.
# Usage: bash aws/fetch_results.sh RUN_ID
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="${1:?Usage: fetch_results.sh RUN_ID (list runs with: bash aws/status.sh)}"
DEST="$REPO_DIR/aws_results/$RUN_ID"
mkdir -p "$DEST"
aws s3 sync "$S3/runs/$RUN_ID/results" "$DEST" --only-show-errors
aws s3 cp "$S3/runs/$RUN_ID/run.log" "$DEST/run.log" --only-show-errors 2>/dev/null || true
echo "Saved to $DEST"
ls "$DEST"
