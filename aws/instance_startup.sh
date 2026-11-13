#!/bin/bash
# Runs on the EC2 instance at boot (EC2 "user data"). launch.sh fills in the __PLACEHOLDERS__.
# The instance installs Python, downloads code and data from S3, runs the pipeline,
# uploads results, then shuts down. It was launched with shutdown behaviour "terminate",
# so shutting down also deletes it and billing stops.
set -euo pipefail

RUN_URI="__RUN_URI__"
DATA_URI="__DATA_URI__"
MAX_HOURS="__MAX_HOURS__"
LOCAL_DATA_ROOT="__LOCAL_DATA_ROOT__"
CLASSIFIER_ARGS=(__CLASSIFIER_ARGS__)

LOG=/var/log/optosplit.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

status() { echo "$1" | aws s3 cp - "$RUN_URI/STATUS" --only-show-errors || true; }

finish() {
    code=$?
    if [[ $code -eq 0 ]]; then status "succeeded"; else status "failed (exit $code)"; fi
    log "Finished with exit code $code; uploading results and log, then shutting down."
    [[ -d /results ]] && aws s3 sync /results "$RUN_URI/results" --only-show-errors || true
    aws s3 cp "$LOG" "$RUN_URI/run.log" --only-show-errors || true
    shutdown -h now
}
trap finish EXIT

# Safety net: the pipeline is stopped after MAX_HOURS (so partial results still get uploaded),
# and the machine is forced off 15 minutes later no matter what.
shutdown -h "+$((MAX_HOURS * 60 + 15))"

status "running"
# Upload the log every 2 minutes so progress can be followed with aws/status.sh.
( while sleep 120; do aws s3 cp "$LOG" "$RUN_URI/run.log" --only-show-errors || true; done ) &

log "Installing Python 3.11"
dnf install -y -q python3.11 python3.11-pip tar gzip

log "Downloading code"
mkdir -p /opt/optosplit && cd /opt/optosplit
aws s3 cp "$RUN_URI/code.tar.gz" - | tar xz
python3.11 -m venv .venv311
.venv311/bin/pip install -q --upgrade pip
.venv311/bin/pip install -q -r requirements.txt

log "Downloading data from $DATA_URI"
DATA_ROOT=/data/$(basename "$LOCAL_DATA_ROOT")
mkdir -p "$DATA_ROOT"
aws s3 sync "$DATA_URI" "$DATA_ROOT" --only-show-errors
log "Data: $(du -sh "$DATA_ROOT" | cut -f1), $(find "$DATA_ROOT" -iname '*.tif*' | wc -l) TIFFs"

log "Running pipeline on $(nproc) CPUs"
LOCAL_DATA_ROOT="$LOCAL_DATA_ROOT" timeout "${MAX_HOURS}h" \
    bash run_pipeline.sh "$DATA_ROOT" /results "${CLASSIFIER_ARGS[@]}"
