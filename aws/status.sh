#!/bin/bash
# Show the state of a run. With no RUN_ID, lists all runs.
# Usage: bash aws/status.sh [RUN_ID] [LOG_LINES]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -eq 0 ]]; then
    echo "Runs in $S3/runs/:"
    aws s3 ls "$S3/runs/" | awk '{print "  " $2}' | tr -d /
    exit 0
fi

RUN_URI="$S3/runs/$1"
LINES="${2:-25}"
echo "Status:   $(aws s3 cp "$RUN_URI/STATUS" - 2>/dev/null || echo 'starting (no status yet)')"
if INSTANCE_ID="$(aws s3 cp "$RUN_URI/INSTANCE_ID" - 2>/dev/null)"; then
    STATE="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo 'gone')"
    echo "Instance: $INSTANCE_ID ($STATE)"
fi
echo "--- last $LINES log lines (uploaded every 2 min) ---"
aws s3 cp "$RUN_URI/run.log" - 2>/dev/null | tail -n "$LINES" || echo "(no log yet: the instance needs a few minutes to boot and install)"
