#!/bin/bash
# Start one EC2 instance that runs the full pipeline and then deletes itself.
# Any arguments are passed on to classification.py.
#
# Usage:
#   bash aws/launch.sh                       # defaults: RandomForest, manual split
#   bash aws/launch.sh --model hgb           # faster model
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="$(date '+%Y%m%d-%H%M%S')"
RUN_URI="$S3/runs/$RUN_ID"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Run $RUN_ID"

# --- package the code (scripts + configs only, no data or virtualenv) ---
(cd "$REPO_DIR" && COPYFILE_DISABLE=1 tar czf "$WORK/code.tar.gz" \
    *.py *.yaml requirements.txt run_imageanalysis.sh run_pipeline.sh)
aws s3 cp "$WORK/code.tar.gz" "$RUN_URI/code.tar.gz" --only-show-errors

# --- fill in the startup script ---
CLASSIFIER_ARGS=""
for a in "$@"; do CLASSIFIER_ARGS+="$(printf '%q ' "$a")"; done
sed -e "s#__RUN_URI__#${RUN_URI}#" \
    -e "s#__DATA_URI__#${S3}/data/#" \
    -e "s#__MAX_HOURS__#${MAX_HOURS}#" \
    -e "s#__LOCAL_DATA_ROOT__#${LOCAL_DATA_ROOT}#" \
    -e "s#__CLASSIFIER_ARGS__#${CLASSIFIER_ARGS}#" \
    "$AWS_DIR/instance_startup.sh" > "$WORK/user_data.sh"

# --- latest Amazon Linux 2023 image ---
AMI_ID="$(aws ssm get-parameter \
    --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query Parameter.Value --output text)"

INSTANCE_ID="$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=$ROLE_NAME" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --metadata-options "HttpTokens=required" \
    --user-data "file://$WORK/user_data.sh" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=optosplit-$RUN_ID},{Key=project,Value=optosplit}]" \
    --query 'Instances[0].InstanceId' --output text)"

echo "$INSTANCE_ID" | aws s3 cp - "$RUN_URI/INSTANCE_ID" --only-show-errors
echo "Started instance $INSTANCE_ID ($INSTANCE_TYPE in $AWS_REGION)."
echo "It shuts itself down when finished, or after $MAX_HOURS h at most."
echo
echo "Follow progress:   bash aws/status.sh $RUN_ID"
echo "Download results:  bash aws/fetch_results.sh $RUN_ID"
