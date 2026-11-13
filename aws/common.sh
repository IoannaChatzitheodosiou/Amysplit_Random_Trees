#!/bin/bash
# Sourced by the other aws/*.sh scripts.
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$AWS_DIR/.." && pwd)"
# shellcheck source=config.env
source "$AWS_DIR/config.env"
export AWS_REGION AWS_PROFILE
export AWS_DEFAULT_REGION="$AWS_REGION"

if ! command -v aws >/dev/null 2>&1; then
    echo "The AWS CLI is not installed. See aws/README.md (step 2)." >&2
    exit 1
fi

if ! ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"; then
    echo "AWS CLI is not signed in for profile '$AWS_PROFILE'." >&2
    echo "Run: aws sso login --profile $AWS_PROFILE   (or see aws/README.md step 3)" >&2
    exit 1
fi

BUCKET="${BUCKET_PREFIX}-${ACCOUNT_ID}-${AWS_REGION}"
S3="s3://${BUCKET}"
