#!/bin/bash
# One-time setup: S3 bucket + an IAM role that lets the EC2 machine read/write only that bucket.
# Safe to run again; existing resources are kept.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Account $ACCOUNT_ID, region $AWS_REGION, bucket $BUCKET"

# --- S3 bucket (private) ---
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "Bucket exists: $BUCKET"
else
    if [[ "$AWS_REGION" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$BUCKET" >/dev/null
    else
        aws s3api create-bucket --bucket "$BUCKET" \
            --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
    fi
    echo "Created bucket: $BUCKET"
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# --- IAM role for the instance ---
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    echo "Role exists: $ROLE_NAME"
else
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }' >/dev/null
    echo "Created role: $ROLE_NAME"
fi

aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name optosplit-bucket-access --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:ListBucket\"], \"Resource\": \"arn:aws:s3:::${BUCKET}\"},
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\"], \"Resource\": \"arn:aws:s3:::${BUCKET}/*\"}
  ]
}"
# Lets you open a terminal on a running instance from the AWS console (Session Manager), no SSH keys needed.
aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

if ! aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME"
    echo "Created instance profile: $ROLE_NAME (IAM can take ~10 s to make it usable)"
fi

echo "Setup complete."
