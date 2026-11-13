# Running the pipeline on AWS

One EC2 machine runs the whole pipeline: image analysis on every folder, then `classification.py`. Data, code and results go through a private S3 bucket. The machine deletes itself when it finishes, so you only pay while it runs.

```
your Mac ──upload_data.sh──▶ S3 bucket ◀──results── EC2 machine (16 CPUs)
         ◀─fetch_results.sh─            ──data/code─▶  runs run_pipeline.sh, then shuts down
```

## One-time setup

1. **Create an AWS account** at https://aws.amazon.com (needs a credit card). Then, in the console under *Billing → Budgets*, create a monthly cost budget (e.g. $20) with an email alert.

2. **Install the AWS CLI** on the Mac: `brew install awscli`, or the installer from https://aws.amazon.com/cli/. Check it with `aws --version`.

3. **Let the CLI sign in.** Don't use the root account's keys. The simplest safe route:
   - In the console, open *IAM Identity Center*, enable it, and create a user for yourself with the *AdministratorAccess* permission set on your account.
   - Then run `aws configure sso` on the Mac. Name the profile `default`, or set `AWS_PROFILE` in `aws/config.env` to whatever name you choose.
   - Each day you use it, sign in with `aws sso login`.
   - Check it worked: `aws sts get-caller-identity`.

4. **Edit `aws/config.env`.** At minimum, set `AWS_REGION` to the region closest to you.

5. **Allow 16-CPU machines.** New accounts often have a low limit on how many CPUs can run at once. Go to *Service Quotas → Amazon EC2 → "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances"* and request at least 16 vCPUs in your region. This can take a few hours to a day. Until then, you can set `INSTANCE_TYPE="m7i.xlarge"` (4 CPUs, slower).

6. **Create the bucket and permissions** (safe to re-run):
   ```bash
   bash aws/setup.sh
   ```

## Each run

```bash
bash aws/upload_data.sh          # first time ~7 GB; later runs only upload new/changed TIFFs (drive must be mounted)
bash aws/launch.sh               # starts the machine; prints a RUN_ID
bash aws/status.sh RUN_ID        # status + latest log lines (log refreshes every 2 min)
bash aws/fetch_results.sh RUN_ID # downloads results to aws_results/RUN_ID/
```

Any options after `launch.sh` go to `classification.py`, e.g. `bash aws/launch.sh --model hgb --search-iters 50`.

The code sent is whatever is in this folder when you run `launch.sh`, including `settings.yaml` and `strain_names_and_paths.yaml`. It does not need to be committed. Paths in `strain_names_and_paths.yaml` that start with `LOCAL_DATA_ROOT` are rewritten to the machine's copy of the data automatically.

`aws/status.sh` with no RUN_ID lists all runs.

**Results** (`aws_results/RUN_ID/`): `classification.log` (all scores and reports), `imageanalysis.log`, `particle_data/<strain>/<sample>/out_ml_full/particle_data.csv`, the dataset config used, and `run.log` (the machine's full log). The intermediate TIFFs stay on the machine and are deleted with it.

## Cost and time (approximate; check current prices for your region)

- **m7i.4xlarge:** roughly $0.8–1 per hour while running. Startup and installing packages take about 5 min. Image analysis takes roughly 15 min (folders run in parallel, so the largest folder sets the pace).
- **Classifier:** `--model hgb` takes minutes. The default RandomForest search (30 candidates × 5 folds × up to 1200 trees on ~800k particles) can take hours.
- **Time limit:** a run is stopped after `MAX_HOURS` (default 6). Whatever results exist are still uploaded.
- **S3 storage:** for ~7 GB, cents per month.

## Troubleshooting

- **`VcpuLimitExceeded` from `launch.sh`:** see step 5.
- **Status stays "running" with no new log lines:** the log uploads every 2 min. For a live terminal on the machine, use *EC2 console → the instance → Connect → Session Manager*.
- **Stop a run early:** `aws ec2 terminate-instances --instance-ids <id from status.sh>`.
- **Remove everything:**
  - Delete the data, runs and bucket: `aws s3 rb s3://<bucket> --force` (the bucket name is printed by `setup.sh`).
  - Delete the IAM role named in `config.env`: remove its instance profile, policies and the role itself in the IAM console.
