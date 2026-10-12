# AWS Free Tier Deployment Guide — Air Export Pricing Engine

Comprehensive, step-by-step walkthrough for deploying the **Air Export Pricing Engine** to AWS under the **AWS Free Tier** using **Amazon S3**, **AWS Lambda (Function URL)**, optional **Amazon CloudFront CDN**, and automated **GitHub Actions CI/CD**.

---

## Table of Contents

1. [Architecture & Free Tier Budget Strategy](#1-architecture--free-tier-budget-strategy)
2. [Step 1: Amazon S3 Bucket Setup](#step-1-amazon-s3-bucket-setup)
3. [Step 2: AWS Lambda Execution IAM Role](#step-2-aws-lambda-execution-iam-role)
4. [Step 3: AWS Lambda Serverless Function](#step-3-aws-lambda-serverless-function)
5. [Step 4: Optional CloudFront CDN Distribution](#step-4-optional-cloudfront-cdn-distribution)
6. [Step 5: IAM Deployer User for GitHub Actions](#step-5-iam-deployer-user-for-github-actions)
7. [Step 6: GitHub Actions Secrets Configuration](#step-6-github-actions-secrets-configuration)
8. [Step 7: Automated Deployment Workflow](#step-7-automated-deployment-workflow)
9. [Step 8: Verification & Live Health Checks](#step-8-verification--live-health-checks)

---

## 1. Architecture & Free Tier Budget Strategy

### System Architecture Overview

```
                      +-----------------------------+
                      |   Pricing Analyst / User    |
                      +--------------+--------------+
                                     |
                                     v
                        HTTPS Requests (Web UI / API)
                                     |
               +---------------------+---------------------+
               | (Direct Lambda URL)                       | (Optional CDN)
               v                                           v
    +-----------------------+                    +--------------------+
    |  AWS Lambda Function  | <----------------- |  Amazon CloudFront |
    |      (Python 3.11)    |   Origin (HTTPS)   |    (Global Edge)   |
    |  Flask + apig-wsgi    |                    +--------------------+
    +-----------+-----------+
                |
                | Reads Models & Syncs Backups
                v
    +-----------------------+
    |   Amazon S3 Bucket    |
    | (Artifacts & Builds)  |
    +-----------------------+
```

### Free Tier Allowance & Safety Matrix

| AWS Service | Monthly Free Tier Allowance | Pricing Engine Usage | Cost Impact |
| :--- | :--- | :--- | :--- |
| **AWS Lambda** | **1,000,000 requests/month**<br>3,200,000 compute seconds (at 128MB) | 512MB–1024MB RAM, ~300ms execution per quote | ✅ **100% Free** |
| **Amazon S3** | **5 GB Standard Storage**<br>20,000 GET requests, 2,000 PUT requests | ~50 MB (Model bundle `engine_v3_bundle.joblib` + zip packages) | ✅ **100% Free** |
| **Amazon CloudFront** | **1 TB Data Transfer Out**<br>10,000,000 HTTP/HTTPS requests | ~500 MB data transfer/month | ✅ **100% Free** |
| **AWS WAF** | **NO FREE TIER** ($5/month base + $1/rule) | Keep **Disabled** in CloudFront | ⚠️ **Never enable WAF** |

---

## Step 1: Amazon S3 Bucket Setup

You will create **one private S3 bucket** to store the Lambda zip builds, machine learning model bundles, and processed datasets.

1. Open the [Amazon S3 Console](https://s3.console.aws.amazon.com/s3/).
2. Click **Create bucket**.
3. **Bucket name**: `jfll-pricing-engine-data` *(must be globally unique across all AWS accounts; if taken, use `jfll-pricing-engine-data-<yourname>`)*.
4. **AWS Region**: Select your preferred region (e.g. `ap-south-1` Mumbai or `ap-southeast-2` Sydney).
5. **Block Public Access settings for this bucket**:
   - Keep **Block *all* public access** checked (all 4 boxes checked).
6. **Bucket Versioning**: Keep *Disabled*.
7. **Default encryption**: Keep *Server-side encryption with Amazon S3 managed keys (SSE-S3)*.
8. Click **Create bucket**.

---

## Step 2: AWS Lambda Execution IAM Role

The Lambda function requires permissions to write CloudWatch logs and read model bundles from your S3 data bucket.

1. Open the [IAM Roles Console](https://us-east-1.console.aws.amazon.com/iam/home#/roles).
2. Click **Create role**.
3. Under **Trusted entity type**:
   - Select **AWS service**.
   - Under **Use case**, select **Lambda**.
   - Click **Next**.
4. Under **Add permissions**:
   - Search for `AWSLambdaBasicExecutionRole` and check the checkbox.
   - Click **Next**.
5. **Role details**:
   - **Role name**: `jfll-pricing-engine-lambda-role`.
   - Click **Create role**.
6. **Attach S3 Access Policy**:
   - In the Roles list, click on `jfll-pricing-engine-lambda-role`.
   - Under the **Permissions** tab, click **Add permissions** > **Create inline policy**.
   - Click the **JSON** tab and paste:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Sid": "AllowS3ArtifactsAccess",
           "Effect": "Allow",
           "Action": [
             "s3:GetObject",
             "s3:PutObject",
             "s3:ListBucket"
           ],
           "Resource": [
             "arn:aws:s3:::jfll-pricing-engine-data",
             "arn:aws:s3:::jfll-pricing-engine-data/*"
           ]
         }
       ]
     }
     ```
     *(If your bucket name differs from `jfll-pricing-engine-data`, replace the bucket name above).*
   - Click **Next**.
   - **Policy name**: `PricingEngineS3AccessPolicy`.
   - Click **Create policy**.

---

## Step 3: AWS Lambda Serverless Function

1. Open the [AWS Lambda Console](https://console.aws.amazon.com/lambda/) in your selected region.
2. Click **Create function**:
   - Select **Author from scratch**.
   - **Function name**: `jfll-pricing-engine-api`
   - **Runtime**: `Python 3.11`
   - **Architecture**: `x86_64`
   - Expand **Change default execution role**:
     - Choose **Use an existing role**.
     - Select: `jfll-pricing-engine-lambda-role`.
   - Click **Create function**.

### Configure Memory & Timeout
1. In your function page, go to the **Configuration** tab > **General configuration** > click **Edit**:
   - **Memory**: `1024 MB` *(provides fast ML inference & LightGBM/scikit-learn processing)*.
   - **Timeout**: `30 sec`.
   - **Ephemeral storage (`/tmp`)**: `512 MB` (default).
   - Click **Save**.

### Configure Environment Variables
1. Go to **Configuration** tab > **Environment variables** > click **Edit**:
   - `S3_BUCKET` = `jfll-pricing-engine-data`
   - `AWS_REGION` = `<your-selected-region>` (e.g. `ap-south-1` or `ap-southeast-2`)
   - `ALLOW_SKLEARN_MISMATCH` = `1`
2. Click **Save**.

### Configure Runtime Handler
1. In the **Code** tab, scroll down to the **Runtime settings** section and click **Edit**:
   - **Handler**: `lambda_handler.handler`
2. Click **Save**.

### Create Lambda Function URL (Public Web UI & API Endpoint)
1. Go to **Configuration** tab > **Function URL** > click **Create function URL**:
   - **Auth type**: `NONE` *(allows your browser to access the Web UI and `/api/quote` directly)*.
   - Check **Configure cross-origin resource sharing (CORS)**:
     - **Allow origin**: `*`
     - **Allow headers**: `*`
     - **Allow methods**: Check `GET`, `POST`, `OPTIONS`.
2. Click **Save**.
3. **Copy your Function URL**:
   It will look like:
   `https://abcdefg123456.lambda-url.ap-south-1.on.aws/`

---

## Step 4: Optional CloudFront CDN Distribution

If you want a custom domain or global edge caching:
1. Open the [CloudFront Console](https://console.aws.amazon.com/cloudfront/).
2. Click **Create distribution**:
   - **Origin domain**: Paste your Lambda Function URL domain (without `https://` and without trailing slash, e.g. `abcdefg123456.lambda-url.ap-south-1.on.aws`).
   - **Protocol**: Select `HTTPS only`.
   - **Viewer protocol policy**: `Redirect HTTP to HTTPS`.
   - **Allowed HTTP methods**: `GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE`.
   - **Cache policy**: Choose `Managed-CachingDisabled` (since pricing quotes are dynamic and require real-time margin calculations).
   - **Origin request policy**: Choose `Managed-AllViewerExceptHostHeader`.
   - **Web Application Firewall (WAF)**: Select **Do not enable security protections** *(critical: avoids the $5/month charge)*.
3. Click **Create distribution**. Note down your **Distribution ID** (e.g. `E1ABC234XYZ567`).

---

## Step 5: IAM Deployer User for GitHub Actions

Create dedicated deployer credentials for GitHub Actions:

1. Open the [IAM Users Console](https://us-east-1.console.aws.amazon.com/iam/home#/users).
2. Click **Create user**.
3. **User name**: `github-actions-pricing-deployer`. Click **Next**.
4. Under **Set permissions**:
   - Select **Attach policies directly**.
   - Click **Create policy** (opens in a new tab).
   - In the policy editor, switch to the **JSON** tab and paste:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Sid": "S3DeployPermissions",
           "Effect": "Allow",
           "Action": [
             "s3:PutObject",
             "s3:GetObject",
             "s3:ListBucket",
             "s3:DeleteObject"
           ],
           "Resource": [
             "arn:aws:s3:::jfll-pricing-engine-data",
             "arn:aws:s3:::jfll-pricing-engine-data/*"
           ]
         },
         {
           "Sid": "LambdaDeployPermissions",
           "Effect": "Allow",
           "Action": [
             "lambda:UpdateFunctionCode",
             "lambda:GetFunction",
             "lambda:GetFunctionConfiguration"
           ],
           "Resource": "arn:aws:lambda:*:*:function:jfll-pricing-engine-api"
         },
         {
           "Sid": "CloudFrontInvalidationPermissions",
           "Effect": "Allow",
           "Action": [
             "cloudfront:CreateInvalidation"
           ],
           "Resource": "*"
         }
       ]
     }
     ```
   - Click **Next**. Name the policy `GitHubActionsPricingDeployPolicy` and click **Create policy**.
5. Switch back to the **Create user** tab, click refresh, check `GitHubActionsPricingDeployPolicy`, and click **Next** > **Create user**.
6. Open the newly created `github-actions-pricing-deployer` user:
   - Click the **Security credentials** tab.
   - Under **Access keys**, click **Create access key**.
   - Choose **Application running outside AWS** > click **Next** > **Create access key**.
   - Copy or download:
     - `Access key ID`
     - `Secret access key`

---

## Step 6: GitHub Actions Secrets Configuration

Open your repository on GitHub:
👉 [https://github.com/DevAI-DeepakJFLL/jfll-pricing-engine/settings/secrets/actions](https://github.com/DevAI-DeepakJFLL/jfll-pricing-engine/settings/secrets/actions)

Click **New repository secret** and add the following 6 secrets:

| Secret Name | Example Value | Description |
| :--- | :--- | :--- |
| `AWS_ACCESS_KEY_ID` | `AKIAIOSFODNN7EXAMPLE` | Generated in Step 5 |
| `AWS_SECRET_ACCESS_KEY` | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | Generated in Step 5 |
| `AWS_REGION` | `ap-south-1` | The AWS region where your resources live |
| `S3_DATA_BUCKET` | `jfll-pricing-engine-data` | S3 bucket created in Step 1 |
| `LAMBDA_FUNCTION_NAME` | `jfll-pricing-engine-api` | Lambda function created in Step 3 |
| `CLOUDFRONT_DISTRIBUTION_ID` | `E1ABC234XYZ567` *(Optional)* | CloudFront Distribution ID from Step 4 |

---

## Step 7: Automated Deployment Workflow

The deployment workflow is defined in [`.github/workflows/deploy.yml`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/.github/workflows/deploy.yml).

### What it does automatically on every push to `main`:
1. **Verification Gate**:
   - Executes `scripts/verify_vocab_alignment.py` verifying all 36 dropdown options match the trained models.
   - Executes all 226 unit & integration tests (`python -m unittest discover tests`).
2. **S3 Synchronization**:
   - Syncs model artifacts (`models/`) and datasets (`data/processed/`) to `s3://jfll-pricing-engine-data/`.
3. **Serverless Packaging & Lambda Update**:
   - Bundles the Flask app, `lambda_handler.py`, LightGBM/scikit-learn dependencies, and model files into a zip package.
   - Uploads the package to S3 and updates `jfll-pricing-engine-api` code with zero downtime.

---

## Step 8: Verification & Live Health Checks

### 1. Web UI Live Verification
Open your Lambda Function URL in any web browser:
`https://<YOUR_LAMBDA_FUNCTION_URL>/`
- Verify that the authentic Air Export Pricing Engine UI renders cleanly with Outfit/Mulish typography and cyan/lime accents.
- Test one of the quick test presets (e.g. *Perishables BOM -> DXB* or *Auto Parts DEL -> FRA*).

### 2. API Quoting Test (cURL)
Run the following cURL command against your live Lambda URL:
```bash
curl -X POST https://<YOUR_LAMBDA_FUNCTION_URL>/api/quote \
  -H "Content-Type: application/json" \
  -d '{
    "buy": 125000,
    "gross_wt": 450,
    "chargeable_wt": 450,
    "origin": "Indira Gandhi International Airport",
    "dest": "Heathrow Apt/London",
    "client": "Shipper / Consignee",
    "commodity": "General Cargo",
    "frequency": 10,
    "revision": 1,
    "strategy": "balanced"
  }'
```

Expected JSON response:
```json
{
  "status": "success",
  "optimal": {
    "Strategy_Key": "balanced",
    "Strategy_Name": "Recommended Standard Price",
    "Margin_Percentage": 5.01,
    "Quoted_Sell_Price_INR": 131262.5,
    "Rate_Per_Kg": 291.69,
    "Win_Probability": 0.45,
    "Expected_Profit_INR": 2818.12
  },
  "negotiation_corridor": {
    "confidence_coverage": "90%",
    "min_margin_pct": 2.28,
    "target_margin_pct": 5.01,
    "max_margin_pct": 28.43
  }
}
```
