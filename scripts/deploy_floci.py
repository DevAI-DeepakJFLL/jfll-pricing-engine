#!/usr/bin/env python3
"""
================================================================================
Floci Cloud Resource Provisioning & Deployment Script
================================================================================
Provisions AWS resources on the local Floci emulator (http://localhost:4566):
- Amazon S3: Initializes 'air-export-pricing-models' bucket and uploads ML bundle
- AWS SSM: Sets engine configuration parameters in Parameter Store
- Validates cloud resources and outputs deployment confirmation
================================================================================
"""

import os
import sys
import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("floci_deploy")

try:
    import aws_floci
except ImportError as e:
    logger.error(f"Failed to import aws_floci: {e}")
    sys.exit(1)


def main():
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
    bucket_name = os.environ.get("S3_BUCKET_NAME", "air-export-pricing-models")
    model_path = os.path.join(PROJECT_ROOT, "models", "freight_margin_recommender.joblib")

    print("\n" + "=" * 70)
    print("AIR EXPORT PRICING ENGINE — FLOCI CLOUD DEPLOYMENT")
    print("=" * 70)
    print(f"Target Floci Endpoint : {endpoint}")
    print(f"S3 Model Bucket       : {bucket_name}")
    print(f"Local Model Bundle    : {model_path}")
    print("=" * 70 + "\n")

    # 1. Check Floci health
    logger.info("Step 1: Checking Floci status...")
    if not aws_floci.is_floci_healthy(endpoint):
        logger.error(f"Floci is not reachable at {endpoint}. Please verify Docker container 'floci' is running.")
        sys.exit(1)
    logger.info("[SUCCESS] Floci emulator is healthy and responding.")

    # 2. Verify model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model artifact not found at {model_path}. Please run './run.sh train' first.")
        sys.exit(1)

    # 3. Provision S3 Bucket and Upload Model
    logger.info(f"Step 2: Provisioning Amazon S3 bucket '{bucket_name}' in Floci...")
    s3_uploaded_v24 = aws_floci.upload_model_to_s3(
        local_path=model_path,
        bucket=bucket_name,
        s3_key="v2.4/freight_margin_recommender.joblib",
        endpoint_url=endpoint
    )
    s3_uploaded_latest = aws_floci.upload_model_to_s3(
        local_path=model_path,
        bucket=bucket_name,
        s3_key="latest/freight_margin_recommender.joblib",
        endpoint_url=endpoint
    )

    if not (s3_uploaded_v24 and s3_uploaded_latest):
        logger.error("Failed to upload model artifacts to Floci S3.")
        sys.exit(1)
    logger.info("[SUCCESS] Model artifacts successfully published to Floci S3!")

    # 4. Provision SSM Parameters
    logger.info("Step 3: Configuring AWS SSM Parameter Store parameters...")
    ssm_params = {
        "/air-export-pricing/model-version": "v2.4",
        "/air-export-pricing/s3-bucket": bucket_name,
        "/air-export-pricing/s3-key": "v2.4/freight_margin_recommender.joblib",
        "/air-export-pricing/environment": "floci-local",
        "/air-export-pricing/status": "ACTIVE"
    }

    for key, val in ssm_params.items():
        ok = aws_floci.set_ssm_parameter(key, val, endpoint_url=endpoint)
        if ok:
            logger.info(f"  [OK] SSM Parameter: {key} = {val}")
        else:
            logger.warning(f"  [FAIL] Could not set parameter: {key}")

    # 5. Verify Resources in Floci
    logger.info("\nStep 4: Validating cloud resources in Floci...")
    s3_client = aws_floci.get_boto3_client("s3", endpoint_url=endpoint)
    s3_objects = s3_client.list_objects_v2(Bucket=bucket_name)
    uploaded_keys = [obj["Key"] for obj in s3_objects.get("Contents", [])]
    logger.info(f"Verified S3 objects in '{bucket_name}': {uploaded_keys}")

    version_param = aws_floci.get_ssm_parameter("/air-export-pricing/model-version", endpoint_url=endpoint)
    logger.info(f"Verified SSM /air-export-pricing/model-version: {version_param}")

    print("\n" + "=" * 70)
    print("FLOCI CLOUD RESOURCES SUCCESSFULLY PROVISIONED!")
    print("=" * 70)
    print(f"S3 Artifact URI   : s3://{bucket_name}/v2.4/freight_margin_recommender.joblib")
    print(f"SSM Config Tag    : /air-export-pricing/model-version -> {version_param}")
    print(f"Floci Console Web : http://localhost:4500")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
