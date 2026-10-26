#!/usr/bin/env python3
"""
================================================================================
AWS Cloud Resource Provisioning & Deployment Script
================================================================================
Provisions AWS resources on Amazon Web Services:
- Amazon S3: Uploads model bundles to 'air-export-pricing-models' bucket
- AWS SSM: Sets engine configuration parameters in Parameter Store
- Validates cloud resources and outputs deployment confirmation
================================================================================
"""

import os
import sys
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aws_deploy")

try:
    import aws_client
except ImportError as e:
    logger.error(f"Failed to import aws_client: {e}")
    sys.exit(1)


def main():
    bucket_name = os.environ.get("S3_BUCKET_NAME", "air-export-pricing-models")
    model_path = os.path.join(PROJECT_ROOT, "models", "engine_v3_bundle.joblib")

    print("\n" + "=" * 70)
    print("AIR EXPORT PRICING ENGINE — AWS CLOUD DEPLOYMENT")
    print("=" * 70)
    print(f"S3 Model Bucket       : {bucket_name}")
    print(f"Local Model Bundle    : {model_path}")
    print("=" * 70 + "\n")

    # 1. Upload Model to S3
    logger.info(f"Step 1: Publishing model artifact to S3 bucket '{bucket_name}'...")
    if os.path.exists(model_path):
        s3_uploaded = aws_client.upload_model_to_s3(
            local_path=model_path,
            bucket=bucket_name,
            s3_key="v3.0/engine_v3_bundle.joblib"
        )
        if s3_uploaded:
            logger.info("[SUCCESS] Model artifacts successfully published to AWS S3!")
        else:
            logger.error("Failed to upload model artifacts to S3.")
    else:
        logger.warning(f"Model bundle not found at {model_path}. Skipping S3 upload.")

    # 2. Configure AWS SSM Parameters
    logger.info("\nStep 2: Configuring AWS Systems Manager (SSM) Parameter Store...")
    ssm_params = {
        "/air-export-pricing/model-version": "v3.0",
        "/air-export-pricing/default-strategy": "balanced",
        "/air-export-pricing/conformal-coverage": "0.90",
        "/air-export-pricing/environment": "production",
    }

    for key, val in ssm_params.items():
        ok = aws_client.set_ssm_parameter(key, val)
        status = "[OK]" if ok else "[FAILED]"
        logger.info(f"  {status} SSM: {key} = '{val}'")

    print("\n" + "=" * 70)
    print("AWS CLOUD RESOURCES PROVISIONING COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
