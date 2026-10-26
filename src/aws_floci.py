"""
================================================================================
AWS & Floci Cloud Integration Module
================================================================================
Provides client initialization, model synchronization with Amazon S3, and
parameter management via SSM Parameter Store for the Air Export Pricing Engine.
Supports Floci local emulation (endpoint: http://localhost:4566) and production AWS.
================================================================================
"""

import os
import logging
import json
import urllib.request
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception

logger = logging.getLogger("aws_floci")

DEFAULT_FLOCI_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET_NAME", "air-export-pricing-models")


def is_floci_healthy(endpoint_url: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Checks if Floci or LocalStack is responsive."""
    endpoint = endpoint_url or DEFAULT_FLOCI_ENDPOINT
    health_url = f"{endpoint.rstrip('/')}/_localstack/health"
    if not (health_url.startswith("http://") or health_url.startswith("https://")):
        logger.warning(f"Invalid health URL scheme: {health_url}")
        return False
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "pricing-engine-health"})
        with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310
            if response.status == 200:
                data = json.loads(response.read().decode())
                return "services" in data or "version" in data
    except Exception as e:
        logger.debug(f"Floci health check at {health_url} failed: {e}")
    return False


def get_boto3_client(service_name: str, endpoint_url: Optional[str] = None):
    """
    Constructs a boto3 client pointing to the Floci endpoint (or AWS if unset).
    """
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 is not installed. Please install boto3 to use AWS/Floci features.")

    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL", DEFAULT_FLOCI_ENDPOINT)
    region = os.environ.get("AWS_REGION", DEFAULT_REGION)
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "test")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")

    return boto3.client(
        service_name,
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )


def sync_model_from_s3(
    bucket: str = DEFAULT_BUCKET,
    s3_key: str = "v2.4/freight_margin_recommender.joblib",
    local_dest: str = "models/freight_margin_recommender.joblib",
    endpoint_url: Optional[str] = None
) -> bool:
    """
    Downloads model bundle from S3 if present.
    Returns True if successfully retrieved, False otherwise.
    """
    try:
        s3 = get_boto3_client("s3", endpoint_url=endpoint_url)
        os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
        logger.info(f"Downloading model bundle from s3://{bucket}/{s3_key} to {local_dest}...")
        s3.download_file(bucket, s3_key, local_dest)
        logger.info("Model download from S3 succeeded.")
        return True
    except Exception as e:
        logger.warning(f"Failed to sync model from S3 (s3://{bucket}/{s3_key}): {e}. Using local fallback.")
        return False


def upload_model_to_s3(
    local_path: str,
    bucket: str = DEFAULT_BUCKET,
    s3_key: str = "v2.4/freight_margin_recommender.joblib",
    endpoint_url: Optional[str] = None
) -> bool:
    """
    Uploads local model artifact to S3 bucket in Floci or AWS.
    Creates bucket if it does not exist.
    """
    try:
        s3 = get_boto3_client("s3", endpoint_url=endpoint_url)
        # Ensure bucket exists
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            logger.info(f"Creating bucket {bucket}...")
            s3.create_bucket(Bucket=bucket)

        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}...")
        s3.upload_file(local_path, bucket, s3_key)
        logger.info(f"Upload complete: s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading model to S3: {e}")
        return False


def get_ssm_parameter(param_name: str, default: Optional[str] = None, endpoint_url: Optional[str] = None) -> Optional[str]:
    """Retrieves SSM parameter with safe fallback."""
    try:
        ssm = get_boto3_client("ssm", endpoint_url=endpoint_url)
        resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception as e:
        logger.debug(f"Could not fetch SSM parameter {param_name}: {e}. Returning default: {default}")
        return default


def set_ssm_parameter(param_name: str, value: str, param_type: str = "String", endpoint_url: Optional[str] = None) -> bool:
    """Sets SSM parameter in Floci/AWS."""
    try:
        ssm = get_boto3_client("ssm", endpoint_url=endpoint_url)
        ssm.put_parameter(Name=param_name, Value=value, Type=param_type, Overwrite=True)
        return True
    except Exception as e:
        logger.error(f"Error putting SSM parameter {param_name}: {e}")
        return False
