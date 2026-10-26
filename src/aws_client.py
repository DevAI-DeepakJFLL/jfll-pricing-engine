"""
================================================================================
AWS Cloud Integration Module
================================================================================
Provides client initialization, model synchronization with Amazon S3, and
parameter management via SSM Parameter Store for the Air Export Pricing Engine.
Supports production AWS (IAM execution roles, real S3/SSM endpoints).
================================================================================
"""

import os
import logging
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception

logger = logging.getLogger("aws_client")

DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-south-1")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET_NAME", "air-export-pricing-models")


def get_boto3_client(service_name: str, endpoint_url: Optional[str] = None):
    """
    Constructs a boto3 client pointing to standard AWS endpoints or optional custom URL.
    Uses native AWS IAM roles if explicit credentials are not provided.
    """
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 is not installed. Please install boto3 to use AWS features.")

    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")
    region = os.environ.get("AWS_REGION", DEFAULT_REGION)
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    client_kwargs = {
        "region_name": region,
    }
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key

    return boto3.client(service_name, **client_kwargs)


def sync_model_from_s3(
    bucket: str = DEFAULT_BUCKET,
    s3_key: str = "v3.0/engine_v3_bundle.joblib",
    local_dest: str = "models/engine_v3_bundle.joblib",
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
    s3_key: str = "v3.0/engine_v3_bundle.joblib",
    endpoint_url: Optional[str] = None
) -> bool:
    """
    Uploads local model artifact to S3 bucket in AWS.
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
    """Sets SSM parameter in AWS Parameter Store."""
    try:
        ssm = get_boto3_client("ssm", endpoint_url=endpoint_url)
        ssm.put_parameter(Name=param_name, Value=value, Type=param_type, Overwrite=True)
        return True
    except Exception as e:
        logger.error(f"Error putting SSM parameter {param_name}: {e}")
        return False
