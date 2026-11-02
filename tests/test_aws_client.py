"""
Unit tests for AWS integration module (aws_client.py).
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import aws_client


class TestAwsClient(unittest.TestCase):
    def test_get_boto3_client_defaults(self):
        client = aws_client.get_boto3_client("s3")
        self.assertIsNotNone(client)
        self.assertEqual(client.meta.region_name, "ap-south-1")

    def test_get_ssm_parameter_fallback(self):
        with patch.object(aws_client, "get_boto3_client") as mock_get_client:
            mock_ssm = MagicMock()
            mock_ssm.get_parameter.side_effect = Exception("SSM Error")
            mock_get_client.return_value = mock_ssm

            val = aws_client.get_ssm_parameter("/test/param", default="v3.0")
            self.assertEqual(val, "v3.0")

    def test_sync_model_skips_when_file_exists(self):
        with patch.object(aws_client, "get_boto3_client") as mock_get_client:
            # Existing file (models/engine_v3_bundle.joblib)
            bundle_path = os.path.join(BASE_DIR, "models", "engine_v3_bundle.joblib")
            result = aws_client.sync_model_from_s3(local_dest=bundle_path, force=False)
            self.assertTrue(result)
            mock_get_client.assert_not_called()

    def test_get_boto3_client_timeout_config(self):
        client = aws_client.get_boto3_client("s3")
        self.assertIsNotNone(client)
        cfg = getattr(client.meta, "config", None)
        if cfg:
            self.assertEqual(cfg.connect_timeout, 3)
            self.assertEqual(cfg.read_timeout, 5)


if __name__ == "__main__":
    unittest.main()
