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


if __name__ == "__main__":
    unittest.main()
