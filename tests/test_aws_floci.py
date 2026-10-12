"""
Unit tests for AWS & Floci integration module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import aws_floci

class TestAwsFloci(unittest.TestCase):
    def test_get_boto3_client_defaults(self):
        client = aws_floci.get_boto3_client("s3", endpoint_url="http://localhost:4566")
        self.assertIsNotNone(client)
        self.assertEqual(client.meta.region_name, "us-east-1")

    def test_is_floci_healthy_mock(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b'{"version":"2.2.0","services":{"s3":"running"}}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            
            healthy = aws_floci.is_floci_healthy("http://localhost:4566")
            self.assertTrue(healthy)

    def test_is_floci_healthy_failure(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            healthy = aws_floci.is_floci_healthy("http://localhost:9999")
            self.assertFalse(healthy)

    def test_get_ssm_parameter_fallback(self):
        with patch.object(aws_floci, "get_boto3_client") as mock_get_client:
            mock_ssm = MagicMock()
            mock_ssm.get_parameter.side_effect = Exception("SSM Error")
            mock_get_client.return_value = mock_ssm
            
            val = aws_floci.get_ssm_parameter("/test/param", default="v2.4")
            self.assertEqual(val, "v2.4")

if __name__ == "__main__":
    unittest.main()
