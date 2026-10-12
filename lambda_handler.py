"""
AWS Lambda WSGI Handler for Air Export Pricing Engine
Compatible with AWS Lambda Function URLs and API Gateway HTTP API (Payload v2.0).
"""
import os
import sys

# Ensure application source code is available in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
src_dir = os.path.join(BASE_DIR, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from apig_wsgi import make_lambda_handler
from src.app import app

# Create the WSGI lambda handler with binary support
handler = make_lambda_handler(app, binary_support=True)
