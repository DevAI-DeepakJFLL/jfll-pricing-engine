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

import json

_init_err = None
_init_tb = None
_wsgi_handler = None

try:
    from apig_wsgi import make_lambda_handler
    from src.app import app
    _wsgi_handler = make_lambda_handler(app, binary_support=True)
except BaseException as err:
    import traceback
    _init_err = str(err)
    _init_tb = traceback.format_exc()
    print(f"FATAL: Error during lambda module initialization: {_init_err}\n{_init_tb}")
    _wsgi_handler = None


def handler(event, context):
    if _wsgi_handler is None:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Service Unavailable: Application initialization failed."})
        }
    try:
        return _wsgi_handler(event, context)
    except Exception as exc:
        import traceback
        print(f"FATAL: Unhandled exception in Lambda invocation: {exc}")
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Internal Server Error"})
        }


