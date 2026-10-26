#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "======================================================================"
echo "Air Export Pricing Recommendation Engine (v2.0)"
echo "======================================================================"

if [ -d "$DIR/.venv" ]; then
    PYTHON="$DIR/.venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
else
    PYTHON="python3"
fi

case "$1" in
    verify)
        echo "[*] Running verification gate: checking UI vs model vocabulary alignment..."
        "$PYTHON" scripts/verify_vocab_alignment.py
        ;;
    preprocess)
        echo "[*] Running data preprocessing..."
        "$PYTHON" src/preprocess.py
        ;;
    train)
        echo "[*] Training two-stage machine learning model..."
        "$PYTHON" src/train.py
        ;;
    test)
        echo "[*] Executing unit tests and scenario tests T1-T15..."
        "$PYTHON" -m unittest discover tests
        ;;
    serve)
        echo "[*] Starting Web UI on http://127.0.0.1:5050..."
        "$PYTHON" src/app.py
        ;;
    drift)
        echo "[*] Running offline market drift & concept shift watchdog..."
        "$PYTHON" scripts/monitor_market_drift.py
        ;;
    pipeline)
        echo "[*] Running end-to-end pipeline: Preprocess -> Train -> Verify -> Test -> Serve..."
        "$PYTHON" src/preprocess.py
        "$PYTHON" src/train.py
        "$PYTHON" scripts/verify_vocab_alignment.py
        "$PYTHON" scripts/test_runner.py
        "$PYTHON" src/app.py
        ;;
    aws-provision)
        echo "[*] Provisioning S3 & SSM cloud resources in AWS..."
        "$PYTHON" scripts/deploy_aws.py
        ;;
    docker-deploy)
        echo "[*] Deploying pricing engine via Docker Compose..."
        docker compose up -d --build
        echo "[SUCCESS] Container running on http://localhost:5050"
        ;;
    docker-test)
        echo "[*] Executing live E2E smoke tests..."
        "$PYTHON" -m unittest tests/test_e2e_smoke.py
        ;;
    docker-down)
        echo "[*] Stopping pricing engine container..."
        docker compose down
        ;;
    *)
        echo "Usage: ./run.sh {verify|preprocess|train|test|serve|drift|pipeline|aws-provision|docker-deploy|docker-test|docker-down}"
        echo "Defaulting to starting the web app..."
        "$PYTHON" src/app.py
        ;;
esac
