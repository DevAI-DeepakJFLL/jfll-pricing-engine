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
elif [ -f "/Users/deepak/.gemini/antigravity-cli/brain/89badbe2-0b7a-449a-ae32-01d520b4a325/scratch/venv/bin/python" ]; then
    PYTHON="/Users/deepak/.gemini/antigravity-cli/brain/89badbe2-0b7a-449a-ae32-01d520b4a325/scratch/venv/bin/python"
else
    PYTHON="python3"
fi

case "$1" in
    verify)
        echo "[*] Running verification gate: checking UI vs model vocabulary alignment..."
        $PYTHON scripts/verify_vocab_alignment.py
        ;;
    preprocess)
        echo "[*] Running data preprocessing..."
        $PYTHON src/preprocess.py
        ;;
    train)
        echo "[*] Training two-stage machine learning model..."
        $PYTHON src/train.py
        ;;
    test)
        echo "[*] Executing unit tests and scenario tests T1-T15..."
        $PYTHON -m unittest discover tests
        ;;
    serve)
        echo "[*] Starting Web UI on http://127.0.0.1:5050..."
        $PYTHON src/app.py
        ;;
    pipeline)
        echo "[*] Running end-to-end pipeline: Preprocess -> Train -> Verify -> Test -> Serve..."
        $PYTHON src/preprocess.py
        $PYTHON src/train.py
        $PYTHON scripts/verify_vocab_alignment.py
        $PYTHON scripts/test_runner.py
        $PYTHON src/app.py
        ;;
    *)
        echo "Usage: ./run.sh {verify|preprocess|train|test|serve|pipeline}"
        echo "Defaulting to starting the web app..."
        $PYTHON src/app.py
        ;;
esac
