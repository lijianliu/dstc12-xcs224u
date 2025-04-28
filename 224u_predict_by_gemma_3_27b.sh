#!/usr/bin/env bash
# Usage: ./gemma-3-27b.sh  <dataset_jsonl> <preference_json> <output_jsonl>

set -e
cd "$(dirname "$0")"                      # repo root

# 1) ensure PYTHONPATH contains ./src
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 2) run everything in ONE Python process
python - <<'PY' "$@"
import sys, runpy, patch_gemma_llm  # patch happens on import

if len(sys.argv) != 4:
    sys.exit("Usage: gemma-3-27b.sh data.jsonl prefs.json out.jsonl")

# Re-construct sys.argv for run_theme_detection.py
sys.argv = [
    "scripts/run_theme_detection.py",
    sys.argv[1],  # dataset_file
    sys.argv[2],  # preferences_file
    sys.argv[3],  # result_file
]
runpy.run_module("scripts.run_theme_detection", run_name="__main__")
PY

