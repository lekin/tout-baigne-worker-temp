#!/bin/bash
set -e

# Remote Audio QA Worker startup for RunPod Serverless Flex.

export PYTHONUNBUFFERED=1
export AIRTABLE_API_KEY=dummy
export AIRTABLE_BASE_ID=dummy
export AIRTABLE_TABLE_NAME=Tracks
export QA_CACHE_DIR=/workspace/qa_cache
export TORCH_HOME=/workspace/torch_home

REPO_ROOT=/workspace/tout-baigne
source "$REPO_ROOT/worker/venv/bin/activate"

cd "$REPO_ROOT"
python worker/runpod_handler.py
