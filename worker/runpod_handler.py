#!/usr/bin/env python3
"""RunPod Serverless worker handler for the Audio QA Worker."""
import json
import os
import sys
import time

# Repo root is wherever this file lives (network volume or container disk).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AIRTABLE_API_KEY", "dummy")
os.environ.setdefault("AIRTABLE_BASE_ID", "dummy")
os.environ.setdefault("AIRTABLE_TABLE_NAME", "Tracks")
os.environ.setdefault("QA_CACHE_DIR", os.environ.get("RUNPOD_VOLUME_PATH", "/workspace") + "/qa_cache")

import runpod

from src.qa.audio_qa_executor import AudioQARequest, run_audio_qa


def handler(job):
    """Handle a single RunPod Serverless job."""
    job_input = job.get("input", {})
    if not job_input:
        return {"success": False, "error": "missing_input"}

    try:
        request = AudioQARequest(**job_input)
    except Exception as e:
        return {"success": False, "error": f"invalid_request: {e}"}

    # GPU name is not known until after PyTorch loads, but run_audio_qa captures it.
    work_dir = os.environ.get("QA_WORK_DIR", "/workspace/runpod_qa_work")
    result = run_audio_qa(request, work_dir=work_dir, gpu_name=None)
    return {
        "success": result.success,
        "record_id": result.record_id,
        "status": result.status,
        "diagnosis": result.diagnosis,
        "metrics": result.metrics,
        "gates": result.gates,
        "separator": result.separator,
        "worker": result.worker,
        "error": result.error,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
