"""Persistence for QA results: local JSON and Airtable summary sync."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings
from src.qa.models import SyncReport, to_json


QA_STATUS_FIELD = "Karaoke QA Status"
QA_SYNC_VERIFIED_FIELD = "Karaoke QA Sync Verified"
QA_LINE_MEDIAN_ERROR_FIELD = "Karaoke QA Line Start Median Error (ms)"
QA_LINE_P90_ERROR_FIELD = "Karaoke QA Line Start P90 Error (ms)"
QA_DIAGNOSIS_FIELD = "Karaoke QA Diagnosis"
QA_LAST_RUN_FIELD = "Karaoke QA Last Run At"


def get_qa_output_dir() -> Path:
    path = Path(settings.qa_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_path(record_id: str) -> Path:
    return get_qa_output_dir() / f"{record_id}.json"


def save_sync_report(report: SyncReport) -> str:
    """Save a sync report to local JSON. Returns the file path."""
    path = _report_path(report.record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(report), f, indent=2, ensure_ascii=False)
    return str(path)


def load_sync_report(record_id: str) -> Optional[SyncReport]:
    """Load a sync report from local JSON."""
    from src.qa.models import from_json

    path = _report_path(record_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_json(SyncReport, data)


def build_airtable_summary(report: SyncReport) -> Dict[str, Any]:
    """Build the minimal Phase A Airtable summary fields."""
    return {
        QA_STATUS_FIELD: report.status.value,
        QA_SYNC_VERIFIED_FIELD: report.status == "SYNC_VERIFIED",
        QA_LINE_MEDIAN_ERROR_FIELD: report.metrics.line_start.median_error_ms,
        QA_LINE_P90_ERROR_FIELD: report.metrics.line_start.p90_error_ms,
        QA_DIAGNOSIS_FIELD: f"{report.diagnosis.type.value}: {report.diagnosis.description}",
        QA_LAST_RUN_FIELD: report.created_at,
    }


def update_airtable_summary(record_id: str, report: SyncReport) -> Optional[Dict[str, Any]]:
    """
    Update Airtable with a minimal QA summary.

    Disabled by default unless QA_UPDATE_AIRTABLE is set to 'true'. This keeps
    Phase A safe while still supporting schema updates when requested.
    """
    if os.getenv("QA_UPDATE_AIRTABLE", "false").lower() not in ("true", "1", "yes"):
        return None

    try:
        from src.airtable_client import AirtableClient

        client = AirtableClient()
        fields = build_airtable_summary(report)
        # Airtable cannot store NaN; clean it up.
        clean_fields: Dict[str, Any] = {}
        for k, v in fields.items():
            if isinstance(v, float) and (v != v):  # NaN
                clean_fields[k] = None
            else:
                clean_fields[k] = v
        return client.update_record(record_id, clean_fields)
    except Exception as e:
        print(f"⚠️  Could not update Airtable QA summary for {record_id}: {e}")
        return None
