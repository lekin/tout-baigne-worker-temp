"""CLI review UI for labeling QA benchmark tracks and refreshing metrics."""
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config import settings
from src.qa.benchmark import load_benchmark_labels, run_benchmark
from src.qa.models import BenchmarkLabel


VALID_LABELS = {
    "unknown",
    "good",
    "global_offset",
    "local_sync_problem",
    "wrong_version",
}
VALID_SEVERITY = {"unknown", "obvious", "subtle"}


def _labels_path(path: Optional[str]) -> Path:
    if path:
        return Path(path)
    # Default to the most recent 20-track label file.
    return Path("input/qa_benchmark_labels.yaml")


def load_labels(path: Optional[str] = None) -> List[BenchmarkLabel]:
    """Load benchmark labels from a YAML file."""
    labels_file = _labels_path(path)
    return load_benchmark_labels(str(labels_file))


def save_labels(labels: List[BenchmarkLabel], path: Optional[str] = None) -> str:
    """Save benchmark labels back to a YAML file."""
    labels_file = _labels_path(path)
    data = {
        "tracks": [
            {
                "record_id": label.record_id,
                "manual_status": label.manual_status,
                "human_severity": label.human_severity,
                "problem_ranges": label.problem_ranges,
                "known_global_offset_ms": label.known_global_offset_ms,
                "manual_line_onsets": label.manual_line_onsets,
            }
            for label in labels
        ]
    }
    with open(labels_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return str(labels_file)


def set_label(
    record_id: str,
    manual_status: str,
    *,
    human_severity: Optional[str] = None,
    problem_ranges: Optional[List[Dict[str, Any]]] = None,
    labels_path: Optional[str] = None,
) -> List[BenchmarkLabel]:
    """Update the manual label for one record and save the file."""
    if manual_status not in VALID_LABELS:
        raise ValueError(
            f"Invalid status '{manual_status}'. Valid: {sorted(VALID_LABELS)}"
        )
    if human_severity and human_severity not in VALID_SEVERITY:
        raise ValueError(
            f"Invalid severity '{human_severity}'. Valid: {sorted(VALID_SEVERITY)}"
        )

    labels = load_labels(labels_path)
    updated = False
    for label in labels:
        if label.record_id == record_id:
            label.manual_status = manual_status
            if human_severity:
                label.human_severity = human_severity
            if problem_ranges is not None:
                label.problem_ranges = problem_ranges
            updated = True
            break
    else:
        # Record not in labels; append it.
        labels.append(
            BenchmarkLabel(
                record_id=record_id,
                manual_status=manual_status,
                human_severity=human_severity or "unknown",
                problem_ranges=problem_ranges or [],
            )
        )

    save_labels(labels, labels_path)
    return labels


def refresh_benchmark(
    labels_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Re-run the benchmark report with the current labels."""
    labels_file = _labels_path(labels_path)
    out_dir = output_dir or str(settings.get_qa_output_path())
    result = run_benchmark(str(labels_file), output_dir=out_dir)
    return str(result["json_path"])


def render_table(labels: List[BenchmarkLabel], reports_dir: Optional[str] = None) -> str:
    """Return a Markdown table of current labels."""
    lines = ["| record | track | status | severity |", "|---|---|---|---|"]
    for label in labels:
        name = ""
        artist = ""
        if reports_dir:
            report_path = Path(reports_dir) / f"{label.record_id}.json"
            if report_path.exists():
                import json

                with open(report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("track_name", "")
                artist = data.get("artist_name", "")
        track = f"{artist} - {name}".strip(" -")
        lines.append(
            f"| {label.record_id} | {track} | {label.manual_status} | {label.human_severity or 'unknown'} |"
        )
    return "\n".join(lines)
