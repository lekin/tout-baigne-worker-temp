"""Hard gates that determine whether sync QA passes."""
import math
from typing import Any, Dict

from src.config import settings
from src.qa.models import SyncMetrics


def evaluate_gates(metrics: SyncMetrics) -> Dict[str, Any]:
    """Evaluate every hard gate. Returns a dict with gate name -> pass/fail + value."""
    gates: Dict[str, Any] = {}

    median = _finite(metrics.line_start.median_error_ms)
    p90 = _finite(metrics.line_start.p90_error_ms)
    max_err = _finite(metrics.line_start.max_error_ms)
    line_coverage = _finite(metrics.line_alignment_coverage)
    word_coverage = _finite(metrics.word.coverage)
    unresolved = _finite(metrics.largest_unresolved_lyric_region_ms)
    slope_dev = abs(_finite(metrics.drift_slope) - 1.0)

    gates["line_start_median"] = {
        "value_ms": median,
        "threshold_ms": settings.qa_max_line_start_median_ms,
        "pass": median <= settings.qa_max_line_start_median_ms,
    }
    gates["line_start_p90"] = {
        "value_ms": p90,
        "threshold_ms": settings.qa_max_line_start_p90_ms,
        "pass": p90 <= settings.qa_max_line_start_p90_ms,
    }
    gates["line_start_max"] = {
        "value_ms": max_err,
        "threshold_ms": settings.qa_max_line_start_single_ms,
        "pass": max_err <= settings.qa_max_line_start_single_ms,
    }
    gates["line_coverage"] = {
        "value": line_coverage,
        "threshold": settings.qa_min_line_coverage,
        "pass": line_coverage >= settings.qa_min_line_coverage,
    }
    gates["word_coverage"] = {
        "value": word_coverage,
        "threshold": settings.qa_min_word_coverage,
        "pass": word_coverage >= settings.qa_min_word_coverage,
    }
    gates["unresolved_lyric_region"] = {
        "value_ms": unresolved,
        "threshold_ms": settings.qa_max_unresolved_lyric_region_ms,
        "pass": unresolved <= settings.qa_max_unresolved_lyric_region_ms,
    }
    gates["drift_slope"] = {
        "value": slope_dev,
        "threshold": settings.qa_max_drift_slope,
        "pass": slope_dev <= settings.qa_max_drift_slope,
    }

    return gates


def _finite(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    return float(value)
