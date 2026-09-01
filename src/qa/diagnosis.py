"""Diagnose synchronization problems from sync metrics."""
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import settings
from src.qa.models import Confidence, DiagnosisType, StructuredLyrics, SyncDiagnosis, SyncMetrics, SyncStatus


def _is_nan(value: Optional[float]) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _finite(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return value if not math.isnan(value) else 0.0


def diagnose(
    metrics: SyncMetrics,
    lyrics: Optional[StructuredLyrics] = None,
    estimated_global_offset_ms: Optional[float] = None,
    lyrics_to_source_offset_ms: float = 0.0,
) -> SyncDiagnosis:
    """
    Determine the most likely diagnosis from sync metrics.

    The ordering is intentionally conservative: wrong-version / alignment
    failures are considered first, then structural problems, then recoverable
    timing problems.
    """
    coverage = _finite(metrics.line_alignment_coverage)
    median_err = _finite(metrics.line_start.median_error_ms)
    max_err = _finite(metrics.line_start.max_error_ms)
    residual_std = _finite(metrics.drift_residual_std_ms)
    slope = metrics.drift_slope
    slope_deviation = abs(slope - 1.0)
    duration_mismatch = _finite(metrics.duration_mismatch_ms)
    unresolved = _finite(metrics.largest_unresolved_lyric_region_ms)

    # Alignment failure: no usable data.
    if metrics.line_start.sample_count == 0 or coverage == 0.0:
        return SyncDiagnosis(
            type=DiagnosisType.ALIGNMENT_FAILURE,
            confidence=Confidence.HIGH,
            description="Aligner produced no usable line predictions.",
        )

    # Very low coverage or huge duration mismatch -> likely wrong version/master.
    if coverage < 0.3 or duration_mismatch > 30_000:
        reason = (
            f"very low alignment coverage ({coverage:.1%})"
            if coverage < 0.3
            else f"large duration mismatch ({duration_mismatch:.0f} ms)"
        )
        return SyncDiagnosis(
            type=DiagnosisType.SUSPECTED_WRONG_VERSION,
            confidence=Confidence.HIGH,
            description=(
                f"{reason}. The Musixmatch candidate is probably for a different "
                f"recording."
            ),
        )

    # Progressive drift: systematic speed difference.
    if slope_deviation > settings.qa_max_drift_slope and residual_std < 800:
        return SyncDiagnosis(
            type=DiagnosisType.PROGRESSIVE_DRIFT,
            confidence=Confidence.HIGH,
            description=(
                f"Detected progressive drift: regression slope = {slope:.6f} "
                f"(deviation {slope_deviation:.6f}) with low residual std "
                f"({residual_std:.0f} ms). Likely different master/edit/version."
            ),
        )

    # Structural-analyzer global offset.  The estimated offset is the audio's
    # lead *in addition to* any user-supplied `Lyrics to singing offset` that
    # has already been folded into the source timeline.  If this residual is
    # large, the source lyrics need a further constant shift.
    if estimated_global_offset_ms is not None:
        residual_offset = estimated_global_offset_ms
        if abs(residual_offset) > 200.0:
            return SyncDiagnosis(
                type=DiagnosisType.GLOBAL_OFFSET,
                confidence=Confidence.HIGH,
                description=(
                    f"Structural match is consistently shifted by "
                    f"{residual_offset:+.0f} ms. Apply this correction to the "
                    f"source lyrics and re-run."
                ),
                estimated_global_offset_ms=residual_offset,
            )

    # Global offset: near-constant shift. Detect this before local mismatch so
    # a recoverable systematic offset is not mistaken for a broken section.
    if median_err > settings.qa_max_line_start_median_ms and slope_deviation <= settings.qa_max_drift_slope:
        if _looks_like_global_offset(metrics, lyrics):
            # Use median signed error as the offset estimate when lyrics are
            # available; otherwise fall back to the regression intercept.
            offset = _median_signed_error_ms(lyrics)
            if offset is None:
                offset = _finite(metrics.drift_intercept_ms)
            mad = _median_abs_deviation_ms(lyrics) or 0.0
            conf = Confidence.HIGH if mad < 1.5 * settings.qa_max_line_start_median_ms else Confidence.MEDIUM
            return SyncDiagnosis(
                type=DiagnosisType.GLOBAL_OFFSET,
                confidence=conf,
                description=(
                    f"Near-constant timing shift of approximately {offset:.0f} ms "
                    f"(median abs error {median_err:.0f} ms, MAD {mad:.0f} ms)."
                ),
                estimated_global_offset_ms=offset,
            )

    # Local mismatch: low residual after accounting for global offset.
    # Check for clusters of lines with large residual errors.
    if residual_std > 400 and max_err > 1500:
        return SyncDiagnosis(
            type=DiagnosisType.LOCAL_MISMATCH,
            confidence=Confidence.MEDIUM,
            description=(
                f"High local residual std ({residual_std:.0f} ms) and large "
                f"single-line error ({max_err:.0f} ms). Some sections align but "
                f"others do not."
            ),
        )

    # Unresolved lyric region.
    if unresolved > settings.qa_max_unresolved_lyric_region_ms:
        conf = Confidence.HIGH if unresolved > 4000.0 else Confidence.MEDIUM
        return SyncDiagnosis(
            type=DiagnosisType.LOCAL_MISMATCH,
            confidence=conf,
            description=(
                f"Unresolved lyric region of {unresolved:.0f} ms exceeds threshold "
                f"({settings.qa_max_unresolved_lyric_region_ms:.0f} ms)."
            ),
        )

    # Borderline wrong version / low coverage.
    if coverage < settings.qa_min_line_coverage:
        return SyncDiagnosis(
            type=DiagnosisType.SUSPECTED_WRONG_VERSION,
            confidence=Confidence.MEDIUM,
            description=(
                f"Line coverage ({coverage:.1%}) is below threshold "
                f"({settings.qa_min_line_coverage:.1%})."
            ),
        )

    # Looks good.
    return SyncDiagnosis(
        type=DiagnosisType.GOOD,
        confidence=Confidence.HIGH,
        description=(
            f"Line median error {median_err:.0f} ms, P90 "
            f"{metrics.line_start.p90_error_ms:.0f} ms, max {max_err:.0f} ms, "
            f"coverage {coverage:.1%}, no significant drift."
        ),
    )


def _line_signed_errors_ms(lyrics: Optional[StructuredLyrics]) -> List[float]:
    if lyrics is None:
        return []
    return [
        float(line.start_error_ms)
        for line in lyrics.lines
        if line.start_error_ms is not None
    ]


def _median_signed_error_ms(lyrics: Optional[StructuredLyrics]) -> Optional[float]:
    errors = _line_signed_errors_ms(lyrics)
    if not errors:
        return None
    return float(np.median(np.array(errors, dtype=float)))


def _median_abs_deviation_ms(lyrics: Optional[StructuredLyrics]) -> Optional[float]:
    errors = _line_signed_errors_ms(lyrics)
    if not errors:
        return None
    arr = np.array(errors, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return mad


def _looks_like_global_offset(
    metrics: SyncMetrics, lyrics: Optional[StructuredLyrics]
) -> bool:
    """Return True if the signed line errors cluster around a constant shift."""
    mad = _median_abs_deviation_ms(lyrics)
    if mad is not None:
        return mad < 1.5 * settings.qa_max_line_start_median_ms
    # Fallback to regression residual if per-line errors are unavailable.
    return _finite(metrics.drift_residual_std_ms) < 1.5 * settings.qa_max_line_start_median_ms


CORE_GATE_KEYS = [
    "line_start_median",
    "line_start_p90",
    "line_start_max",
    "line_coverage",
    "unresolved_lyric_region",
    "drift_slope",
]


def status_from_gates_and_diagnosis(
    gate_results: Dict[str, Any], diagnosis: SyncDiagnosis
) -> SyncStatus:
    """Determine SYNC_* status from gate results and diagnosis."""
    all_pass = all(
        gate_results.get(k, {}).get("pass", False) for k in CORE_GATE_KEYS
    )

    # Even if all numeric gates pass, a confident non-good diagnosis means the
    # track is not verified (e.g. wrong version with otherwise small errors).
    bad_diagnoses = {
        DiagnosisType.PROGRESSIVE_DRIFT,
        DiagnosisType.SUSPECTED_WRONG_VERSION,
        DiagnosisType.ALIGNMENT_FAILURE,
        DiagnosisType.LOCAL_MISMATCH,
    }
    if (
        diagnosis.confidence == Confidence.HIGH
        and diagnosis.type in bad_diagnoses
    ):
        return SyncStatus.SYNC_FAILED

    # Global offset is confidently identified but recoverable; not verified yet.
    if (
        diagnosis.confidence == Confidence.HIGH
        and diagnosis.type == DiagnosisType.GLOBAL_OFFSET
    ):
        return SyncStatus.SYNC_FAILED

    if all_pass:
        return SyncStatus.SYNC_VERIFIED

    return SyncStatus.SYNC_NEEDS_REVIEW
