"""Sync scoring and metrics for karaoke QA."""
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import linregress

from src.config import settings
from src.qa.models import (
    LineTimingMetrics,
    LyricLine,
    StructuredLyrics,
    SyncMetrics,
    TimelineTransform,
    WordTimingMetrics,
)
from src.qa.timeline import apply_transform_to_source_ms


def _p90(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values), 90))


def _safe_stats(values: List[float]) -> Tuple[float, float, float, float, float, int]:
    if not values:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0
    arr = np.array(values, dtype=float)
    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    maxv = float(np.max(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return median, p90, maxv, mean, std, len(arr)


def compute_sync_metrics(
    lyrics: StructuredLyrics,
    transform: TimelineTransform,
    source_duration_ms: Optional[float] = None,
    predicted_duration_ms: Optional[float] = None,
) -> SyncMetrics:
    """Compute line/word timing metrics from aligned lyrics."""
    lines = lyrics.lines

    # Source timings are on the Musixmatch candidate timeline. Convert them to
    # the source audio timeline for comparison with predicted timings.
    line_start_errors: List[float] = []
    line_end_errors: List[float] = []

    # Source timings were already converted to the source audio timeline when
    # the lyrics were built; do not re-apply the transform here.
    # Lyrics cannot be physically rendered before the start of the audio, so
    # clamp source start to 0 for scoring purposes.
    for line in lines:
        if line.source_start_ms is None or line.predicted_start_ms is None:
            continue
        src_start = max(0.0, line.source_start_ms)
        err = line.predicted_start_ms - src_start
        line.start_error_ms = err
        line_start_errors.append(abs(err))

        if line.source_end_ms is not None and line.predicted_end_ms is not None:
            err_end = line.predicted_end_ms - line.source_end_ms
            line.end_error_ms = err_end
            line_end_errors.append(abs(err_end))

    line_start_metrics = _line_metrics_from_errors(line_start_errors)
    line_end_metrics = _line_metrics_from_errors(line_end_errors)

    total_expected_lines = max(1, len([ln for ln in lyrics.lines if ln.text.strip()]))
    line_alignment_coverage = line_start_metrics.sample_count / total_expected_lines

    word_metrics = _compute_word_metrics(lyrics)
    unresolved_region_ms = _largest_unresolved_lyric_region_ms(lyrics)
    slope, intercept, residual_std = _compute_drift(lyrics, transform)
    duration_mismatch = _duration_mismatch(
        source_duration_ms, predicted_duration_ms
    )

    return SyncMetrics(
        line_start=line_start_metrics,
        line_end=line_end_metrics,
        word=word_metrics,
        total_expected_lines=total_expected_lines,
        line_alignment_coverage=line_alignment_coverage,
        largest_unresolved_lyric_region_ms=unresolved_region_ms,
        drift_slope=slope,
        drift_intercept_ms=intercept,
        drift_residual_std_ms=residual_std,
        duration_mismatch_ms=duration_mismatch,
        source_duration_ms=source_duration_ms,
        predicted_duration_ms=predicted_duration_ms,
    )


def _line_metrics_from_errors(errors: List[float]) -> LineTimingMetrics:
    if not errors:
        return LineTimingMetrics(
            median_error_ms=float("nan"),
            p90_error_ms=float("nan"),
            max_error_ms=float("nan"),
            mean_error_ms=float("nan"),
            std_error_ms=float("nan"),
            sample_count=0,
        )
    median, p90, maxv, mean, std, n = _safe_stats(errors)
    return LineTimingMetrics(
        median_error_ms=median,
        p90_error_ms=p90,
        max_error_ms=maxv,
        mean_error_ms=mean,
        std_error_ms=std,
        sample_count=n,
    )


def _compute_word_metrics(lyrics: StructuredLyrics) -> WordTimingMetrics:
    total = 0
    aligned = 0
    errors: List[float] = []

    for line in lyrics.lines:
        for word in line.words:
            # Only count words that have source timing. Plain LRC/SRT sources
            # may be tokenised into words but have no timestamped word data, in
            # which case word alignment is not applicable.
            if word.source_start_ms is None:
                continue
            total += 1
            if word.predicted_start_ms is not None:
                aligned += 1
                err = abs(word.predicted_start_ms - word.source_start_ms)
                errors.append(err)

    if total == 0:
        # No word-level timing in the source, so there is nothing to align.
        # Treat word coverage as vacuously complete.
        coverage = 1.0
    else:
        coverage = aligned / total
    median, p90, maxv, mean, std, n = _safe_stats(errors)
    return WordTimingMetrics(
        coverage=coverage,
        aligned_count=aligned,
        total_count=total,
        median_error_ms=median if n else None,
        p90_error_ms=p90 if n else None,
        max_error_ms=maxv if n else None,
    )


def _largest_unresolved_lyric_region_ms(lyrics: StructuredLyrics) -> float:
    """
    Largest expected lyric-bearing region with no confident alignment.

    A line is unresolved when the aligner did not produce a predicted start
    time. We measure the duration of the longest contiguous block of unresolved
    expected lyric lines. Instrumental gaps between resolved lyric lines are
    not counted as unresolved because they contain no expected lyrics.
    """
    sorted_lines = sorted(
        [ln for ln in lyrics.lines if ln.text.strip()],
        key=lambda ln: ln.source_start_ms if ln.source_start_ms is not None else float("inf"),
    )
    if not sorted_lines:
        return 0.0

    def _line_end_ms(idx: int) -> float:
        line = sorted_lines[idx]
        if line.source_end_ms is not None:
            return line.source_end_ms
        # Infer end from next line start only if the next line is also
        # unresolved; otherwise do not absorb the following resolved/instrumental
        # gap into the unresolved region.
        if idx + 1 < len(sorted_lines):
            next_line = sorted_lines[idx + 1]
            if next_line.predicted_start_ms is None:
                return next_line.source_start_ms or (line.source_start_ms + 3000.0)
        return line.source_start_ms + 3000.0

    max_region = 0.0
    unresolved_start: Optional[float] = None

    for idx, line in enumerate(sorted_lines):
        resolved = line.predicted_start_ms is not None
        start = line.source_start_ms
        end = _line_end_ms(idx)

        if not resolved:
            if unresolved_start is None:
                unresolved_start = start
            # Extend region to the end of this line.
            region_end = end
            max_region = max(max_region, region_end - unresolved_start)
        else:
            unresolved_start = None

    return max_region


def _compute_drift(
    lyrics: StructuredLyrics,
    transform: TimelineTransform,
) -> Tuple[float, float, float]:
    """
    Fit predicted_start = source_start * slope + intercept.

    Returns (slope, intercept_ms, residual_std_ms). Slope of 1.0 and intercept
    near 0 indicate perfect alignment. A slope materially different from 1.0
    suggests progressive drift (different master / speed difference).
    """
    xs: List[float] = []
    ys: List[float] = []

    for line in lyrics.lines:
        if line.source_start_ms is None or line.predicted_start_ms is None:
            continue
        xs.append(line.source_start_ms)
        ys.append(line.predicted_start_ms)

    if len(xs) < 2:
        return 1.0, 0.0, float("nan")

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)

    try:
        res = linregress(x, y)
        slope = float(res.slope)
        intercept = float(res.intercept)
        # Predicted residual std in ms.
        y_pred = slope * x + intercept
        residuals = y - y_pred
        residual_std = float(np.std(residuals))
        return slope, intercept, residual_std
    except Exception:
        return 1.0, 0.0, float("nan")


def _duration_mismatch(
    source_duration_ms: Optional[float], predicted_duration_ms: Optional[float]
) -> float:
    """Return the duration mismatch between lyrics and audio.

    A normal karaoke source can have a short instrumental outro after the last
    lyric, so an audio tail of up to `qa_max_allowed_audio_tail_ms` is allowed.
    A tail longer than that (or lyrics ending after the audio) suggests the
    candidate belongs to a different edit/version.
    """
    if source_duration_ms is None or predicted_duration_ms is None:
        return float("nan")
    if source_duration_ms <= 0:
        return float("nan")
    predicted_excess = max(0.0, predicted_duration_ms - source_duration_ms)
    allowed_tail = settings.qa_max_allowed_audio_tail_ms
    audio_tail_excess = max(
        0.0, source_duration_ms - predicted_duration_ms - allowed_tail
    )
    return max(predicted_excess, audio_tail_excess)
