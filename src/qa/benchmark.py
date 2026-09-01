"""Benchmark harness for Phase A sync QA."""
import json
import math
import shutil
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config import settings
from src.qa.audio_sync import StableTsAligner
from src.qa.models import BenchmarkLabel, BenchmarkResult, SyncReport, to_json
from src.qa.runner import KaraokeQARunner
from src.qa.timeline import build_timeline_transform


def load_benchmark_labels(path: str) -> List[BenchmarkLabel]:
    """Load a YAML or JSON file of benchmark labels."""
    p = Path(path)
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        if p.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)

    labels = []
    for item in data.get("tracks", []):
        labels.append(
            BenchmarkLabel(
                record_id=item["record_id"],
                manual_status=item.get("manual_status", "good"),
                problem_ranges=item.get("problem_ranges", []),
                known_global_offset_ms=item.get("known_global_offset_ms"),
                human_severity=item.get("human_severity"),
                manual_line_onsets=item.get("manual_line_onsets", []),
            )
        )
    return labels


def _is_bad_label(label: BenchmarkLabel) -> Optional[bool]:
    """Return True for a known-bad label, False for known-good, None for unknown."""
    if label.manual_status == "good":
        return False
    if label.manual_status in ("bad", "global_offset", "local_sync_problem", "wrong_version"):
        return True
    return None


def _bad_subcategory(label: BenchmarkLabel) -> str:
    """Map a bad manual status to a canonical subcategory."""
    if label.manual_status == "good":
        return "good"
    if label.manual_status in ("global_offset",):
        return "global_offset"
    if label.manual_status in ("wrong_version",):
        return "wrong_version"
    if label.manual_status in ("bad", "local_sync_problem"):
        return "local_sync_problem"
    return "unknown"


def _compare_manual_vs_predicted_onsets(
    report: SyncReport, label: BenchmarkLabel
) -> List[Dict[str, Any]]:
    """Compare manually annotated line onsets to predicted onsets."""
    comparisons = []
    if not label.manual_line_onsets:
        return comparisons

    lines_by_index = {i: ln for i, ln in enumerate(report.lyrics.lines)}
    for onset in label.manual_line_onsets:
        line_idx = onset.get("line")
        expected_ms = onset.get("time_ms")
        if line_idx is None or expected_ms is None:
            continue
        line = lines_by_index.get(int(line_idx))
        if not line:
            continue
        predicted_ms = line.predicted_start_ms
        error_ms = (
            (predicted_ms - expected_ms) if predicted_ms is not None else None
        )
        comparisons.append(
            {
                "line": line_idx,
                "expected_ms": expected_ms,
                "predicted_ms": predicted_ms,
                "error_ms": error_ms,
                "lyric": line.text,
            }
        )
    return comparisons


def _stable_ts_cross_check(
    primary_report: SyncReport,
    primary_runner: KaraokeQARunner,
    record_id: str,
) -> Optional[SyncReport]:
    """Run a secondary Stable-ts QA on the same audio/vocals/lyrics.

    Uses the cached vocal stem from the primary run so no re-separation is done.
    """
    if not primary_report.vocal_path:
        return None

    try:
        record = primary_runner.airtable.get_record(record_id)
        fields = record.get("fields", {})
        artist_name = primary_report.artist_name
        track_name = primary_report.track_name

        st_runner = KaraokeQARunner(
            aligner=StableTsAligner(
                model_name=settings.qa_aligner_model,
                device=settings.qa_aligner_device,
                compute_type=settings.qa_aligner_compute_type,
            ),
            verbose=False,
        )

        # Re-build lyrics from the original Airtable source so predictions are not
        # polluted by the primary alignment. The transform is the same.
        lyrics = primary_runner._build_structured_lyrics(
            fields, primary_report.transform, primary_report.source_duration_ms
        )

        return st_runner.evaluate(
            record_id=record_id,
            track_name=track_name,
            artist_name=artist_name,
            musixmatch_track_id=primary_report.musixmatch_track_id,
            transform=primary_report.transform,
            audio_path=primary_report.audio_path or "",
            lyrics=lyrics,
            source_duration_ms=primary_report.source_duration_ms or 0.0,
            vocals_path=primary_report.vocal_path,
            save_report=False,
        )
    except Exception as e:
        print(f"  ⚠️ Stable-ts cross-check failed for {record_id}: {e}")
        return None


def run_benchmark(
    labels: List[BenchmarkLabel],
    output_dir: Optional[str] = None,
    runner: Optional[KaraokeQARunner] = None,
    verbose: bool = False,
    include_stable_ts: bool = True,
) -> BenchmarkResult:
    """Run sync QA for each label and compare to manual ground truth."""
    runner = runner or KaraokeQARunner(verbose=verbose)
    out_dir = Path(output_dir or settings.qa_output_dir) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_track: List[Dict[str, Any]] = []
    tp = tn = fp = fn = unknown = 0

    # Operational/performance counters.
    total_wall_start = time.perf_counter()
    total_separation_time_s = 0.0
    cache_hits = 0
    cache_misses = 0
    separation_times: List[float] = []
    track_wall_times: List[float] = []

    for idx, label in enumerate(labels, start=1):
        print(f"\n[{idx}/{len(labels)}] Benchmark track: {label.record_id}")
        track_start = time.perf_counter()
        try:
            report = runner.run(label.record_id)
        except Exception as e:
            import traceback

            traceback.print_exc()
            per_track.append(
                {
                    "record_id": label.record_id,
                    "manual_status": label.manual_status,
                    "error": str(e),
                    "predicted_status": None,
                }
            )
            actual = _is_bad_label(label)
            if actual is None:
                unknown += 1
            elif actual:
                fn += 1
            else:
                tn += 1
            continue

        track_wall = time.perf_counter() - track_start
        track_wall_times.append(track_wall)

        predicted_bad = report.status != "SYNC_VERIFIED"
        actual_bad = _is_bad_label(label)

        if actual_bad is None:
            unknown += 1
        elif actual_bad and predicted_bad:
            tp += 1
        elif not actual_bad and not predicted_bad:
            tn += 1
        elif not actual_bad and predicted_bad:
            fn += 1
        else:
            fp += 1

        # Stable-ts secondary cross-check.
        st_report = None
        if include_stable_ts:
            print(f"  🎤 Stable-ts cross-check for {label.record_id}...")
            st_report = _stable_ts_cross_check(report, runner, label.record_id)

        if st_report:
            report.stable_ts_status = st_report.status.value
            report.stable_ts_diagnosis = st_report.diagnosis.type.value
            report.stable_ts_median_error_ms = st_report.metrics.line_start.median_error_ms
            report.stable_ts_p90_error_ms = st_report.metrics.line_start.p90_error_ms

        if verbose:
            print(
                f"  status: {report.status.value} | "
                f"diagnosis: {report.diagnosis.type.value} ({report.diagnosis.confidence.value})\n"
                f"  median={report.metrics.line_start.median_error_ms:.0f}ms "
                f"p90={report.metrics.line_start.p90_error_ms:.0f}ms "
                f"max={report.metrics.line_start.max_error_ms:.0f}ms "
                f"cov={report.metrics.line_alignment_coverage:.2%} "
                f"unresolved={report.metrics.largest_unresolved_lyric_region_ms:.0f}ms"
            )

        # Operational metrics.
        sep_time = report.vocal_separation_time_s or 0.0
        total_separation_time_s += sep_time
        if report.vocal_cache_hit is True:
            cache_hits += 1
        elif report.vocal_cache_hit is False:
            cache_misses += 1
        if sep_time > 0:
            separation_times.append(sep_time)

        onset_comparisons = _compare_manual_vs_predicted_onsets(report, label)

        per_track.append(
            {
                "record_id": label.record_id,
                "track_name": report.track_name,
                "artist_name": report.artist_name,
                "manual_status": label.manual_status,
                "manual_bad_subcategory": _bad_subcategory(label),
                "predicted_status": report.status.value,
                "diagnosis": {
                    "type": report.diagnosis.type.value
                    if hasattr(report.diagnosis.type, "value")
                    else str(report.diagnosis.type),
                    "confidence": report.diagnosis.confidence.value,
                    "description": report.diagnosis.description,
                    "estimated_global_offset_ms": report.diagnosis.estimated_global_offset_ms,
                },
                "line_start_median_error_ms": report.metrics.line_start.median_error_ms,
                "line_start_p90_error_ms": report.metrics.line_start.p90_error_ms,
                "line_start_max_error_ms": report.metrics.line_start.max_error_ms,
                "line_alignment_coverage": report.metrics.line_alignment_coverage,
                "word_coverage": report.metrics.word.coverage if report.metrics.word else None,
                "largest_unresolved_lyric_region_ms": report.metrics.largest_unresolved_lyric_region_ms,
                "drift_slope": report.metrics.drift_slope,
                "drift_intercept_ms": report.metrics.drift_intercept_ms,
                "duration_mismatch_ms": report.metrics.duration_mismatch_ms,
                "source_duration_ms": report.source_duration_ms,
                "vocal_backend": report.vocal_backend,
                "vocal_model": report.vocal_model,
                "vocal_cache_hit": report.vocal_cache_hit,
                "vocal_separation_time_s": report.vocal_separation_time_s,
                "vocal_stem_duration_ms": report.vocal_stem_duration_ms,
                "vocal_stem_delta_ms": report.vocal_stem_delta_ms,
                "stable_ts_status": report.stable_ts_status,
                "stable_ts_diagnosis": report.stable_ts_diagnosis,
                "stable_ts_median_error_ms": report.stable_ts_median_error_ms,
                "stable_ts_p90_error_ms": report.stable_ts_p90_error_ms,
                "track_wall_time_s": round(track_wall, 3),
                "gate_results": report.gate_results,
                "manual_onset_comparisons": onset_comparisons,
                "report_path": str(Path(settings.qa_output_dir) / f"{label.record_id}.json"),
            }
        )

        # Re-save the primary report so the stable-ts cross-check is persisted.
        from src.qa.persistence import save_sync_report

        try:
            save_sync_report(report)
        except Exception:
            pass

    total_wall_time = time.perf_counter() - total_wall_start

    total = len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    result = BenchmarkResult(
        total=total,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        per_track=per_track,
        details={
            "generated_at": datetime.now().isoformat(),
            "unknown_count": unknown,
            "known_count": total - unknown,
            "configuration": {
                "qa_vocal_separator": settings.qa_vocal_separator,
                "qa_vocal_model": settings.qa_vocal_model,
                "qa_vocal_overlap": settings.qa_vocal_overlap,
                "qa_vocal_shifts": settings.qa_vocal_shifts,
                "qa_vocal_split": settings.qa_vocal_split,
                "qa_aligner_model": settings.qa_aligner_model,
                "qa_aligner_device": settings.qa_aligner_device,
                "qa_aligner_compute_type": settings.qa_aligner_compute_type,
            },
            "performance": {
                "total_benchmark_wall_time_s": round(total_wall_time, 3),
                "total_vocal_separation_time_s": round(total_separation_time_s, 3),
                "average_track_wall_time_s": round(_avg(track_wall_times), 3),
                "average_separation_time_s": round(_avg(separation_times), 3),
                "median_separation_time_s": round(_median(separation_times), 3),
                "min_separation_time_s": round(min(separation_times), 3) if separation_times else 0.0,
                "max_separation_time_s": round(max(separation_times), 3) if separation_times else 0.0,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            },
        },
    )

    _write_benchmark_report(result, out_dir)
    return result


def _write_benchmark_report(result: BenchmarkResult, out_dir: Path) -> None:
    """Write both JSON and Markdown benchmark reports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{timestamp}.json"
    md_path = out_dir / f"benchmark_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_json(result), f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Karaoke QA Phase A Benchmark Report",
        "",
        f"Generated at: {result.details.get('generated_at', '')}",
        "",
        "## Configuration",
        "",
    ]
    cfg = result.details.get("configuration", {})
    for k, v in cfg.items():
        md_lines.append(f"- `{k}`: `{v}`")

    md_lines.extend([
        "",
        "## Aggregate statistics",
        "",
        f"- Total tracks: {result.total}",
        f"- True positives (bad correctly rejected): {result.true_positive}",
        f"- True negatives (good correctly verified): {result.true_negative}",
        f"- False positives (bad incorrectly verified): {result.false_positive}",
        f"- False negatives (good incorrectly rejected): {result.false_negative}",
        f"- Precision: {result.precision:.3f}",
        f"- Recall: {result.recall:.3f}",
        f"- F1: {result.f1:.3f}",
        f"- False-positive rate: {result.fpr:.3f}",
        f"- False-negative rate: {result.fnr:.3f}",
        "",
        "## Performance",
        "",
    ])
    perf = result.details.get("performance", {})
    for k, v in perf.items():
        md_lines.append(f"- {k}: `{v}`")

    md_lines.extend([
        "",
        "## Per-track results",
        "",
        "| record | track | manual | predicted | median (ms) | P90 (ms) | max (ms) | line cov | word cov | diagnosis | stable-ts | vocal | time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for t in result.per_track:
        err = t.get("error")
        track_name = t.get("track_name", "")
        if err:
            md_lines.append(
                f"| {t['record_id']} | {track_name} | {t.get('manual_status')} | ERROR | | | | | | {err[:80]} | | | |"
            )
            continue
        st = t.get("stable_ts_status") or ""
        vocal = f"{t.get('vocal_backend', '')}/{t.get('vocal_model', '')}"
        if t.get("vocal_cache_hit"):
            vocal += " (cached)"
        md_lines.append(
            f"| {t['record_id']} | {track_name} | {t.get('manual_status')} | {t.get('predicted_status')} | "
            f"{_fmt(t.get('line_start_median_error_ms'))} | "
            f"{_fmt(t.get('line_start_p90_error_ms'))} | "
            f"{_fmt(t.get('line_start_max_error_ms'))} | "
            f"{_fmt(t.get('line_alignment_coverage'))} | "
            f"{_fmt(t.get('word_coverage'))} | "
            f"{t.get('diagnosis', {}).get('type', '')} | "
            f"{st} | "
            f"{vocal} | "
            f"{_fmt(t.get('vocal_separation_time_s'))} |"
        )

    md_lines.append("")
    md_lines.append("## Detailed per-track data")
    md_lines.append("")
    md_lines.append("See JSON report for full details and per-line comparisons.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Also write the canonical report name requested by the benchmark plan.
    canonical_md_path = out_dir / "phase_a_20_track_report.md"
    shutil.copy2(md_path, canonical_md_path)
    canonical_json_path = out_dir / "phase_a_20_track_report.json"
    shutil.copy2(json_path, canonical_json_path)

    print(f"\n📊 Benchmark report saved to:\n  {json_path}\n  {md_path}")
    print(f"📄 Canonical report copied to:\n  {canonical_md_path}\n  {canonical_json_path}")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.0f}" if value == int(value) else f"{value:.2f}"
    return str(value)
