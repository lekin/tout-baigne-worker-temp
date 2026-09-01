"""Standalone audio QA executor that runs Demucs + Structural QA without Airtable."""
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests


@dataclass
class AudioQARequest:
    """Request to run audio QA on a source audio and a set of lyrics."""

    record_id: str
    audio_url: str
    audio_sha256: Optional[str] = None
    lyrics: List[Dict[str, Any]] = field(default_factory=list)
    lyrics_source: str = "lrc"
    lyrics_source_track_id: Optional[str] = None
    lyrics_language: Optional[str] = None
    transform: Dict[str, float] = field(default_factory=dict)
    source_duration_ms: Optional[float] = None
    separator_model: str = "htdemucs"
    separator_overlap: float = 0.10
    separator_shifts: int = 0
    separator_split: bool = True
    run_structural_qa: bool = True
    run_stable_ts: bool = False
    stem_retention: str = "failures_only"  # never, failures_only, always
    return_stem: bool = False


@dataclass
class AudioQAResult:
    """Compact result from the audio QA worker."""

    success: bool
    record_id: str
    status: str = "SYNC_FAILED"
    diagnosis: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    gates: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    separator: Optional[Dict[str, Any]] = None
    worker: Optional[Dict[str, Any]] = None


def _download_url(url: str, dst: str, timeout: int = 300) -> None:
    import shutil
    if url.startswith("file://"):
        shutil.copy(url[7:], dst)
        return
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_settings() -> None:
    """Provide dummy settings so src.config can import without a real .env."""
    # The remote worker does not need Airtable/Musixmatch, but the existing
    # src.config module requires these keys.  We set them lazily.
    os.environ.setdefault("AIRTABLE_API_KEY", "dummy")
    os.environ.setdefault("AIRTABLE_BASE_ID", "dummy")
    os.environ.setdefault("AIRTABLE_TABLE_NAME", "Tracks")


def _import_qa_modules():
    _ensure_settings()
    from src.qa.diagnosis import diagnose, status_from_gates_and_diagnosis
    from src.qa.gates import evaluate_gates
    from src.qa.models import LyricLine, LyricWord, StructuredLyrics, TimelineTransform
    from src.qa.scoring import compute_sync_metrics
    from src.qa.separator import PyTorchDemucsSeparator, VocalSeparationConfig
    from src.qa.structural import StructuralAnalyzer

    return {
        "diagnose": diagnose,
        "status_from_gates_and_diagnosis": status_from_gates_and_diagnosis,
        "evaluate_gates": evaluate_gates,
        "LyricLine": LyricLine,
        "LyricWord": LyricWord,
        "StructuredLyrics": StructuredLyrics,
        "TimelineTransform": TimelineTransform,
        "compute_sync_metrics": compute_sync_metrics,
        "PyTorchDemucsSeparator": PyTorchDemucsSeparator,
        "VocalSeparationConfig": VocalSeparationConfig,
        "StructuralAnalyzer": StructuralAnalyzer,
    }


def _build_lyrics(lines_data: List[Dict[str, Any]], source: str, track_id: Optional[str], language: Optional[str]) -> Any:
    mods = _import_qa_modules()
    LyricLine = mods["LyricLine"]
    LyricWord = mods["LyricWord"]
    StructuredLyrics = mods["StructuredLyrics"]

    lines: List[Any] = []
    for idx, item in enumerate(lines_data):
        words: List[Any] = []
        for widx, w in enumerate(item.get("words", [])):
            words.append(
                LyricWord(
                    id=f"W{idx+1:04d}-{widx+1:04d}",
                    text=str(w.get("text", "")),
                    source_start_ms=w.get("source_start_ms"),
                    source_end_ms=w.get("source_end_ms"),
                )
            )
        lines.append(
            LyricLine(
                id=f"L{idx+1:04d}",
                text=str(item.get("text", "")).strip(),
                source_start_ms=float(item["source_start_ms"]),
                source_end_ms=item.get("source_end_ms"),
                words=words,
            )
        )
    return StructuredLyrics(
        source=source,
        source_track_id=str(track_id) if track_id else None,
        language=language,
        lines=lines,
    )


def _lyrics_to_dict(lyrics: Any) -> List[Dict[str, Any]]:
    """Dump only the source-side data for the cache key."""
    out = []
    for ln in lyrics.lines:
        words = [
            {"text": w.text, "source_start_ms": w.source_start_ms, "source_end_ms": w.source_end_ms}
            for w in ln.words
        ]
        out.append(
            {
                "text": ln.text,
                "source_start_ms": ln.source_start_ms,
                "source_end_ms": ln.source_end_ms,
                "words": words,
            }
        )
    return out


def _predicted_duration_ms(lyrics: Any) -> Optional[float]:
    ends = [ln.predicted_end_ms for ln in lyrics.lines if ln.predicted_end_ms is not None]
    return max(ends) if ends else None


def _source_duration_ms(audio_path: str, provided: Optional[float]) -> float:
    if provided is not None:
        return provided
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        return float(wav.shape[-1]) / sr * 1000.0
    except Exception:
        return 0.0


def _media_duration_ms(path: str) -> Optional[float]:
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return float(wav.shape[-1]) / sr * 1000.0
    except Exception:
        return None


def _to_json(obj: Any) -> Any:
    """Convert dataclasses/enums to JSON-serializable dicts."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_json(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if obj != obj:  # NaN
        return None
    return obj


def run_audio_qa(request: AudioQARequest, work_dir: Optional[str] = None, gpu_name: Optional[str] = None) -> AudioQAResult:
    """Run Demucs + Structural QA for a single candidate."""
    mods = _import_qa_modules()
    VocalSeparationConfig = mods["VocalSeparationConfig"]
    PyTorchDemucsSeparator = mods["PyTorchDemucsSeparator"]
    StructuralAnalyzer = mods["StructuralAnalyzer"]
    TimelineTransform = mods["TimelineTransform"]
    compute_sync_metrics = mods["compute_sync_metrics"]
    evaluate_gates = mods["evaluate_gates"]
    diagnose = mods["diagnose"]
    status_from_gates_and_diagnosis = mods["status_from_gates_and_diagnosis"]

    import torch

    # Prepare working directory.
    work_dir = work_dir or os.path.join(tempfile.gettempdir(), "runpod_qa_worker")
    os.makedirs(work_dir, exist_ok=True)

    total_start = time.time()
    timings: Dict[str, float] = {"download_ms": 0.0, "separation_ms": 0.0, "alignment_ms": 0.0, "scoring_ms": 0.0}

    try:
        # Download source audio.
        print(f"[worker] {request.record_id}: downloading source audio...")
        audio_filename = os.path.basename(request.audio_url.split("?")[0]) or "source.mp3"
        audio_path = os.path.join(work_dir, f"{request.record_id}_{audio_filename}")
        if not os.path.exists(audio_path):
            t0 = time.time()
            _download_url(request.audio_url, audio_path)
            timings["download_ms"] = (time.time() - t0) * 1000.0

        # Validate hash.
        if request.audio_sha256:
            actual = _sha256_file(audio_path)
            if actual.lower() != request.audio_sha256.lower():
                return AudioQAResult(success=False, record_id=request.record_id, error="sha256_mismatch")

        # Build lyrics and transform.
        lyrics = _build_lyrics(
            request.lyrics,
            request.lyrics_source,
            request.lyrics_source_track_id,
            request.lyrics_language,
        )
        transform = TimelineTransform(**request.transform)
        source_duration_ms = _source_duration_ms(audio_path, request.source_duration_ms)

        # Vocal separation.
        # Prefer CUDA on the worker, fall back to MPS/CPU for local testing.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[worker] {request.record_id}: separating vocals with {request.separator_model} on {device}...")

        sep_config = VocalSeparationConfig(
            backend="pytorch_demucs",
            model=request.separator_model,
            overlap=request.separator_overlap,
            shifts=request.separator_shifts,
            split=request.separator_split,
            device=device,
            package_version=None,
        )

        from src.qa.cache import hash_audio_file

        # Content-addressed stem cache on worker.
        audio_hash = hash_audio_file(audio_path)
        stem_cache_path = Path(sep_config.cached_path(audio_path))
        cache_hit = stem_cache_path.exists()
        if not cache_hit:
            t0 = time.time()
            sep = PyTorchDemucsSeparator(device=device)
            ok = sep.separate(audio_path, stem_cache_path, sep_config)
            if not ok:
                return AudioQAResult(success=False, record_id=request.record_id, error="vocal_separation_failed")
            timings["separation_ms"] = (time.time() - t0) * 1000.0
            if stem_cache_path.exists():
                cache_hit = True
        else:
            timings["separation_ms"] = 0.0

        if not cache_hit or not stem_cache_path.exists():
            return AudioQAResult(success=False, record_id=request.record_id, error="vocal_stem_missing")

        stem_duration_ms = _media_duration_ms(str(stem_cache_path))
        stem_delta_ms = (stem_duration_ms or 0.0) - source_duration_ms

        # Structural QA.
        print(f"[worker] {request.record_id}: running Structural QA...")
        t0 = time.time()
        aligner = StructuralAnalyzer()
        alignment = aligner.align(audio_path, lyrics, vocals_path=str(stem_cache_path))
        if alignment.error or not any(ln.predicted_start_ms is not None for ln in alignment.lyrics.lines):
            return AudioQAResult(
                success=False,
                record_id=request.record_id,
                error=alignment.error or "alignment_no_predictions",
            )
        timings["alignment_ms"] = (time.time() - t0) * 1000.0

        print(f"[worker] {request.record_id}: computing metrics, gates, and diagnosis...")
        t0 = time.time()
        predicted_duration_ms = _predicted_duration_ms(alignment.lyrics)
        metrics = compute_sync_metrics(
            alignment.lyrics, transform, source_duration_ms, predicted_duration_ms
        )
        gate_results = evaluate_gates(metrics)
        diagnosis = diagnose(
            metrics,
            alignment.lyrics,
            estimated_global_offset_ms=alignment.estimated_global_offset_ms,
            lyrics_to_source_offset_ms=transform.lyrics_to_source_offset_ms,
        )
        status = status_from_gates_and_diagnosis(gate_results, diagnosis)
        timings["scoring_ms"] = (time.time() - t0) * 1000.0

        # Optional: include lightweight return stem for debug/benchmark.
        stem_reference = None
        if request.return_stem:
            stem_reference = str(stem_cache_path)
        elif request.stem_retention == "always" or (request.stem_retention == "failures_only" and status.value != "SYNC_VERIFIED"):
            stem_reference = str(stem_cache_path)

        total_ms = (time.time() - total_start) * 1000.0
        print(f"[worker] {request.record_id}: status={status.value} runtime_ms={total_ms:.0f}")

        return AudioQAResult(
            success=True,
            record_id=request.record_id,
            status=status.value,
            diagnosis=_to_json(diagnosis),
            metrics=_to_json(metrics),
            gates=_to_json(gate_results),
            separator={
                "backend": "pytorch_demucs",
                "model": request.separator_model,
                "overlap": request.separator_overlap,
                "shifts": request.separator_shifts,
                "split": request.separator_split,
                "separation_time_ms": timings["separation_ms"],
                "cache_hit": cache_hit,
                "stem_reference": stem_reference,
                "stem_duration_ms": stem_duration_ms,
                "stem_delta_ms": stem_delta_ms,
                "demucs_version": getattr(__import__("demucs", fromlist=["__version__"]), "__version__", "unknown"),
            },
            worker={
                "gpu": gpu_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                "runtime_ms": total_ms,
                "timings_ms": timings,
                "audio_hash": audio_hash,
            },
        )
    except Exception as e:
        return AudioQAResult(success=False, record_id=request.record_id, error=str(e))
