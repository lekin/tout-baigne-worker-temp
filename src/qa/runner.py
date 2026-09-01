"""Phase A sync QA runner: orchestrate one track end-to-end."""
import demucs
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.airtable_client import AirtableClient
from src.config import settings
from src.karaoke_generator import KaraokeGenerator
from src.musixmatch_lyrics import LRCParser, RichsyncParser
from src.qa.audio_sync import KnownTextAligner, StableTsAligner
from src.qa.structural import StructuralAnalyzer
from src.qa.cache import (
    alignment_cache_key,
    cached_alignment_path,
    cached_audio_path,
    cached_vocals_path,
    find_existing_cached_alignment,
    find_existing_cached_vocals,
    get_audio_cache_path,
    get_qa_cache_dir,
    hash_audio_file,
    vocals_cache_key,
)
from src.qa.diagnosis import diagnose, status_from_gates_and_diagnosis
from src.qa.gates import evaluate_gates
from src.qa.models import (
    AlignmentResult,
    Confidence,
    DiagnosisType,
    LyricLine,
    LyricWord,
    StructuredLyrics,
    SyncDiagnosis,
    SyncReport,
    SyncStatus,
    to_json,
)
from src.qa.persistence import save_sync_report, update_airtable_summary
from src.qa.scoring import compute_sync_metrics
from src.qa.timeline import build_timeline_transform


class KaraokeQARunner:
    """Run Phase A sync QA for a single Airtable track record."""

    def __init__(self, aligner: Optional[KnownTextAligner] = None, verbose: bool = False):
        self.airtable = AirtableClient()
        self.generator = KaraokeGenerator()
        self.aligner = aligner or StructuralAnalyzer()
        self.verbose = verbose
        self.last_vocal_info: Dict[str, Any] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"    {message}")

    def run(self, record_id: str) -> SyncReport:
        """End-to-end sync QA for one track."""
        record = self.airtable.get_record(record_id)
        fields = record.get("fields", {})

        track_name = fields.get("Name", "")
        artist_name = _extract_artist(fields)
        musixmatch_track_id = fields.get("Musixmatch Track ID")

        print(f"🎵 QA Phase A: {record_id} — {track_name}")

        self._log("Building timeline transform...")
        transform = build_timeline_transform(record)

        self._log("Downloading source audio...")
        audio_path = self._download_source_audio(record)
        if not audio_path:
            lyrics = self._build_structured_lyrics(fields, transform)
            return self._error_report(
                record_id,
                track_name,
                artist_name,
                musixmatch_track_id,
                transform,
                lyrics,
                "Could not download source audio.",
            )

        source_duration_ms = self._media_duration_ms(audio_path)
        self._log(f"Audio duration: {source_duration_ms / 1000.0:.2f}s")

        self._log("Building structured lyrics...")
        lyrics = self._build_structured_lyrics(fields, transform, source_duration_ms)

        report = self.evaluate(
            record_id=record_id,
            track_name=track_name,
            artist_name=artist_name,
            musixmatch_track_id=musixmatch_track_id,
            transform=transform,
            audio_path=audio_path,
            lyrics=lyrics,
            source_duration_ms=source_duration_ms,
            save_report=True,
        )

        if isinstance(report, SyncReport):
            save_sync_report(report)
            update_airtable_summary(record_id, report)
        return report

    def evaluate(
        self,
        *,
        record_id: str,
        track_name: str,
        artist_name: str,
        musixmatch_track_id: Any,
        transform: Any,
        audio_path: str,
        lyrics: StructuredLyrics,
        source_duration_ms: float,
        vocals_path: Optional[str] = None,
        save_report: bool = False,
    ) -> SyncReport:
        """Run alignment and scoring for a set of structured lyrics.

        This is the reusable Phase A engine.  Phase B calls it repeatedly with
        alternate Musixmatch candidates while reusing the same audio/vocals.
        """
        self._log("Separating vocals (Demucs)...")
        if vocals_path is None:
            vocals_path = self._get_or_separate_vocals(audio_path)
        if not vocals_path:
            return self._error_report(
                record_id,
                track_name,
                artist_name,
                musixmatch_track_id,
                transform,
                lyrics,
                "Could not separate vocals.",
                source_duration_ms=source_duration_ms,
            )

        self._log(f"Running {self.aligner.name} alignment...")
        # Speech-oriented aligners need isolated vocals.  The structural analyzer
        # is more robust when it can combine the full mix with the vocal track,
        # so we pass both paths when available.
        if isinstance(self.aligner, StableTsAligner):
            alignment = self._get_or_run_alignment(vocals_path, lyrics, vocals_path=None)
        else:
            alignment = self._get_or_run_alignment(audio_path, lyrics, vocals_path=vocals_path)
        if alignment.error or not _has_predictions(alignment.lyrics):
            return self._error_report(
                record_id,
                track_name,
                artist_name,
                musixmatch_track_id,
                transform,
                lyrics,
                alignment.error or "Alignment produced no predictions.",
                alignment_result=alignment,
                source_duration_ms=source_duration_ms,
                predicted_duration_ms=_lyrics_predicted_duration_ms(alignment.lyrics),
            )

        predicted_duration_ms = _lyrics_predicted_duration_ms(alignment.lyrics)

        self._log("Scoring sync metrics...")
        metrics = compute_sync_metrics(
            alignment.lyrics, transform, source_duration_ms, predicted_duration_ms
        )
        self._log("Evaluating hard gates...")
        gate_results = evaluate_gates(metrics)
        failed_gates = [k for k, v in gate_results.items() if not v["pass"]]
        if failed_gates:
            self._log(f"Failed gates: {', '.join(failed_gates)}")
        else:
            self._log("All gates passed")
        self._log("Running diagnosis...")
        diagnosis = diagnose(
            metrics,
            alignment.lyrics,
            estimated_global_offset_ms=alignment.estimated_global_offset_ms,
            lyrics_to_source_offset_ms=transform.lyrics_to_source_offset_ms,
        )
        status = status_from_gates_and_diagnosis(gate_results, diagnosis)

        vocal_info = getattr(self, "last_vocal_info", {}) or {}
        stem_duration_ms = None
        stem_delta_ms = None
        if vocals_path:
            stem_duration_ms = self._media_duration_ms(vocals_path)
            if stem_duration_ms is not None and source_duration_ms:
                stem_delta_ms = stem_duration_ms - source_duration_ms

        report = SyncReport(
            record_id=record_id,
            track_name=str(track_name or ""),
            artist_name=artist_name,
            musixmatch_track_id=str(musixmatch_track_id) if musixmatch_track_id else None,
            transform=transform,
            metrics=metrics,
            diagnosis=diagnosis,
            status=status,
            lyrics=alignment.lyrics,
            alignment_result=alignment,
            gate_results=gate_results,
            source_duration_ms=source_duration_ms,
            vocal_backend=vocal_info.get("backend"),
            vocal_model=vocal_info.get("model"),
            vocal_cache_hit=vocal_info.get("cache_hit"),
            vocal_separation_time_s=vocal_info.get("separation_time_s"),
            vocal_stem_duration_ms=stem_duration_ms,
            vocal_stem_delta_ms=stem_delta_ms,
            vocal_path=vocals_path,
            audio_path=audio_path,
        )

        if save_report:
            save_sync_report(report)
            update_airtable_summary(record_id, report)
        return report

    def _build_structured_lyrics(
        self,
        fields: Dict[str, Any],
        transform: Any,
        source_duration_ms: Optional[float] = None,
    ) -> StructuredLyrics:
        """Parse LRC/SRT/Richsync from Airtable fields into a canonical StructuredLyrics."""
        richsync_json = fields.get("Richsync JSON (Musixmatch)")
        lrc_content = fields.get("LRC (Musixmatch)")
        srt_content = (
            fields.get("SRT (Musixmatch)")
            or fields.get("Lyrics SRT")
            or fields.get("Lyrics")
        )
        track_id = fields.get("Musixmatch Track ID")
        return self.build_structured_lyrics_from_source(
            richsync_json, lrc_content, srt_content, track_id, transform, source_duration_ms
        )

    def build_structured_lyrics_from_source(
        self,
        richsync_json: Optional[str],
        lrc_content: Optional[str],
        srt_content: Optional[str],
        track_id: Optional[Any],
        transform: Any,
        source_duration_ms: Optional[float] = None,
    ) -> StructuredLyrics:
        """Parse LRC/SRT/Richsync into a canonical StructuredLyrics."""
        from src.qa.timeline import apply_transform_to_source_ms

        if richsync_json:
            parsed = RichsyncParser.parse(richsync_json)
            source = "richsync"
        elif lrc_content:
            parsed = LRCParser.parse(lrc_content)
            source = "lrc"
        elif srt_content:
            parsed = self._parse_srt_lines(srt_content)
            source = "srt"
        else:
            parsed = []
            source = "none"

        lines: List[LyricLine] = []
        for idx, item in enumerate(parsed):
            if not item.text.strip():
                continue
            # Source timestamps are on the Musixmatch candidate timeline;
            # convert to source audio timeline immediately for QA comparison.
            src_start = apply_transform_to_source_ms(transform, item.start * 1000.0)
            src_end = (
                apply_transform_to_source_ms(transform, item.end * 1000.0)
                if item.end is not None
                else None
            )

            # For word-level source timestamps, only Richsync provides them.
            if source == "richsync" and richsync_json:
                words = _richsync_words_for_line(richsync_json, idx, item, transform)
            else:
                words = _tokenize_line(item.text)

            lines.append(
                LyricLine(
                    id=f"L{idx+1:04d}",
                    text=item.text.strip(),
                    source_start_ms=src_start,
                    source_end_ms=src_end,
                    words=words,
                )
            )

        # Drop lyric events that start after the audio has ended.  Musixmatch
        # candidates for a different (usually longer) edit often append a
        # trailing block that has no corresponding audio, which would otherwise
        # create a huge unresolved lyric region.
        if source_duration_ms is not None:
            lines = [ln for ln in lines if (ln.source_start_ms or 0) <= source_duration_ms + 500.0]

        return StructuredLyrics(
            source=source,
            source_track_id=str(track_id) if track_id else None,
            language=None,
            lines=lines,
        )

    def _parse_srt_lines(self, srt_content: str) -> list:
        """Minimal SRT parser to SyncedLine-compatible objects."""
        from src.qa.timeline import apply_transform_to_source_ms
        # Simple SRT parser reused from karaoke_generator would be cleaner, but
        # for Phase A we keep it self-contained.
        import re

        pattern = re.compile(
            r"\d+\s+"
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+"
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*\n"
            r"((?:.|\n)*?)(?=\n\n+|\Z)",
            re.DOTALL,
        )

        class _SyncedLine:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        result = []
        for m in pattern.finditer(srt_content):
            start = (
                int(m.group(1)) * 3600
                + int(m.group(2)) * 60
                + int(m.group(3))
                + int(m.group(4)) / 1000.0
            )
            end = (
                int(m.group(5)) * 3600
                + int(m.group(6)) * 60
                + int(m.group(7))
                + int(m.group(8)) / 1000.0
            )
            text = m.group(9).replace("\n", " ").strip()
            if text:
                result.append(_SyncedLine(start, end, text))
        return result

    def _download_source_audio(self, record: Dict[str, Any]) -> Optional[str]:
        """Download the source audio to a content-hashed cache file."""
        url = self.airtable.get_audio_file_url(record)
        if not url:
            url = self.airtable.get_video_url(record)
        if not url:
            return None

        # Download to a temp file first, then hash and move to cache.
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            if "drive.google.com" in url:
                from src.batch_musixmatch_fetch import download_google_drive_file

                ok = download_google_drive_file(url, tmp_path, timeout_seconds=180)
                if not ok:
                    _download_streaming_url(url, tmp_path)
            else:
                _download_streaming_url(url, tmp_path)

            cache_key = hash_audio_file(tmp_path)
            ext = Path(url).suffix or ".mp3"
            if ext not in (".mp3", ".wav", ".m4a", ".flac"):
                ext = ".mp3"
            dest = cached_audio_path(cache_key, ext)
            os.replace(tmp_path, dest)
            return dest
        except Exception as e:
            print(f"⚠️  Could not download audio: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return None

    def _get_or_separate_vocals(self, audio_path: str) -> Optional[str]:
        """Return cached vocals path or separate vocals and cache."""
        config = self._vocal_separation_config()
        cache_key = vocals_cache_key(audio_path, config=config)

        # Existing stems produced by the legacy key remain valid and are not
        # confused with new backend-specific keys.
        cached = find_existing_cached_vocals(cache_key)
        if not cached and config.backend == "demucs_cpu" and config.model == "htdemucs":
            legacy_key = vocals_cache_key(audio_path, separator="demucs")
            cached = find_existing_cached_vocals(legacy_key)
        if cached:
            print(f"  ✅ Using cached vocals: {cached}")
            self.last_vocal_info = {
                "backend": config.backend,
                "model": config.model,
                "cache_hit": True,
                "separation_time_s": 0.0,
                "path": str(cached),
            }
            return cached

        output_dir = get_qa_cache_dir() / "vocals"
        output_dir.mkdir(parents=True, exist_ok=True)
        target = cached_vocals_path(cache_key)

        from src.qa.separator import get_vocal_separator

        mlx_venv = Path(settings.qa_mlx_venv)
        if not mlx_venv.is_absolute():
            mlx_venv = Path(__file__).resolve().parents[2] / mlx_venv
        separator = get_vocal_separator(config, mlx_venv=mlx_venv)
        print(f"  🎵 Separating vocals with {config.backend} ({config.model})...")

        import time
        t0 = time.perf_counter()
        ok = separator.separate(audio_path, target, config)
        t1 = time.perf_counter()
        if not ok or not Path(target).exists():
            return None

        self.last_vocal_info = {
            "backend": config.backend,
            "model": config.model,
            "cache_hit": False,
            "separation_time_s": round(t1 - t0, 3),
            "path": str(target),
        }
        return str(target)

    def _vocal_separation_config(self) -> Any:
        """Build the VocalSeparationConfig from project settings."""
        from src.qa.separator import VocalSeparationConfig

        backend = settings.qa_vocal_separator
        model = getattr(settings, "qa_vocal_model", "htdemucs")
        overlap = float(getattr(settings, "qa_vocal_overlap", 0.25))
        split = bool(getattr(settings, "qa_vocal_split", True))
        shifts = int(getattr(settings, "qa_vocal_shifts", 0))
        package_version = None

        # Backwards compatibility with the legacy "demucs" alias.
        if backend in ("demucs", "demucs_cpu"):
            backend = "demucs_cpu"
            package_version = demucs.__version__
            device = "cpu"
        elif backend == "demucs_mps":
            backend = "demucs_mps"
            package_version = demucs.__version__
            device = "mps"
        elif backend == "auto":
            mlx_venv = Path(settings.qa_mlx_venv)
            if not mlx_venv.is_absolute():
                mlx_venv = Path(__file__).resolve().parents[2] / mlx_venv
            if (mlx_venv / "bin" / "python").exists():
                backend = "mlx"
                package_version = "demucs-mlx"
                device = None
            else:
                backend = "demucs_cpu"
                package_version = demucs.__version__
                device = "cpu"
        elif backend == "mlx":
            package_version = "demucs-mlx"
            device = None
        else:
            device = None

        return VocalSeparationConfig(
            backend=backend,
            model=model,
            package_version=package_version,
            shifts=shifts,
            overlap=overlap,
            split=split,
            device=device,
        )

    def _get_or_run_alignment(
        self,
        audio_path: str,
        lyrics: StructuredLyrics,
        vocals_path: Optional[str] = None,
    ) -> AlignmentResult:
        """Check alignment cache; otherwise run the aligner."""
        lyrics_dict = to_json(lyrics)
        cache_key = alignment_cache_key(
            audio_path,
            lyrics_dict,
            self.aligner.name,
            self.aligner.model_name,
            self.aligner.settings(),
            vocals_path=vocals_path,
        )
        cached_path = find_existing_cached_alignment(cache_key)
        if cached_path:
            self._log(f"Using cached alignment: {cached_path}")
            try:
                with open(cached_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lyrics = _structured_lyrics_from_dict(data["lyrics"])
                return AlignmentResult(
                    aligner_name=data["aligner_name"],
                    model_name=data["model_name"],
                    settings=data["settings"],
                    lyrics=lyrics,
                    raw_output_path=cached_path,
                    error=data.get("error"),
                )
            except Exception:
                pass

        self._log("No alignment cache; running aligner...")
        alignment = self.aligner.align(
            audio_path, lyrics, vocals_path=vocals_path
        )
        self._save_alignment_cache(cache_key, alignment)
        return alignment

    def _save_alignment_cache(self, cache_key: str, alignment: AlignmentResult) -> None:
        path = cached_alignment_path(cache_key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "aligner_name": alignment.aligner_name,
                    "model_name": alignment.model_name,
                    "settings": alignment.settings,
                    "lyrics": to_json(alignment.lyrics),
                    "error": alignment.error,
                    "raw_output_path": alignment.raw_output_path,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _media_duration_ms(self, media_path: str) -> Optional[float]:
        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    media_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"]) * 1000.0
        except Exception:
            return None

    def _error_report(
        self,
        record_id: str,
        track_name: str,
        artist_name: str,
        musixmatch_track_id: Optional[str],
        transform: Any,
        lyrics: StructuredLyrics,
        error: str,
        alignment_result: Optional[AlignmentResult] = None,
        source_duration_ms: Optional[float] = None,
        predicted_duration_ms: Optional[float] = None,
    ) -> SyncReport:
        alignment = alignment_result or AlignmentResult(
            aligner_name="",
            model_name="",
            settings={},
            lyrics=lyrics,
            error=error,
        )
        metrics = compute_sync_metrics(
            lyrics,
            transform,
            source_duration_ms=source_duration_ms,
            predicted_duration_ms=predicted_duration_ms,
        )
        gate_results = evaluate_gates(metrics)
        diagnosis = SyncDiagnosis(
            type=DiagnosisType.ALIGNMENT_FAILURE,
            confidence=Confidence.HIGH,
            description=error,
        )
        status = status_from_gates_and_diagnosis(gate_results, diagnosis)
        report = SyncReport(
            record_id=record_id,
            track_name=track_name,
            artist_name=artist_name,
            musixmatch_track_id=str(musixmatch_track_id) if musixmatch_track_id else None,
            transform=transform,
            metrics=metrics,
            diagnosis=diagnosis,
            status=status,
            lyrics=lyrics,
            alignment_result=alignment,
            gate_results=gate_results,
        )
        save_sync_report(report)
        update_airtable_summary(record_id, report)
        return report


def _extract_artist(fields: Dict[str, Any]) -> str:
    for key in [
        "Artist (string)",
        "Artist",
        "Name (from Artist)",
        "Artist name",
        "Artists",
    ]:
        v = fields.get(key)
        if v:
            if isinstance(v, list):
                return str(v[0]) if v else ""
            return str(v)
    return ""


def _tokenize_line(text: str) -> List[LyricWord]:
    """Simple whitespace tokenization for lyric words."""
    import re

    words = re.findall(r"\S+", text.strip())
    return [
        LyricWord(id=f"W{i+1:04d}", text=w)
        for i, w in enumerate(words)
    ]


def _richsync_words_for_line(
    richsync_json: str,
    line_index: int,
    item: Any,
    transform: Any,
) -> List[LyricWord]:
    """Build LyricWord objects with source timestamps from Richsync character data."""
    import json
    from src.qa.timeline import apply_transform_to_source_ms

    try:
        data = json.loads(richsync_json)
    except (json.JSONDecodeError, TypeError):
        return _tokenize_line("")

    if not isinstance(data, list) or line_index >= len(data):
        return _tokenize_line("")

    entry = data[line_index]
    line_start_s = entry.get("ts", 0.0)
    line_end_s = entry.get("te", line_start_s + 3.0)
    chars = entry.get("l", [])

    raw_words: List[Tuple[str, float, Optional[float]]] = []
    current_word = ""
    current_start: Optional[float] = None

    for char in chars:
        c = char.get("c", "")
        off = char.get("o", 0.0)
        if c.isspace():
            if current_word and current_start is not None:
                raw_words.append((current_word, current_start, None))
                current_word = ""
                current_start = None
        else:
            if not current_word:
                current_start = off
            current_word += c

    if current_word and current_start is not None:
        raw_words.append((current_word, current_start, None))

    # Convert offsets to absolute source times and resolve end times.
    words: List[LyricWord] = []
    for i, (text, start_off, _) in enumerate(raw_words):
        start_s = line_start_s + start_off
        if i + 1 < len(raw_words):
            end_s = line_start_s + raw_words[i + 1][1]
        else:
            end_s = line_end_s

        src_word_start = apply_transform_to_source_ms(transform, start_s * 1000.0)
        src_word_end = apply_transform_to_source_ms(transform, end_s * 1000.0)

        words.append(
            LyricWord(
                id=f"W{i+1:04d}",
                text=text,
                source_start_ms=src_word_start,
                source_end_ms=src_word_end,
            )
        )
    return words


def _download_streaming_url(url: str, dst_path: str, timeout_seconds: int = 180) -> None:
    with requests.get(url, stream=True, allow_redirects=True, timeout=timeout_seconds) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _has_predictions(lyrics: StructuredLyrics) -> bool:
    return any(line.predicted_start_ms is not None for line in lyrics.lines)


def _lyrics_predicted_duration_ms(lyrics: StructuredLyrics) -> Optional[float]:
    ends = [
        line.predicted_end_ms
        for line in lyrics.lines
        if line.predicted_end_ms is not None
    ]
    if not ends:
        return None
    return max(ends)


def _structured_lyrics_from_dict(data: Any) -> StructuredLyrics:
    """Rehydrate StructuredLyrics from a JSON dict."""
    lines = []
    for ln in data.get("lines", []):
        words = []
        for w in ln.get("words", []):
            words.append(
                LyricWord(
                    id=w["id"],
                    text=w["text"],
                    source_start_ms=w.get("source_start_ms"),
                    source_end_ms=w.get("source_end_ms"),
                    predicted_start_ms=w.get("predicted_start_ms"),
                    predicted_end_ms=w.get("predicted_end_ms"),
                )
            )
        lines.append(
            LyricLine(
                id=ln["id"],
                text=ln["text"],
                source_start_ms=ln["source_start_ms"],
                source_end_ms=ln.get("source_end_ms"),
                words=words,
                predicted_start_ms=ln.get("predicted_start_ms"),
                predicted_end_ms=ln.get("predicted_end_ms"),
                start_error_ms=ln.get("start_error_ms"),
                end_error_ms=ln.get("end_error_ms"),
            )
        )
    return StructuredLyrics(
        source=data.get("source", "unknown"),
        source_track_id=data.get("source_track_id"),
        language=data.get("language"),
        lines=lines,
    )
