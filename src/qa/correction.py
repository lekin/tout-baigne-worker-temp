"""Phase B fallback: resolve sync failures by trying alternate Musixmatch candidates."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.karaoke_generator import KaraokeGenerator
from src.musixmatch_lyrics import MusixmatchClient, MusixmatchTrack
from src.qa.models import StructuredLyrics, SyncReport, TimelineTransform
from src.qa.timeline import build_timeline_transform


@dataclass
class CandidateAttempt:
    """One Phase A evaluation of an alternate Musixmatch candidate."""

    musixmatch_track_id: str
    track_name: str
    artist_name: str
    album_name: str
    track_length: Optional[int]
    source: str  # richsync / lrc / srt / subtitle
    report: SyncReport


@dataclass
class CorrectionResult:
    """Outcome of a Phase B fallback search."""

    record_id: str
    original_report: Optional[SyncReport] = None
    attempts: List[CandidateAttempt] = field(default_factory=list)
    best_attempt: Optional[CandidateAttempt] = None
    selected: bool = False
    error: Optional[str] = None


def _default_api_key() -> Optional[str]:
    return os.getenv("MUSIXMATCH_API_KEY") or settings.musixmatch_api_key


class MusixmatchFallbackResolver:
    """Search for alternate Musixmatch candidates and run Phase A on each."""

    def __init__(self, runner, api_key: Optional[str] = None):
        self.runner = runner
        self.client = MusixmatchClient(api_key or _default_api_key() or "")

    def resolve(
        self,
        record_id: str,
        fields: Dict[str, Any],
        audio_path: str,
        source_duration_ms: float,
    ) -> CorrectionResult:
        """Find and evaluate alternate Musixmatch candidates for a failed track."""
        track_name = self._extract_track_name(fields)
        artist_name = self._extract_artist(fields)
        original_track_id = fields.get("Musixmatch Track ID")

        result = CorrectionResult(record_id=record_id)

        try:
            print(f"🔍 Phase B: searching alternate Musixmatch candidates for {record_id}")
            candidates = self.client.search_track(track_name, artist_name)
        except Exception as e:
            result.error = f"Musixmatch search failed: {e}"
            return result

        if not candidates:
            result.error = "No candidates found."
            return result

        # Hydrate durations and filter to plausible versions (within ~30 s of source).
        candidates = [c for c in candidates if self._is_plausible(c, source_duration_ms)]
        if not candidates:
            result.error = "No candidates with a plausible duration."
            return result

        vocals_path: Optional[str] = None
        for candidate in candidates:
            # Skip the original candidate if it is the one we already tried.
            if str(candidate.track_id) == str(original_track_id):
                continue

            attempt = self._try_candidate(
                record_id,
                track_name,
                artist_name,
                audio_path,
                source_duration_ms,
                candidate,
                vocals_path=vocals_path,
            )
            if attempt is None:
                continue

            if vocals_path is None:
                # Reuse the same separated vocal stem for all alternate candidates.
                vocals_path = self._find_vocals_for_audio(audio_path)

            result.attempts.append(attempt)

        result.best_attempt = self._select_best(result.attempts)
        if result.best_attempt:
            best = result.best_attempt
            # Accept the best only if it is a clear improvement (verified, or at
            # least fewer failures than the original).
            if best.report.status.value == "SYNC_VERIFIED" or self._better_than(
                best.report, result.original_report
            ):
                result.selected = True

        self._save_result(result)
        return result

    def _extract_track_name(self, fields: Dict[str, Any]) -> str:
        for key in ["Title", "Name (from Tracks API Metadata)", "Name", "Name (string)"]:
            v = fields.get(key)
            if v:
                if isinstance(v, list):
                    return str(v[0]) if v else ""
                return str(v)
        return ""

    def _extract_artist(self, fields: Dict[str, Any]) -> str:
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

    def _is_plausible(
        self, candidate: MusixmatchTrack, source_duration_ms: float
    ) -> bool:
        if not candidate.track_length:
            return True
        return abs(candidate.track_length * 1000 - source_duration_ms) <= 35_000

    def _try_candidate(
        self,
        record_id: str,
        track_name: str,
        artist_name: str,
        audio_path: str,
        source_duration_ms: float,
        candidate: MusixmatchTrack,
        vocals_path: Optional[str] = None,
    ) -> Optional[CandidateAttempt]:
        """Download a candidate's lyrics and run Phase A on it."""
        source, content = self._fetch_lyrics(candidate)
        if not content:
            return None

        transform = TimelineTransform(
            lyrics_to_source_offset_ms=0.0,
            source_to_karaoke_audio_offset_ms=0.0,
            karaoke_audio_to_video_offset_ms=0.0,
        )

        try:
            lyrics = self.runner.build_structured_lyrics_from_source(
                richsync_json=content if source == "richsync" else None,
                lrc_content=content if source == "lrc" else None,
                srt_content=content if source == "srt" else None,
                track_id=candidate.track_id,
                transform=transform,
                source_duration_ms=source_duration_ms,
            )
        except Exception as e:
            print(f"  ⚠️ Could not parse {source} for candidate {candidate.track_id}: {e}")
            return None

        report = self.runner.evaluate(
            record_id=record_id,
            track_name=track_name,
            artist_name=artist_name,
            musixmatch_track_id=candidate.track_id,
            transform=transform,
            audio_path=audio_path,
            lyrics=lyrics,
            source_duration_ms=source_duration_ms,
            vocals_path=vocals_path,
            save_report=False,
        )

        print(
            f"  🎵 Candidate {candidate.track_id} ({source}): "
            f"{report.status.value} | {report.diagnosis.type.value} | "
            f"median={report.metrics.line_start.median_error_ms:.0f}ms "
            f"p90={report.metrics.line_start.p90_error_ms:.0f}ms "
            f"cov={report.metrics.line_alignment_coverage:.2%}"
        )

        return CandidateAttempt(
            musixmatch_track_id=str(candidate.track_id),
            track_name=candidate.track_name,
            artist_name=candidate.artist_name,
            album_name=candidate.album_name,
            track_length=candidate.track_length,
            source=source,
            report=report,
        )

    def _fetch_lyrics(self, candidate: MusixmatchTrack) -> tuple[str, Optional[str]]:
        """Return (source_type, content) for the richest available synced lyrics."""
        if candidate.has_richsync:
            try:
                content = self.client.get_richsync(candidate.track_id)
                if content:
                    return "richsync", content
            except Exception:
                pass

        if candidate.has_subtitles:
            try:
                content = self.client.get_subtitle(candidate.track_id)
                if content:
                    return "lrc", content
            except Exception:
                pass

        return "none", None

    def _find_vocals_for_audio(self, audio_path: str) -> Optional[str]:
        """Return the cached vocal stem for an audio file, if it exists."""
        from src.qa.cache import find_existing_cached_vocals, vocals_cache_key

        key = vocals_cache_key(audio_path, separator=settings.qa_vocal_separator)
        return find_existing_cached_vocals(key)

    def _select_best(
        self, attempts: List[CandidateAttempt]
    ) -> Optional[CandidateAttempt]:
        """Pick the best alternate candidate."""
        if not attempts:
            return None

        def score(a: CandidateAttempt) -> tuple:
            verified = 1 if a.report.status.value == "SYNC_VERIFIED" else 0
            coverage = a.report.metrics.line_alignment_coverage
            median = a.report.metrics.line_start.median_error_ms or 1e9
            p90 = a.report.metrics.line_start.p90_error_ms or 1e9
            return (
                verified,  # 1 = verified, 0 = not
                coverage,  # higher is better
                -median,  # negate so smaller error sorts first
                -p90,
            )

        # Sort descending: verified first, then highest coverage, then lowest errors.
        return sorted(attempts, key=score, reverse=True)[0]

    def _better_than(
        self, best_report: SyncReport, original_report: Optional[SyncReport]
    ) -> bool:
        if original_report is None:
            return True
        if best_report.status.value == "SYNC_VERIFIED":
            return original_report.status.value != "SYNC_VERIFIED"
        # If still failing, only accept if coverage is higher.
        return (
            best_report.metrics.line_alignment_coverage
            > original_report.metrics.line_alignment_coverage
        )

    def _save_result(self, result: CorrectionResult) -> None:
        path = (
            Path(settings.get_qa_output_path())
            / "corrections"
            / f"{result.record_id}_correction.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_correction_json(result), f, indent=2)


def to_correction_json(result: CorrectionResult) -> Dict[str, Any]:
    """Serialize a CorrectionResult for persistence."""
    from src.qa.models import to_json

    return {
        "record_id": result.record_id,
        "selected": result.selected,
        "error": result.error,
        "best_attempt_id": result.best_attempt.musixmatch_track_id
        if result.best_attempt
        else None,
        "attempts": [
            {
                "musixmatch_track_id": a.musixmatch_track_id,
                "track_name": a.track_name,
                "artist_name": a.artist_name,
                "album_name": a.album_name,
                "track_length": a.track_length,
                "source": a.source,
                "report": to_json(a.report),
            }
            for a in result.attempts
        ],
    }
