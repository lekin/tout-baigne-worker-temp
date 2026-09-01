"""Known-text alignment backends."""
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import stable_whisper

from src.config import settings
from src.qa.models import AlignmentResult, LyricLine, LyricWord, StructuredLyrics
from src.qa.normalization import TokenMapper


class KnownTextAligner(ABC):
    """Abstract base for aligners that map known lyrics to audio."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def align(
        self,
        audio_path: str,
        lyrics: StructuredLyrics,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> AlignmentResult:
        ...

    def settings(self) -> Dict[str, Any]:
        return {}


class StableTsAligner(KnownTextAligner):
    """Known-text alignment using stable-ts / Whisper."""

    _models: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self.model_name = model_name or settings.qa_aligner_model
        self.device = device or settings.qa_aligner_device
        self.compute_type = compute_type or settings.qa_aligner_compute_type

    @property
    def name(self) -> str:
        return "stable-ts"

    def settings(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "original_split": True,
        }

    def _load_model(self):
        cache_key = f"{self.model_name}:{self.device}:{self.compute_type}"
        if cache_key in StableTsAligner._models:
            return StableTsAligner._models[cache_key]
        print(f"🎤 Loading Stable-ts aligner model '{self.model_name}'...")
        model = stable_whisper.load_faster_whisper(
            self.model_name, device=self.device, compute_type=self.compute_type
        )
        StableTsAligner._models[cache_key] = model
        return model

    def align(
        self,
        audio_path: str,
        lyrics: StructuredLyrics,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> AlignmentResult:
        try:
            model = self._load_model()
            text = _lyrics_to_alignment_text(lyrics)
            lang = language or lyrics.language or "en"

            result = model.align(
                audio_path,
                text,
                language=lang,
                original_split=True,
                failure_threshold=0.5,
            )

            if result is None:
                return AlignmentResult(
                    aligner_name=self.name,
                    model_name=self.model_name,
                    settings=self.settings(),
                    lyrics=lyrics,
                    error="stable-ts align() returned None",
                )

            mapped_lyrics = _map_whisper_result_to_lyrics(result, lyrics)
            raw_path = _save_raw_alignment(result, lyrics)

            return AlignmentResult(
                aligner_name=self.name,
                model_name=self.model_name,
                settings=self.settings(),
                lyrics=mapped_lyrics,
                raw_output_path=raw_path,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return AlignmentResult(
                aligner_name=self.name,
                model_name=self.model_name,
                settings=self.settings(),
                lyrics=lyrics,
                error=str(e),
            )


def _lyrics_to_alignment_text(lyrics: StructuredLyrics) -> str:
    """Convert lyrics to a newline-separated text with empty lines removed."""
    lines = [line.text.strip() for line in lyrics.lines if line.text.strip()]
    return "\n".join(lines)


def _map_whisper_result_to_lyrics(
    result: stable_whisper.WhisperResult, lyrics: StructuredLyrics
) -> StructuredLyrics:
    """
    Map stable-ts output segments to the original lyric lines by index.

    When original_split=True, stable-ts should return one segment per supplied
    non-empty line. We discard segments with zero/negative duration (alignment
    failure markers) and assign the remaining segments to source lines in order.
    """
    non_empty_source_lines: List[LyricLine] = [
        line for line in lyrics.lines if line.text.strip()
    ]

    result_segments = list(result)
    # A segment is considered failed if its duration is zero or negative. Such
    # segments are produced when the alignment aborts or a line cannot be placed.
    predicted_lines = [
        seg
        for seg in result_segments
        if seg.text.strip() and (float(seg.end) - float(seg.start)) > 0
    ]

    failed_count = len(result_segments) - len(predicted_lines)
    if result_segments and failed_count / len(result_segments) > 0.4:
        # If more than 40% of segments failed, the alignment is unreliable.
        return StructuredLyrics(
            source=lyrics.source,
            source_track_id=lyrics.source_track_id,
            language=lyrics.language,
            lines=[
                LyricLine(
                    id=line.id,
                    text=line.text,
                    source_start_ms=line.source_start_ms,
                    source_end_ms=line.source_end_ms,
                    words=[
                        LyricWord(
                            id=w.id,
                            text=w.text,
                            source_start_ms=w.source_start_ms,
                            source_end_ms=w.source_end_ms,
                        )
                        for w in line.words
                    ],
                )
                for line in lyrics.lines
            ],
        )

    mapped_lines: List[LyricLine] = []

    for source_idx, source_line in enumerate(non_empty_source_lines):
        mapped_line = LyricLine(
            id=source_line.id,
            text=source_line.text,
            source_start_ms=source_line.source_start_ms,
            source_end_ms=source_line.source_end_ms,
            words=[
                LyricWord(
                    id=w.id,
                    text=w.text,
                    source_start_ms=w.source_start_ms,
                    source_end_ms=w.source_end_ms,
                )
                for w in source_line.words
            ],
        )

        if source_idx < len(predicted_lines):
            seg = predicted_lines[source_idx]
            mapped_line.predicted_start_ms = float(seg.start) * 1000.0
            mapped_line.predicted_end_ms = float(seg.end) * 1000.0

            if getattr(seg, "has_words", False) and seg.words:
                _map_segment_words_to_line(seg.words, mapped_line)

        mapped_lines.append(mapped_line)

    # Preserve any empty original lines (no prediction possible).
    for line in lyrics.lines:
        if not line.text.strip():
            mapped_lines.append(
                LyricLine(
                    id=line.id,
                    text=line.text,
                    source_start_ms=line.source_start_ms,
                    source_end_ms=line.source_end_ms,
                    words=[],
                )
            )

    # Sort by source start time to keep original order.
    mapped_lines.sort(key=lambda ln: ln.source_start_ms)

    return StructuredLyrics(
        source=lyrics.source,
        source_track_id=lyrics.source_track_id,
        language=lyrics.language,
        lines=mapped_lines,
    )


def _map_segment_words_to_line(
    segment_words: List[Any], mapped_line: LyricLine
) -> None:
    """Map stable-ts word tokens to the lyric words of a line."""
    source_words = mapped_line.words
    if not source_words:
        return

    predicted_words = [w for w in segment_words if w.word.strip()]
    if len(predicted_words) == 0:
        return

    mapper = TokenMapper()
    source_texts = [w.text for w in source_words]
    predicted_texts = [w.word.strip() for w in predicted_words]
    mapping = mapper.map_predicted_to_source(source_texts, predicted_texts)

    for pred_idx, src_idx in enumerate(mapping):
        if src_idx is None:
            continue
        pred = predicted_words[pred_idx]
        src_word = source_words[src_idx]
        src_word.predicted_start_ms = float(pred.start) * 1000.0
        src_word.predicted_end_ms = float(pred.end) * 1000.0


def _save_raw_alignment(
    result: stable_whisper.WhisperResult, lyrics: StructuredLyrics
) -> str:
    """Persist the raw stable-ts result dict for debugging."""
    from src.config import settings

    out_dir = Path(settings.qa_output_dir) / "raw_alignments"
    out_dir.mkdir(parents=True, exist_ok=True)
    track_id = lyrics.source_track_id or "unknown"
    safe_id = "".join(c for c in track_id if c.isalnum() or c in "_-.")[:80]
    path = out_dir / f"{safe_id}_{os.urandom(4).hex()}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception:
        return ""
    return str(path)


# Backwards compatibility for the protocol type if needed.
__all__ = ["KnownTextAligner", "StableTsAligner"]
