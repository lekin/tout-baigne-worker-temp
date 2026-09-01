"""Structural timing analyzer for music sync QA.

Speech-oriented forced aligners (Whisper, WhisperX, stable-ts) struggle with
music because they are trained on speech and are biased toward phoneme-level
onsets.  For karaoke validation we do not need exact phoneme timestamps: the
source lyrics already provide the intended word timing.  We only need to know
whether the *sequence and placement* of lyric events matches the audio.

This module compares the source lyric line timeline to a structural audio
feature (onset/vocal energy envelope).  It is intentionally robust to:

* a constant "lyrics appear slightly before the vocal" offset (common in karaoke);
* small local tempo drift;
* wrong versions (the lyric sequence will not match the audio event sequence).

It keeps the same ``align(audio_path, lyrics)`` interface as the stable-ts
aligner so it can be dropped into the QA pipeline.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import torch
import torchaudio

from src.config import settings
from src.qa.audio_sync import KnownTextAligner
from src.qa.models import AlignmentResult, LyricLine, LyricWord, StructuredLyrics


def _load_audio_mono(audio_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Load audio with torchaudio and convert to mono, target sample rate."""
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr
    return wav.squeeze(0).numpy(), sr


def _energy_onset_curve(
    audio_path: str,
    target_sr: int = 22050,
    hop_ms: float = 50.0,
    frame_ms: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return an onset-strength curve and time axis.

    We use a simple RMS energy curve with first-order difference.  A positive
    difference that exceeds a percentile-based threshold is treated as an onset
    candidate.  This is intentionally cheap and does not require librosa.
    """
    y, sr = _load_audio_mono(audio_path, target_sr)
    hop = int(sr * hop_ms / 1000.0)
    frame = int(sr * frame_ms / 1000.0)

    rms = []
    for i in range(0, len(y) - frame, hop):
        rms.append(np.sqrt(np.mean(y[i : i + frame] ** 2)))
    rms = np.array(rms, dtype=np.float64)

    # Flux / onset strength: positive differences in log-energy.
    log_rms = np.log1p(rms)
    diff = np.diff(log_rms, prepend=log_rms[0])
    onset_strength = np.maximum(diff, 0.0)

    # Time axis in milliseconds.
    times = np.arange(len(onset_strength)) * hop_ms
    return onset_strength, times


def _extract_onset_times(
    onset_strength: np.ndarray,
    times_ms: np.ndarray,
    percentile: float = 50.0,
    min_gap_ms: float = 300.0,
) -> np.ndarray:
    """Return onset times (ms) where the onset strength exceeds a threshold.

    The threshold is a percentile of the positive-only onset strength, and a
    small minimum gap suppresses double peaks on the same musical event.
    """
    threshold = np.percentile(onset_strength[onset_strength > 0], percentile) if np.any(onset_strength > 0) else 0.0
    candidates = times_ms[onset_strength > threshold]

    filtered: List[float] = []
    for t in candidates:
        if not filtered or (t - filtered[-1]) >= min_gap_ms:
            filtered.append(float(t))
    return np.array(filtered, dtype=np.float64)


def _word_tokenize(text: str) -> List[str]:
    """Simple word tokenization for structural mapping."""
    return re.findall(r"[\w']+|[^\w\s]", text)


def _richsync_word_times(
    richsync_entry: Dict[str, Any], line_start_s: float
) -> List[Tuple[str, float, float]]:
    """Derive word start/end times (seconds, on source timeline) from Richsync.

    Richsync gives per-character offsets inside a line.  We group consecutive
    non-space characters into words and use the next word's start or the line
    end for the word's end time.
    """
    chars = richsync_entry.get("l", [])
    line_end_s = richsync_entry.get("te", line_start_s + 3.0)

    words: List[Tuple[str, float, float]] = []
    current_word = ""
    current_start: Optional[float] = None

    for char in chars:
        c = char.get("c", "")
        off = char.get("o", 0.0)
        if c.isspace():
            if current_word and current_start is not None:
                words.append((current_word, line_start_s + current_start, None))
                current_word = ""
                current_start = None
        else:
            if not current_word:
                current_start = off
            current_word += c

    if current_word and current_start is not None:
        words.append((current_word, line_start_s + current_start, None))

    # Resolve end times from next word start or line end.
    result = []
    for i, (word, start, _) in enumerate(words):
        if i + 1 < len(words):
            end = words[i + 1][1]
        else:
            end = line_end_s
        result.append((word, start, end))
    return result


class StructuralAnalyzer(KnownTextAligner):
    """Compare source lyric timing to the audio's structural onset sequence."""

    def __init__(
        self,
        target_sr: int = 22050,
        hop_ms: float = 50.0,
        frame_ms: float = 100.0,
        onset_percentile: float = 50.0,
        min_gap_ms: float = 300.0,
        search_window_ms: float = 3000.0,
        global_offset_correction: bool = True,
    ):
        self.target_sr = target_sr
        self.hop_ms = hop_ms
        self.frame_ms = frame_ms
        self.onset_percentile = onset_percentile
        self.min_gap_ms = min_gap_ms
        self.search_window_ms = search_window_ms
        self.global_offset_correction = global_offset_correction

    @property
    def name(self) -> str:
        return "structural"

    @property
    def model_name(self) -> str:
        return "rms-onset"

    def settings(self) -> Dict[str, Any]:
        return {
            "target_sr": self.target_sr,
            "hop_ms": self.hop_ms,
            "frame_ms": self.frame_ms,
            "onset_percentile": self.onset_percentile,
            "min_gap_ms": self.min_gap_ms,
            "search_window_ms": self.search_window_ms,
            "global_offset_correction": self.global_offset_correction,
        }

    def align(
        self,
        audio_path: str,
        lyrics: StructuredLyrics,
        language: Optional[str] = None,
        vocals_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AlignmentResult:
        """Return an AlignmentResult with predicted line/word times.

        The predicted times are the best matching onset events in the audio.
        When ``global_offset_correction`` is enabled, a robust median signed
        offset is estimated and applied so that the source lyric timeline and
        the structural audio timeline are aligned.
        """
        try:
            onset_strength, times_ms = _energy_onset_curve(
                audio_path,
                self.target_sr,
                self.hop_ms,
                self.frame_ms,
            )
            # If a separated vocal track is available, add its onset strength.
            # This catches soft vocal sections (e.g. song outros) that have no
            # strong music onset, while still using the full mix for rhythmic
            # energy in the main body of the song.
            if vocals_path:
                vocal_onset_strength, vocal_times_ms = _energy_onset_curve(
                    vocals_path,
                    self.target_sr,
                    self.hop_ms,
                    self.frame_ms,
                )
                # Resample to the same time grid if lengths differ.
                if len(vocal_onset_strength) != len(onset_strength):
                    target_len = len(onset_strength)
                    from scipy.signal import resample

                    vocal_onset_strength = resample(
                        vocal_onset_strength.astype(float), target_len
                    )
                    # If resampling produced negative values from the interpolation,
                    # clip to non-negative.
                    if isinstance(vocal_onset_strength, np.ndarray):
                        np.maximum(vocal_onset_strength, 0.0, out=vocal_onset_strength)
                onset_strength = onset_strength + vocal_onset_strength
            onsets = _extract_onset_times(
                onset_strength,
                times_ms,
                self.onset_percentile,
                self.min_gap_ms,
            )
        except Exception as e:
            return AlignmentResult(
                aligner_name=self.name,
                model_name=self.model_name,
                settings=self.settings(),
                lyrics=lyrics,
                error=f"Failed to compute structural onset curve: {e}",
            )

        if len(onsets) == 0:
            return AlignmentResult(
                aligner_name=self.name,
                model_name=self.model_name,
                settings=self.settings(),
                lyrics=lyrics,
                error="No structural onsets detected in audio.",
            )

        mapped_lyrics, best_shift = self._match_lyrics_to_onsets(lyrics, onsets)

        return AlignmentResult(
            aligner_name=self.name,
            model_name=self.model_name,
            settings=self.settings(),
            lyrics=mapped_lyrics,
            estimated_global_offset_ms=float(best_shift),
        )

    def _match_lyrics_to_onsets(
        self, lyrics: StructuredLyrics, onsets: np.ndarray
    ) -> Tuple[StructuredLyrics, float]:
        """Match each source lyric line to the nearest monotonic onset.

        A robust median offset is estimated from the matched pairs so that a
        constant display offset (e.g. the intentional karaoke lead-in) is not
        mistaken for misalignment.
        """
        non_empty_lines: List[LyricLine] = [
            line for line in lyrics.lines if line.text.strip()
        ]

        # First pass: nearest monotonic onset for each source line.
        matched_onsets: List[Optional[float]] = []
        last_matched_idx = -1
        for line in non_empty_lines:
            if line.source_start_ms is None:
                matched_onsets.append(None)
                continue

            search_start = line.source_start_ms - self.search_window_ms
            search_end = line.source_start_ms + self.search_window_ms

            # Enforce monotonic matching: only consider onsets after the
            # previously matched onset, with a small grace window for tempo drift.
            if last_matched_idx >= 0:
                search_start = max(search_start, onsets[last_matched_idx] - 200.0)

            candidates = []
            for idx, t in enumerate(onsets):
                if idx <= last_matched_idx:
                    continue
                if t < search_start:
                    continue
                if t > search_end:
                    break
                candidates.append((idx, float(t)))

            if not candidates:
                matched_onsets.append(None)
                continue

            # Pick the closest onset to the source start time.
            best_idx, best_onset = min(
                candidates, key=lambda x: abs(x[1] - line.source_start_ms)
            )
            matched_onsets.append(best_onset)
            last_matched_idx = best_idx

        # Estimate a robust global offset from matched pairs.
        signed_errors: List[float] = []
        for line, onset in zip(non_empty_lines, matched_onsets):
            if onset is not None and line.source_start_ms is not None:
                signed_errors.append(onset - line.source_start_ms)

        robust_offset = 0.0
        if signed_errors:
            # Use a robust location estimator: median with MAD outlier clamping.
            median = float(np.median(signed_errors))
            mad = float(np.median(np.abs(np.array(signed_errors) - median)))
            inliers = [e for e in signed_errors if abs(e - median) <= 1.5 * mad + 1e-9]
            if inliers:
                robust_offset = float(np.median(inliers))

        # Build mapped lyric lines with predicted start/end.
        mapped_lines: List[LyricLine] = []
        for i, line in enumerate(lyrics.lines):
            mapped_line = LyricLine(
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

            # Find the line's index in non_empty_lines to get matched onset.
            try:
                ne_idx = non_empty_lines.index(line)
            except ValueError:
                mapped_lines.append(mapped_line)
                continue

            onset = matched_onsets[ne_idx]
            if onset is not None:
                predicted_start = onset
                if self.global_offset_correction:
                    predicted_start -= robust_offset
                mapped_line.predicted_start_ms = predicted_start

                # Derive a predicted line end.  Prefer the source duration if
                # known; otherwise fall back to the next matched onset.
                if line.source_end_ms is not None and line.source_start_ms is not None:
                    dur = line.source_end_ms - line.source_start_ms
                    mapped_line.predicted_end_ms = predicted_start + dur
                else:
                    next_onset = None
                    for j in range(ne_idx + 1, len(matched_onsets)):
                        if matched_onsets[j] is not None:
                            next_onset = matched_onsets[j]
                            if self.global_offset_correction:
                                next_onset -= robust_offset
                            break
                    if next_onset is not None:
                        mapped_line.predicted_end_ms = next_onset
                    elif line.source_end_ms is not None:
                        mapped_line.predicted_end_ms = (
                            predicted_start + (line.source_end_ms - line.source_start_ms)
                        )

                # Spread the same offset over the source words.
                if mapped_line.words:
                    for w in mapped_line.words:
                        if w.source_start_ms is not None and line.source_start_ms is not None:
                            rel = w.source_start_ms - line.source_start_ms
                            w.predicted_start_ms = predicted_start + rel
                        if w.source_end_ms is not None and line.source_start_ms is not None:
                            rel = w.source_end_ms - line.source_start_ms
                            w.predicted_end_ms = predicted_start + rel

            mapped_lines.append(mapped_line)

        # Preserve sort order by source start time.
        mapped_lines.sort(key=lambda ln: ln.source_start_ms or 0.0)

        return (
            StructuredLyrics(
                source=lyrics.source,
                source_track_id=lyrics.source_track_id,
                language=lyrics.language,
                lines=mapped_lines,
            ),
            robust_offset,
        )

    def _monotonic_match(
        self,
        lines: List[LyricLine],
        onsets: np.ndarray,
        shift: float,
    ) -> Tuple[List[Optional[float]], List[float]]:
        """Return matched onsets and signed errors for a constant source shift.

        ``shift`` is the audio lead relative to the source: we look for onsets
        near ``line.source_start_ms + shift``.
        """
        matched: List[Optional[float]] = []
        signed_errors: List[float] = []
        last_matched_idx = -1
        for line in lines:
            if line.source_start_ms is None:
                matched.append(None)
                continue

            target = line.source_start_ms + shift
            search_start = target - self.search_window_ms
            search_end = target + self.search_window_ms

            # Enforce monotonic matching.
            if last_matched_idx >= 0:
                search_start = max(search_start, onsets[last_matched_idx] - 200.0)

            candidates = []
            for idx, t in enumerate(onsets):
                if idx <= last_matched_idx:
                    continue
                if t < search_start:
                    continue
                if t > search_end:
                    break
                candidates.append((idx, float(t)))

            if not candidates:
                matched.append(None)
                continue

            best_idx, best_onset = min(
                candidates, key=lambda x: abs(x[1] - target)
            )
            matched.append(best_onset)
            last_matched_idx = best_idx
            signed_errors.append(best_onset - target)

        return matched, signed_errors
