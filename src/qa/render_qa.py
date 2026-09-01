"""Render QA: verify that generated ASS/final video reflects predicted sync."""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.musixmatch_lyrics import ASSKaraokeGenerator, SyncedLine
from src.qa.models import LyricLine, SyncReport


@dataclass
class RenderEvent:
    """One rendered subtitle event found in ASS output."""

    start_ms: float
    end_ms: float
    style: str
    text: str
    raw: str


@dataclass
class RenderQAResult:
    """Result of a render QA pass."""

    verified: bool
    ass_path: Optional[str]
    total_lines: int
    matched_lines: int
    unmatched_lines: int
    max_timing_error_ms: float
    mean_timing_error_ms: float
    errors: List[str] = field(default_factory=list)
    events: List[RenderEvent] = field(default_factory=list)
    line_results: List[dict] = field(default_factory=list)


def normalize_lyric_text(text: str) -> str:
    """Lowercase and remove all whitespace for text matching.

    ASS karaoke text may have spaces and syllable timing tags.  Removing
    whitespace makes it possible to compare a source lyric like
    'She loves you' against an ASS string like 'S h e l o v e s y o u'.

    Parentheticals and bracketed asides are also removed because the ASS
    generator splits them into separate chorus events, leaving the main line
    text without them.
    """
    text = (text or "").lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def _ass_time_to_ms(t: str) -> float:
    """Convert ASS timestamp 'H:MM:SS.cc' to milliseconds."""
    parts = t.strip().split(":")
    h = float(parts[0]) if parts else 0.0
    m = float(parts[1]) if len(parts) > 1 else 0.0
    s_cs = parts[2].split(".") if len(parts) > 2 else ["0", "0"]
    s = float(s_cs[0])
    cs = float(s_cs[1]) if len(s_cs) > 1 else 0.0
    return ((h * 3600) + (m * 60) + s + cs / 100.0) * 1000.0


def _strip_ass_tags(text: str) -> str:
    """Remove ASS override tags and karaoke timing tags, keep the visible text."""
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace("\\N", " ")
    text = text.replace("\\n", " ")
    return " ".join(text.split())


def parse_ass_events(ass_content: str) -> List[RenderEvent]:
    """Parse Dialogue lines from an ASS string."""
    events: List[RenderEvent] = []
    in_events = False
    for raw in ass_content.splitlines():
        raw = raw.strip()
        if raw.startswith("[Events]"):
            in_events = True
            continue
        if raw.startswith("[") and raw.endswith("]"):
            in_events = False
            continue
        if not in_events or not raw.startswith("Dialogue:"):
            continue

        body = raw[len("Dialogue:") :].strip()
        # ASS event format:
        # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        parts = body.split(",", 9)
        if len(parts) < 10:
            continue

        style = parts[3].strip()
        # Skip shadow layers; we only need one visible line per lyric.
        if style.lower().startswith("shadow"):
            continue

        start_ms = _ass_time_to_ms(parts[1])
        end_ms = _ass_time_to_ms(parts[2])
        text = _strip_ass_tags(parts[9])
        if not text:
            continue

        events.append(
            RenderEvent(start_ms=start_ms, end_ms=end_ms, style=style, text=text, raw=raw)
        )
    return events


def _ms_to_ass_time(ms: float) -> str:
    """Convert milliseconds to ASS timestamp format."""
    total_seconds = ms / 1000.0
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    # centiseconds
    cs = int(round((s - int(s)) * 100))
    s = int(s)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _lyrics_to_richsync_json(report: SyncReport) -> Optional[str]:
    """Convert a StructuredLyrics report to a Richsync JSON string.

    Returns None if not enough word timings are available.
    """
    richsync: List[Dict[str, Any]] = []
    for i, line in enumerate(report.lyrics.lines):
        start_ms = line.predicted_start_ms
        if start_ms is None:
            start_ms = line.source_start_ms
        end_ms = line.predicted_end_ms
        if end_ms is None:
            end_ms = line.source_end_ms
        if start_ms is None:
            continue

        if end_ms is None or end_ms <= start_ms:
            if i + 1 < len(report.lyrics.lines):
                next_line = report.lyrics.lines[i + 1]
                next_start = next_line.predicted_start_ms or next_line.source_start_ms
                if next_start is not None and next_start > start_ms:
                    end_ms = next_start
                else:
                    end_ms = start_ms + 2000.0
            else:
                end_ms = start_ms + 2000.0

        line_start_s = start_ms / 1000.0
        line_end_s = end_ms / 1000.0

        if not line.words:
            return None

        chars: List[Dict[str, Any]] = []
        for wi, word in enumerate(line.words):
            word_start = word.predicted_start_ms or word.source_start_ms
            if word_start is None:
                return None
            word_start_s = (word_start - start_ms) / 1000.0
            # Clamp word start to be inside the line; small floating-point
            # differences can push it slightly negative.
            word_start_s = max(0.0, word_start_s)

            for ch in word.text:
                chars.append({"c": ch, "o": round(word_start_s, 4)})
            # Add a space after the word, with the next word's offset if
            # available, otherwise the word's end offset.
            if wi < len(line.words) - 1:
                next_word = line.words[wi + 1]
                next_start = next_word.predicted_start_ms or next_word.source_start_ms
                if next_start is not None:
                    space_offset = max(0.0, (next_start - start_ms) / 1000.0)
                else:
                    space_offset = word_start_s
            else:
                # End of line: align space with the end of the line.
                space_offset = (line_end_s - line_start_s)
            chars.append({"c": " ", "o": round(space_offset, 4)})

        # Drop trailing space.
        if chars and chars[-1]["c"] == " ":
            chars.pop()

        richsync.append({"ts": round(line_start_s, 4), "te": round(line_end_s, 4), "l": chars})

    return json.dumps(richsync)


def build_ass_from_report(
    report: SyncReport,
    *,
    offset_seconds: float = 0.0,
    video_width: int = 1920,
    video_height: int = 1080,
    progressive_fill: bool = True,
    word_level: bool = True,
) -> str:
    """Regenerate ASS from a sync report's predicted lyric timings.

    If word-level timing is available, use the richsync renderer so each word
    is highlighted. Otherwise fall back to the synced-lines renderer.
    """
    if word_level:
        richsync_json = _lyrics_to_richsync_json(report)
        if richsync_json:
            return ASSKaraokeGenerator.generate_from_richsync(
                richsync_json=richsync_json,
                video_width=video_width,
                video_height=video_height,
                offset_seconds=offset_seconds,
                progressive_fill=progressive_fill,
            )

    synced_lines: List[SyncedLine] = []
    for i, line in enumerate(report.lyrics.lines):
        start_ms = line.predicted_start_ms
        if start_ms is None:
            start_ms = line.source_start_ms
        end_ms = line.predicted_end_ms
        if end_ms is None:
            end_ms = line.source_end_ms

        if start_ms is None:
            continue
        start = start_ms / 1000.0

        if end_ms is None or end_ms <= start_ms:
            # Use the next line's start as a fallback end, or a short default.
            if i + 1 < len(report.lyrics.lines):
                next_line = report.lyrics.lines[i + 1]
                next_start = next_line.predicted_start_ms or next_line.source_start_ms
                if next_start is not None and next_start > start_ms:
                    end_ms = next_start
                else:
                    end_ms = start_ms + 2000.0
            else:
                end_ms = start_ms + 2000.0
        end = end_ms / 1000.0
        synced_lines.append(SyncedLine(start=start, end=end, text=line.text))

    return ASSKaraokeGenerator.generate_from_synced_lines(
        synced_lines=synced_lines,
        video_width=video_width,
        video_height=video_height,
        offset_seconds=offset_seconds,
        progressive_fill=progressive_fill,
    )


def run_render_qa(
    report: SyncReport,
    *,
    offset_seconds: float = 0.0,
    video_width: int = 1920,
    video_height: int = 1080,
    progressive_fill: bool = True,
    timing_tolerance_ms: float = 150.0,
    text_match_threshold: float = 0.6,
) -> RenderQAResult:
    """Generate ASS from predicted timings and verify it matches the report."""
    try:
        ass_content = build_ass_from_report(
            report,
            offset_seconds=offset_seconds,
            video_width=video_width,
            video_height=video_height,
            progressive_fill=progressive_fill,
        )
    except Exception as e:
        return RenderQAResult(
            verified=False,
            ass_path=None,
            total_lines=0,
            matched_lines=0,
            unmatched_lines=0,
            max_timing_error_ms=0.0,
            mean_timing_error_ms=0.0,
            errors=[f"ASS generation failed: {e}"],
        )

    events = parse_ass_events(ass_content)
    if not events:
        return RenderQAResult(
            verified=False,
            ass_path=None,
            total_lines=0,
            matched_lines=0,
            unmatched_lines=0,
            max_timing_error_ms=0.0,
            mean_timing_error_ms=0.0,
            errors=["No Dialogue events found in generated ASS."],
        )

    predicted_lines = [
        ln for ln in report.lyrics.lines if ln.predicted_start_ms is not None
    ]
    total = len(predicted_lines)
    matched = 0
    unmatched = 0
    errors: List[str] = []
    line_results: List[dict] = []
    timing_errors: List[float] = []

    # Keep the search window tight: the rendered event for a line should start
    # very close to the predicted time.  A 1-second window is enough to absorb
    # normal renderer delays without accidentally matching a repeated chorus
    # line or a trailing ad-lib.
    time_window_ms = max(timing_tolerance_ms * 3, 1000.0)
    used_events: set[int] = set()

    for line in predicted_lines:
        predicted_start = line.predicted_start_ms or 0.0
        line_text = normalize_lyric_text(line.text)

        # One-to-one matching: each ASS event can only verify one lyric line.
        # Process lines in source order and prefer the earliest matching unused
        # event inside the time window.
        candidates = []
        for i, event in enumerate(events):
            if i in used_events:
                continue
            if abs(event.start_ms - predicted_start) > time_window_ms:
                continue
            text_score = _text_similarity(line_text, normalize_lyric_text(event.text))
            if text_score < text_match_threshold:
                continue
            candidates.append((text_score, event.start_ms, i, event))

        best: Optional[RenderEvent] = None
        if candidates:
            # Sort by text score descending, then start time ascending.
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best = candidates[0][3]
            used_events.add(candidates[0][2])

        if best is None:
            unmatched += 1
            errors.append(f"No ASS event found for line: {line.text[:40]}")
            line_results.append(
                {
                    "text": line.text,
                    "predicted_start_ms": predicted_start,
                    "matched": False,
                    "error": "no matching event",
                }
            )
            continue

        timing_error = abs(best.start_ms - predicted_start)
        timing_errors.append(timing_error)
        matched += 1

        if timing_error > timing_tolerance_ms:
            errors.append(
                f"Large timing error ({timing_error:.0f} ms) for line: {line.text[:40]}"
            )

        line_results.append(
            {
                "text": line.text,
                "predicted_start_ms": predicted_start,
                "ass_start_ms": best.start_ms,
                "ass_end_ms": best.end_ms,
                "ass_text": best.text,
                "matched": True,
                "timing_error_ms": timing_error,
            }
        )

    max_error = max(timing_errors) if timing_errors else 0.0
    mean_error = sum(timing_errors) / len(timing_errors) if timing_errors else 0.0

    verified = (
        total > 0
        and matched == total
        and not errors
        and max_error <= timing_tolerance_ms
    )

    return RenderQAResult(
        verified=verified,
        ass_path=None,
        total_lines=total,
        matched_lines=matched,
        unmatched_lines=unmatched,
        max_timing_error_ms=max_error,
        mean_timing_error_ms=mean_error,
        errors=errors,
        events=events,
        line_results=line_results,
    )


def _text_similarity(a: str, b: str) -> float:
    """Compare normalized lyric strings.

    Returns 1.0 if one is a substring of the other, otherwise the character
    overlap ratio.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    a_set = set(a)
    b_set = set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(len(a_set), len(b_set))
