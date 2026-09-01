"""Canonical timeline transformations for karaoke QA."""
from typing import Any, Dict, Optional
from src.qa.models import TimelineTransform


def _float_field(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def build_timeline_transform(record: Dict[str, Any]) -> TimelineTransform:
    """
    Build a TimelineTransform from an Airtable record.

    The canonical QA timeline is the source audio timeline. All comparisons
    between predicted (aligner) timings and source (Musixmatch) timings are
    performed on this timeline.

    Offsets:
      * lyrics_to_source_offset_ms: Musixmatch -> source audio. The generator
        adds the 'Lyrics to singing offset (s)' field to Musixmatch timings, so
        audio = musixmatch + lyrics_offset. Therefore the transform from
        Musixmatch to source audio is +lyrics_offset * 1000.
      * source_to_karaoke_audio_offset_ms: any generator-introduced shift between
        the downloaded source audio and the audio used in the rendered karaoke.
      * karaoke_audio_to_video_offset_ms: 'Audio to music video time offset (s)'
        converted to ms. This is only relevant for render QA; sync QA is done on
        the audio timeline.
    """
    fields = record.get("fields", record) if isinstance(record, dict) else {}

    lyrics_offset_s = _float_field(fields.get("Lyrics to singing offset (s)"))
    audio_to_video_offset_s = _float_field(
        fields.get("Audio to music video time offset (s)")
    )

    return TimelineTransform(
        lyrics_to_source_offset_ms=lyrics_offset_s * 1000.0,
        source_to_karaoke_audio_offset_ms=0.0,
        karaoke_audio_to_video_offset_ms=audio_to_video_offset_s * 1000.0,
    )


def apply_transform_to_source_ms(
    transform: TimelineTransform, raw_musixmatch_ms: float
) -> float:
    """Return the source-audio-timeline time for a raw Musixmatch timestamp."""
    return transform.musixmatch_to_source(raw_musixmatch_ms)


def source_ms_to_raw_musixmatch_ms(
    transform: TimelineTransform, source_ms: float
) -> float:
    return transform.source_to_musixmatch(source_ms)
