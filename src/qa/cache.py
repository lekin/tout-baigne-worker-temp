"""Content-addressed caching for QA artifacts."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _hash_dict(d: Dict[str, Any]) -> str:
    return _hash_text(json.dumps(d, sort_keys=True, ensure_ascii=True))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_qa_cache_dir() -> Path:
    return _ensure_dir(Path(settings.qa_cache_dir))


def get_audio_cache_path() -> Path:
    return _ensure_dir(get_qa_cache_dir() / "audio")


def get_vocals_cache_path() -> Path:
    return _ensure_dir(get_qa_cache_dir() / "vocals")


def get_alignment_cache_path() -> Path:
    return _ensure_dir(get_qa_cache_dir() / "alignments")


def hash_audio_file(audio_path: str) -> str:
    """Return a content hash for an audio file."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:32]


def hash_lyrics(lyrics_dict: Dict[str, Any]) -> str:
    """Return a content hash for normalized lyrics."""
    return _hash_dict(lyrics_dict)


def vocals_cache_key(
    audio_path: str,
    separator: Optional[str] = None,
    model: str = "htdemucs",
    config: Optional[Any] = None,
) -> str:
    """Content-addressed cache key for a separated vocal stem.

    Args:
        audio_path: path to the source audio.
        separator: legacy/semi-legacy backend hint (e.g. "demucs", "ffmpeg").
        model: model name, e.g. "htdemucs".
        config: optional VocalSeparationConfig with full backend/model/version/settings.

    Existing CPU Demucs stems produced by the old key remain valid because the
    default ``demucs`` backend still hashes to the historical value.
    """
    audio_hash = hash_audio_file(audio_path)

    if config is not None:
        # Full modern cache key.
        settings_hash = _hash_dict(config.identity_dict())
    elif separator == "demucs":
        # Preserve the original key so existing CPU htdemucs stems stay valid.
        settings_hash = _hash_dict({"separator": "demucs", "model": model})
    else:
        # Generic fallback for other backends (ffmpeg, mlx, ...).
        settings_hash = _hash_dict({"separator": separator or "demucs", "model": model})

    return f"{audio_hash}_{settings_hash}"


def alignment_cache_key(
    audio_path: str,
    lyrics_dict: Dict[str, Any],
    aligner_name: str,
    model_name: str,
    aligner_settings: Dict[str, Any],
    vocals_path: Optional[str] = None,
) -> str:
    """Content-addressed cache key for an alignment result."""
    audio_hash = hash_audio_file(audio_path)
    lyrics_hash = hash_lyrics(lyrics_dict)
    vocals_hash = hash_audio_file(vocals_path) if vocals_path else "none"
    settings_hash = _hash_dict(
        {
            "aligner": aligner_name,
            "model": model_name,
            "vocals": vocals_hash,
            **aligner_settings,
        }
    )
    return f"{audio_hash}_{lyrics_hash}_{settings_hash}"


def cached_vocals_path(cache_key: str) -> str:
    return str(get_vocals_cache_path() / f"{cache_key}_vocals.wav")


def cached_alignment_path(cache_key: str) -> str:
    return str(get_alignment_cache_path() / f"{cache_key}.json")


def cached_audio_path(cache_key: str, ext: str = ".mp3") -> str:
    return str(get_audio_cache_path() / f"{cache_key}{ext}")


def find_existing_cached_vocals(cache_key: str) -> Optional[str]:
    path = cached_vocals_path(cache_key)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    return None


def find_existing_cached_alignment(cache_key: str) -> Optional[str]:
    path = cached_alignment_path(cache_key)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    return None
