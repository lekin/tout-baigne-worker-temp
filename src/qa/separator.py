"""Vocal-separation backends for QA with content-addressed caching."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import torch
import torchaudio

from demucs.pretrained import get_model
from demucs.apply import apply_model

from src.config import settings
from src.karaoke_generator import KaraokeGenerator


@dataclass
class VocalSeparationConfig:
    """Configuration for one vocal-separation run.

    This is the identity of the vocal stem used for the QA cache key.
    """

    backend: str = "demucs_cpu"  # demucs_cpu, demucs_mps, mlx, ffmpeg
    model: str = "htdemucs"      # htdemucs, hdemucs_mmi, ...
    package_version: Optional[str] = None
    shifts: int = 0
    overlap: float = 0.25
    split: bool = True
    device: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def identity_dict(self) -> Dict[str, Any]:
        d = {
            "backend": self.backend,
            "model": self.model,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "split": self.split,
        }
        if self.package_version:
            d["package_version"] = self.package_version
        if self.extra:
            d["extra"] = self.extra
        return d

    def cache_key(self, audio_path: Union[str, Path]) -> str:
        """Return the content-addressed cache key for this stem."""
        from src.qa.cache import hash_audio_file, _hash_dict

        audio_hash = hash_audio_file(str(audio_path))
        settings_hash = _hash_dict(self.identity_dict())
        return f"{audio_hash}_{settings_hash}"

    def cached_path(self, audio_path: Union[str, Path]) -> Path:
        from src.qa.cache import get_vocals_cache_path

        return get_vocals_cache_path() / f"{self.cache_key(audio_path)}_vocals.wav"


class VocalSeparator(Protocol):
    """Backend that can separate a vocal stem and write it to a file."""

    name: str

    def separate(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        config: VocalSeparationConfig,
    ) -> bool:
        """Separate vocals from *audio_path* and write a WAV to *output_path*.

        Returns True on success. The caller is responsible for cache key logic.
        """
        ...


class PyTorchDemucsSeparator:
    """Demucs via the installed PyTorch demucs package."""

    name = "pytorch_demucs"

    def __init__(self, device: Optional[str] = None):
        self.device = device

    def separate(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        config: VocalSeparationConfig,
    ) -> bool:
        try:
            model = get_model(config.model)
            device = torch.device(self._resolve_device())
            model.to(device)
            model.eval()

            wav, sr = torchaudio.load(str(audio_path))
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            if sr != model.samplerate:
                wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
                sr = model.samplerate

            with torch.no_grad():
                wav_input = wav.unsqueeze(0).to(device)
                # The original code catches MPS failure and falls back to CPU.
                try:
                    sources = apply_model(
                        model,
                        wav_input,
                        device=device,
                        shifts=config.shifts,
                        split=config.split,
                        overlap=config.overlap,
                    )[0]
                except (NotImplementedError, RuntimeError) as e:
                    if device.type != "cpu":
                        print(f"⚠️ {device.type.upper()} failed ({e}), falling back to CPU...")
                        device = torch.device("cpu")
                        model.to(device)
                        wav_input = wav_input.cpu()
                        sources = apply_model(
                            model,
                            wav_input,
                            device=device,
                            shifts=config.shifts,
                            split=config.split,
                            overlap=config.overlap,
                        )[0]
                    else:
                        raise

            # source index for "vocals" may differ per model; htdemucs/hdemucs_mmi use 4-source.
            vocals = sources[3]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), vocals.cpu(), sr)
            return os.path.getsize(output_path) > 1000
        except Exception as e:
            print(f"❌ PyTorch Demucs separation failed: {e}")
            return False

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


class FFmpegVocalSeparator:
    """Fast center-channel extraction via FFmpeg; lower quality, CPU only."""

    name = "ffmpeg"

    def separate(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        config: VocalSeparationConfig,
    ) -> bool:
        path = KaraokeGenerator.separate_vocals_ffmpeg(
            str(audio_path),
            output_dir=str(Path(output_path).parent),
            keep_files=True,
        )
        if not path:
            return False
        if Path(path) != Path(output_path):
            os.replace(path, output_path)
        return os.path.getsize(output_path) > 1000


class MLXDemucsSeparator:
    """Demucs via an isolated MLX virtual environment.

    The production venv does not need MLX dependencies; the helper script in
    the isolated venv does the work and writes the requested output path.
    """

    name = "mlx_demucs"

    def __init__(self, mlx_venv: Optional[Union[str, Path]] = None):
        repo_root = Path(__file__).resolve().parents[2]
        self.mlx_venv = Path(mlx_venv) if mlx_venv else repo_root / ".venv_bench_mlx"

    def separate(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        config: VocalSeparationConfig,
    ) -> bool:
        script = Path(__file__).resolve().parents[2] / "scripts" / "separate_mlx.py"
        if not script.exists():
            print(f"❌ MLX helper script not found: {script}")
            return False

        python = self.mlx_venv / "bin" / "python"
        if not python.exists():
            print(f"❌ MLX venv not found: {self.mlx_venv}")
            return False

        cmd = [
            str(python),
            str(script),
            "--audio",
            str(audio_path),
            "--output",
            str(output_path),
            "--model",
            config.model,
            "--shifts",
            str(config.shifts),
            "--overlap",
            str(config.overlap),
        ]
        if config.split:
            cmd.append("--split")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return os.path.getsize(output_path) > 1000
        except subprocess.CalledProcessError as e:
            print(f"❌ MLX Demucs separation failed: {e}")
            return False


def get_vocal_separator(config: VocalSeparationConfig, mlx_venv: Optional[Path] = None) -> VocalSeparator:
    """Return the appropriate separator for the given configuration."""
    if config.backend == "ffmpeg":
        return FFmpegVocalSeparator()
    if config.backend.startswith("mlx"):
        return MLXDemucsSeparator(mlx_venv=mlx_venv)
    return PyTorchDemucsSeparator(device=config.device)
