"""Karaoke QA package."""
from src.qa.separator import (
    FFmpegVocalSeparator,
    MLXDemucsSeparator,
    PyTorchDemucsSeparator,
    VocalSeparationConfig,
    VocalSeparator,
    get_vocal_separator,
)

import warnings

# Suppress non-actionable torchaudio 2.9 deprecation warnings about future
# torchcodec migration. The warning does not affect functionality and the
# underlying API is still supported.
warnings.filterwarnings(
    "ignore",
    message=".*torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*save_with_torchcodec.*",
    category=UserWarning,
)
