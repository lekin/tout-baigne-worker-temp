"""Configuration management for the karaoke application."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Airtable Configuration
    airtable_api_key: str
    airtable_base_id: str
    musixmatch_api_key: Optional[str] = None
    airtable_table_name: str = "Tracks"
    
    # Karaoke Configuration
    karaoke_model_size: str = "large"
    karaoke_video_width: int = 1920
    karaoke_video_height: int = 1080
    karaoke_logo_width: int = 260
    karaoke_logo_height: int = 224
    karaoke_font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf"
    karaoke_logo_path: str = "Logo-YellowDropShadow.png"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Output Configuration
    output_dir: str = "output"
    
    # Genius API
    genius_api_token: Optional[str] = None
    
    # Lalal.ai API
    lalal_api_key: Optional[str] = None
    
    # Musixmatch API
    musixmatch_api_key: Optional[str] = None

    # Spotify API (optional)
    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    spotify_user_access_token: Optional[str] = None
    spotify_refresh_token: Optional[str] = None

    pennylane_access_token: Optional[str] = None
    pennylane_base_url: str = "https://app.pennylane.com/api/external/v2"

    # Twilio WhatsApp Notifications (optional)
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_from: Optional[str] = None
    twilio_whatsapp_to: Optional[str] = None

    # QA Configuration (Phase A)
    qa_output_dir: str = "output/qa"
    qa_cache_dir: str = "output/qa/cache"
    qa_aligner_model: str = "base"  # Whisper model size used by Stable-ts aligner
    qa_aligner_device: str = "cpu"
    qa_aligner_compute_type: str = "int8"
    # Vocal separation backend: demucs (legacy alias = demucs_cpu), demucs_cpu,
    # demucs_mps, mlx, ffmpeg, or auto (prefer mlx when available).
    qa_vocal_separator: str = "demucs"
    qa_vocal_model: str = "htdemucs"  # htdemucs, hdemucs_mmi, ...
    qa_vocal_shifts: int = 0
    qa_vocal_overlap: float = 0.25
    qa_vocal_split: bool = True
    qa_mlx_venv: str = ".venv_bench_mlx"
    qa_max_line_start_median_ms: float = 200.0
    qa_max_line_start_p90_ms: float = 300.0
    qa_max_line_start_single_ms: float = 1500.0
    qa_min_word_coverage: float = 0.85
    qa_min_line_coverage: float = 0.85
    qa_max_unresolved_lyric_region_ms: float = 1200.0
    qa_max_allowed_audio_tail_ms: float = 30000.0
    qa_max_drift_slope: float = 0.005
    qa_diagnostics_verbose: bool = False

    def get_qa_output_path(self) -> Path:
        path = Path(self.qa_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_qa_cache_path(self) -> Path:
        path = Path(self.qa_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_output_path(self) -> Path:
        """Get the output directory path, creating it if necessary."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
