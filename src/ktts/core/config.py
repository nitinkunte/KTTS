from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    default_voice: str = "af_heart"
    model_auto_unload_timeout_seconds: float = 0.0
    model_dir: str = str(Path.home() / ".cache" / "ktts" / "models")
    voices_dir: str = str(Path.home() / ".cache" / "ktts" / "voices")
    sessions_dir: str = str(Path.home() / ".local" / "share" / "ktts" / "sessions")
    session_ttl_days: int = 14
    max_concurrent_sessions: int = 4
    pause_on_disconnect: bool = True
    model_repo_id: str = "hexgrad/Kokoro-82M"
    target_min_tokens: int = 175
    target_max_tokens: int = 250
    absolute_max_tokens: int = 450
    max_pause_duration_s: float = 60.0
    max_total_pause_s: float = 300.0
    advanced_text_normalization: bool = True
    gap_trim_ms: int = 1
    dynamic_gap_trim_padding_ms: int = 410
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
