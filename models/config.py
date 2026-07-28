"""Pydantic models for application configuration."""
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class AppMode(str, Enum):
    LOCAL = "LOCAL"
    VPS = "VPS"


class AppConfig(BaseModel):
    """Central configuration parsed from environment variables."""

    app_mode: AppMode = Field(default=AppMode.LOCAL)
    telegram_bot_token: str = Field(..., min_length=1)
    gemini_api_key: str = Field(..., min_length=1)
    groq_api_key: str = Field(..., min_length=1)

    max_video_duration_minutes: int = Field(default=120, ge=1, le=180)
    max_file_size_mb: int = Field(default=500, ge=10, le=2000)
    max_concurrent_jobs: int = Field(default=2, ge=1, le=10)

    # Gemini video analysis settings
    gemini_max_duration_minutes: int = Field(default=120, ge=1, le=180)
    gemini_frame_sampling_interval: int = Field(default=5, ge=1, le=30,
        description="Extract 1 frame every N seconds for Gemini. Higher = fewer tokens.")

    gemini_model: str = Field(default="gemini-2.0-flash")
    groq_whisper_model: str = Field(default="whisper-large-v3")

    @property
    def temp_base_path(self) -> Path:
        """Resolve temp directory based on APP_MODE."""
        if self.app_mode == AppMode.LOCAL:
            return Path("D:/telegram-video-bot-temp")
        return Path("/tmp/telegram-video-bot")

    @property
    def auto_cleanup(self) -> bool:
        """Auto-delete temp files after processing in VPS mode."""
        return self.app_mode == AppMode.VPS

    @property
    def ffmpeg_preset(self) -> str:
        """FFmpeg encoding preset based on mode."""
        return "ultrafast" if self.app_mode == AppMode.VPS else "medium"

    @property
    def db_path(self) -> Path:
        """SQLite database path for feedback and cache."""
        return self.temp_base_path / "clipper.db"

    @field_validator("app_mode", mode="before")
    @classmethod
    def normalize_mode(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v

    def model_copy(self, update: dict = None) -> "AppConfig":
        """Create a copy with optional field overrides."""
        data = self.model_dump()
        if update:
            data.update(update)
        return AppConfig(**data)
