"""Pydantic models for subtitle and transcription data."""
from typing import List, Optional
from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    """Single word with precise timing from Whisper."""

    word: str = Field(..., min_length=1)
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @property
    def duration(self) -> float:
        return self.end - self.start


class SubtitleStyle(BaseModel):
    """Visual styling configuration for ASS subtitles."""

    font_name: str = Field(default="Arial")
    font_size: int = Field(default=48, ge=12, le=120)
    primary_color: str = Field(default="&H00FFFFFF")  # White in ASS format
    secondary_color: str = Field(default="&H00FFFF00")  # Yellow highlight
    outline_color: str = Field(default="&H00000000")  # Black outline
    back_color: str = Field(default="&H80000000")  # Semi-transparent black
    bold: bool = Field(default=True)
    outline: int = Field(default=2, ge=0, le=4)
    shadow: int = Field(default=1, ge=0, le=4)
    alignment: int = Field(default=2, ge=1, le=9)  # 2 = bottom center
    margin_v: int = Field(default=60, ge=0)

    # Word-highlight effect
    highlight_current_word: bool = Field(default=True)
    highlight_scale: float = Field(default=1.15, ge=1.0, le=2.0)


class TranscriptionResult(BaseModel):
    """Complete transcription result from the Transcriber Agent."""

    words: List[WordTiming] = Field(default_factory=list)
    full_text: str = Field(default="")
    language: Optional[str] = Field(default=None)
    duration: float = Field(default=0.0, ge=0)

    @property
    def word_count(self) -> int:
        return len(self.words)
