"""Pydantic models for AI analysis results."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TimestampSegment(BaseModel):
    """A single viral clip segment identified by the Analyst Agent."""

    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    reasoning: str = Field(..., min_length=1, description="Why this segment is viral-worthy")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence score")
    hook_type: Optional[str] = Field(default=None, description="Type of hook: storytelling, plot_twist, comedy, emotion, quote")

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_time", 0)
        if v <= start:
            raise ValueError("end_time must be greater than start_time")
        if v - start > 300:  # Max 5 minutes per clip
            raise ValueError("segment duration cannot exceed 300 seconds (5 minutes)")
        return v

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class AnalysisResult(BaseModel):
    """Complete analysis result from the Content Analyst Agent."""

    segments: List[TimestampSegment] = Field(default_factory=list)
    video_title: Optional[str] = Field(default=None)
    video_duration: Optional[float] = Field(default=None)
    overall_summary: Optional[str] = Field(default=None)

    @field_validator("segments")
    @classmethod
    def max_segments(cls, v: List[TimestampSegment]) -> List[TimestampSegment]:
        if len(v) > 5:
            raise ValueError("maximum 5 segments allowed per analysis")
        return v


class FeedbackEntry(BaseModel):
    """User feedback entry for continuous improvement."""

    video_url: str
    segment_index: int
    rating: int = Field(..., ge=-1, le=1, description="-1=thumbs down, 0=neutral, 1=thumbs up")
    user_id: int
    timestamp: str
    reasoning: Optional[str] = None
