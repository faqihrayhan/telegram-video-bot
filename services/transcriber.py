"""Agent 3: Audio Transcriber & Subtitle Generator

Uses Groq Whisper API for speech-to-text with word-level timestamps.
Generates styled ASS subtitle files.
"""
import os
import asyncio
from pathlib import Path
from typing import Optional

from groq import Groq

from models import AppConfig, TranscriptionResult, WordTiming, SubtitleStyle
from utils import ASSBuilder


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class TranscriberAgent:
    """Agent that transcribes audio and generates subtitles."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = Groq(api_key=config.groq_api_key)

    async def transcribe(self, audio_path: Path, 
                         segment_start: float = 0,
                         segment_end: Optional[float] = None) -> TranscriptionResult:
        """Transcribe audio segment with word-level timestamps.

        Args:
            audio_path: Path to audio file.
            segment_start: Start time offset in seconds (for trimming context).
            segment_end: End time offset in seconds.

        Returns:
            TranscriptionResult with word timings.

        Raises:
            TranscriptionError: If transcription fails.
        """
        if not audio_path.exists():
            raise TranscriptionError(f"❌ File audio tidak ditemukan: {audio_path}")

        # Check file size (Groq limit ~25MB for audio)
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            raise TranscriptionError(
                f"❌ File audio terlalu besar ({file_size_mb:.1f} MB). "
                "Maksimum 25 MB untuk Groq Whisper."
            )

        loop = asyncio.get_event_loop()

        def _transcribe():
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.config.groq_whisper_model,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    language=None,  # Auto-detect
                )
            return response

        try:
            response = await loop.run_in_executor(None, _transcribe)
        except Exception as e:
            raise TranscriptionError(f"❌ Groq Whisper error: {str(e)}")

        # Parse response
        words = []
        for word_data in getattr(response, "words", []) or []:
            try:
                word = WordTiming(
                    word=word_data.word.strip(),
                    start=float(word_data.start) + segment_start,
                    end=float(word_data.end) + segment_start,
                    confidence=getattr(word_data, "probability", None)
                )
                words.append(word)
            except (ValueError, AttributeError):
                continue

        full_text = " ".join(w.word for w in words)
        duration = words[-1].end if words else 0.0

        return TranscriptionResult(
            words=words,
            full_text=full_text,
            language=getattr(response, "language", None),
            duration=duration
        )

    async def generate_subtitle(self, transcription: TranscriptionResult,
                                   output_path: Path,
                                   style: Optional[SubtitleStyle] = None) -> Path:
        """Generate ASS subtitle file from transcription.

        Args:
            transcription: Transcription result with word timings.
            output_path: Where to save .ass file.
            style: Optional custom style. Uses default if None.

        Returns:
            Path to generated ASS file.
        """
        builder = ASSBuilder(style=style)
        return builder.build_ass(transcription, output_path)
