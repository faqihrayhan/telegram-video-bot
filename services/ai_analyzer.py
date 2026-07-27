"""Agent 2: Content Analyst / Viral Spotter

Uses Google Gemini to analyze video content and identify viral-worthy segments.
Includes retry logic, circuit breaker, and fallback heuristics.
"""
import json
import asyncio
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from models import AppConfig, AnalysisResult, TimestampSegment


class AnalysisError(Exception):
    """Raised when analysis fails."""
    pass


class CircuitBreaker:
    """Simple circuit breaker for external API calls."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if self.last_failure_time and                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"


class ContentAnalystAgent:
    """Agent that analyzes video content to find viral segments."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120)
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini API client."""
        genai.configure(api_key=self.config.gemini_api_key)
        self.model = genai.GenerativeModel(self.config.gemini_model)

    async def analyze(self, video_path: Path, audio_path: Path, 
                      metadata: dict) -> AnalysisResult:
        """Analyze video and return viral segment timestamps.

        Args:
            video_path: Path to downloaded video file.
            audio_path: Path to downloaded audio file.
            metadata: Video metadata dict.

        Returns:
            AnalysisResult with recommended segments.

        Raises:
            AnalysisError: If analysis fails after retries.
        """
        if not self.circuit_breaker.can_execute():
            # Fallback to heuristic if circuit is open
            return self._heuristic_fallback(metadata)

        # Retry logic
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                result = await self._call_gemini(video_path, metadata)
                self.circuit_breaker.record_success()
                return result

            except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable):
                self.circuit_breaker.record_failure()
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    return self._heuristic_fallback(metadata)

            except Exception:
                self.circuit_breaker.record_failure()
                if attempt == max_retries - 1:
                    return self._heuristic_fallback(metadata)
                await asyncio.sleep(base_delay)

        return self._heuristic_fallback(metadata)

    async def _call_gemini(self, video_path: Path, metadata: dict) -> AnalysisResult:
        """Call Gemini API with video for analysis."""
        loop = asyncio.get_event_loop()

        def _analyze():
            # Upload video to Gemini
            video_file = genai.upload_file(str(video_path), mime_type="video/mp4")

            # Wait for processing
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise AnalysisError("❌ Gemini gagal memproses video.")

            prompt = self._build_prompt(metadata)

            response = self.model.generate_content(
                [video_file, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                )
            )

            # Clean up uploaded file
            try:
                genai.delete_file(video_file.name)
            except:
                pass

            return response.text

        try:
            response_text = await loop.run_in_executor(None, _analyze)
            return self._parse_response(response_text, metadata)
        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Gemini API error: {str(e)}")

    def _build_prompt(self, metadata: dict) -> str:
        """Build analysis prompt for Gemini."""
        duration = metadata.get("duration", 0)

        return f"""Analyze this video and identify the most engaging/viral-worthy segments.

Video info:
- Title: {metadata.get("title", "Unknown")}
- Duration: {duration} seconds ({duration/60:.1f} minutes)
- Uploader: {metadata.get("uploader", "Unknown")}

Your task:
1. Watch the entire video and understand the content flow
2. Identify 1-3 segments that have HIGH viral potential for TikTok/Reels/Shorts
3. Look for: strong hooks, emotional peaks, plot twists, comedy moments, motivational quotes, surprising facts
4. Each segment should be 15-60 seconds long (ideal: 30-45 seconds)

Return ONLY a JSON object in this exact format:
{{
  "segments": [
    {{
      "start_time": <float, seconds>,
      "end_time": <float, seconds>,
      "reasoning": "<why this segment is viral-worthy, 1-2 sentences>",
      "confidence": <float 0.0-1.0>,
      "hook_type": "<storytelling|plot_twist|comedy|emotion|quote|educational>"
    }}
  ],
  "overall_summary": "<brief summary of video content>"
}}

Rules:
- start_time must be >= 0
- end_time must be > start_time
- Each segment max 60 seconds
- Total segments: 1-3
- Be precise with timestamps"""

    def _parse_response(self, text: str, metadata: dict) -> AnalysisResult:
        """Parse Gemini JSON response into AnalysisResult."""
        try:
            # Clean up response text
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)

            # Validate and create segments
            segments = []
            for seg_data in data.get("segments", []):
                try:
                    seg = TimestampSegment(
                        start_time=float(seg_data["start_time"]),
                        end_time=float(seg_data["end_time"]),
                        reasoning=str(seg_data.get("reasoning", "Viral segment")),
                        confidence=float(seg_data.get("confidence", 0.8)),
                        hook_type=seg_data.get("hook_type")
                    )
                    segments.append(seg)
                except (ValueError, KeyError):
                    continue

            return AnalysisResult(
                segments=segments,
                video_title=metadata.get("title"),
                video_duration=metadata.get("duration"),
                overall_summary=data.get("overall_summary")
            )

        except json.JSONDecodeError as e:
            raise AnalysisError(f"❌ Gemini mengembalikan format JSON tidak valid: {str(e)}")

    def _heuristic_fallback(self, metadata: dict) -> AnalysisResult:
        """Fallback heuristic when AI analysis fails.

        Takes the middle portion of the video as best guess.
        """
        duration = metadata.get("duration", 300)

        # Take middle 30-45 seconds
        mid = duration / 2
        start = max(0, mid - 20)
        end = min(duration, mid + 25)

        segment = TimestampSegment(
            start_time=start,
            end_time=end,
            reasoning="Fallback: Mengambil bagian tengah video karena analisis AI tidak tersedia.",
            confidence=0.5,
            hook_type="unknown"
        )

        return AnalysisResult(
            segments=[segment],
            video_title=metadata.get("title"),
            video_duration=duration,
            overall_summary="Analisis otomatis (fallback mode)"
        )
