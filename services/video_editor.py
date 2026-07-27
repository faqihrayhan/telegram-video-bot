"""Agent 4: Video Editor / Executor Worker

Handles FFmpeg operations: cut, crop 9:16, burn subtitle, encode.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from models import AppConfig, TimestampSegment
from utils.file_manager import TempFileManager


class EditError(Exception):
    """Raised when video editing fails."""
    pass


class VideoEditorAgent:
    """Agent that executes FFmpeg operations to produce final clips."""

    def __init__(self, config: AppConfig, file_manager: TempFileManager):
        self.config = config
        self.file_manager = file_manager
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verify FFmpeg is installed."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise EditError("❌ FFmpeg tidak terinstall dengan benar.")
        except FileNotFoundError:
            raise EditError("❌ FFmpeg tidak ditemukan. Install FFmpeg terlebih dahulu.")

    async def render_clip(self, video_path: Path, 
                          audio_path: Path,
                          segment: TimestampSegment,
                          subtitle_path: Optional[Path],
                          session_id: str,
                          segment_index: int,
                          preset: Optional[str] = None) -> Path:
        """Render a single clip: cut → crop 9:16 → burn subtitle → encode.

        Args:
            video_path: Source video file.
            audio_path: Source audio file.
            segment: Timestamp segment to extract.
            subtitle_path: Optional ASS subtitle file.
            session_id: Session identifier.
            segment_index: Index of this segment (for filename).

        Returns:
            Path to rendered MP4 file.

        Raises:
            EditError: If rendering fails.
        """
        session_dir = self.file_manager.get_session_dir(session_id)
        output_path = session_dir / f"clip_{segment_index + 1:02d}.mp4"

        # Step 1: Extract segment (cut)
        cut_path = session_dir / f"cut_{segment_index + 1:02d}.mp4"
        await self._cut_segment(video_path, segment, cut_path)

        # Step 2: Crop to 9:16 with smart center crop
        cropped_path = session_dir / f"cropped_{segment_index + 1:02d}.mp4"
        await self._crop_9x16(cut_path, cropped_path)

        # Step 3: Burn subtitle if available
        if subtitle_path and subtitle_path.exists():
            final_path = session_dir / f"final_{segment_index + 1:02d}.mp4"
            await self._burn_subtitle(cropped_path, subtitle_path, final_path)
            # Clean up intermediate
            cut_path.unlink(missing_ok=True)
            cropped_path.unlink(missing_ok=True)
            # Rename to output
            final_path.rename(output_path)
        else:
            cut_path.unlink(missing_ok=True)
            cropped_path.rename(output_path)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise EditError("❌ File output kosong setelah render.")

        return output_path

    async def _cut_segment(self, video_path: Path, segment: TimestampSegment, 
                           output_path: Path) -> None:
        """Cut video segment using FFmpeg."""
        duration = segment.duration

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(segment.start_time),
            "-t", str(duration),
            "-i", str(video_path),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]

        await self._run_ffmpeg(cmd, "cut segment")

    async def _crop_9x16(self, video_path: Path, output_path: Path) -> None:
        """Crop video to 9:16 portrait format using center crop.

        For 1920x1080 input: crop to 608x1080 (center)
        Maintains quality by scaling up after crop.
        """
        # Smart crop: center crop then scale to 1080x1920
        # For landscape 16:9, we crop width to match 9:16 ratio
        # height stays same, width = height * 9/16

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", (
                "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"  # Center crop to 9:16
                "scale=1080:1920:flags=lanczos,"      # Scale to 1080x1920
                "fps=30"                               # Ensure 30fps
            ),
            "-c:v", "libx264",
            "-preset", preset or self.config.ffmpeg_preset,
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ]

        await self._run_ffmpeg(cmd, "crop 9:16")

    async def _burn_subtitle(self, video_path: Path, 
                             subtitle_path: Path,
                             output_path: Path) -> None:
        """Burn ASS subtitle into video using FFmpeg."""
        # Escape subtitle path for FFmpeg filter
        sub_path_escaped = str(subtitle_path).replace(chr(92), "/")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"ass={sub_path_escaped}",
            "-c:v", "libx264",
            "-preset", preset or self.config.ffmpeg_preset,
            "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ]

        await self._run_ffmpeg(cmd, "burn subtitle")

    async def _run_ffmpeg(self, cmd: list, operation: str) -> None:
        """Run FFmpeg command with error handling.

        Args:
            cmd: FFmpeg command list.
            operation: Description of operation for error messages.
        """
        loop = asyncio.get_event_loop()

        def _execute():
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            return result

        try:
            result = await loop.run_in_executor(None, _execute)
            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                raise EditError(f"❌ FFmpeg error saat {operation}: {error_msg}")
        except asyncio.TimeoutError:
            raise EditError(f"❌ FFmpeg timeout saat {operation} (lebih dari 10 menit).")
        except Exception as e:
            if isinstance(e, EditError):
                raise
            raise EditError(f"❌ Error saat {operation}: {str(e)}")
