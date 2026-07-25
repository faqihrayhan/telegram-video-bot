"""Validation utilities for URL, duration, and format checks.

Used by all agents to ensure data integrity before processing.
"""
import re
import asyncio
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

import yt_dlp


YOUTUBE_PATTERNS = [
    r"^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+",
    r"^https?://(?:www\.)?youtube\.com/shorts/[\w-]+",
    r"^https?://youtu\.be/[\w-]+",
    r"^https?://(?:www\.)?youtube\.com/live/[\w-]+",
]

URL_REGEX = re.compile("|".join(YOUTUBE_PATTERNS), re.IGNORECASE)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_youtube_url(url: str) -> str:
    """Validate and normalize a YouTube URL.

    Args:
        url: Raw URL string from user input.

    Returns:
        Normalized URL string.

    Raises:
        ValidationError: If URL is not a valid YouTube link.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not URL_REGEX.match(url):
        raise ValidationError(
            "❌ Link tidak valid. Kirim link YouTube yang benar "
            "(youtube.com/watch?v=..., youtube.com/shorts/..., atau youtu.be/...)"
        )

    # Normalize youtu.be to standard format
    parsed = urlparse(url)
    if parsed.netloc == "youtu.be":
        video_id = parsed.path.strip("/")
        url = f"https://www.youtube.com/watch?v={video_id}"

    return url


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    parsed = urlparse(url)
    if parsed.netloc == "youtu.be":
        return parsed.path.strip("/")
    qs = parse_qs(parsed.query)
    return qs.get("v", [""])[0]


async def fetch_video_metadata(url: str) -> dict:
    """Fetch video metadata using yt-dlp (non-blocking via thread pool).

    Args:
        url: Validated YouTube URL.

    Returns:
        Dictionary with title, duration, uploader, thumbnail, etc.

    Raises:
        ValidationError: If video is unavailable or restricted.
    """
    loop = asyncio.get_event_loop()

    def _fetch():
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "id": info.get("id"),
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
                "thumbnail": info.get("thumbnail"),
                "view_count": info.get("view_count", 0),
                "upload_date": info.get("upload_date"),
                "formats_count": len(info.get("formats", [])),
            }

    try:
        return await loop.run_in_executor(None, _fetch)
    except yt_dlp.utils.DownloadError as e:
        raise ValidationError(f"❌ Gagal mengambil info video: {str(e)}")
    except Exception as e:
        raise ValidationError(f"❌ Error saat fetch metadata: {str(e)}")


def check_duration(duration: float, max_minutes: int) -> None:
    """Check if video duration is within allowed limit.

    Args:
        duration: Video duration in seconds.
        max_minutes: Maximum allowed duration in minutes.

    Raises:
        ValidationError: If duration exceeds limit.
    """
    max_seconds = max_minutes * 60
    if duration > max_seconds:
        raise ValidationError(
            f"❌ Durasi video {duration/60:.1f} menit melebihi batas maksimum "
            f"{max_minutes} menit."
        )
    if duration <= 0:
        raise ValidationError("❌ Durasi video tidak valid (0 detik).")


def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Sanitize video title for safe filesystem usage.

    Args:
        title: Raw video title.
        max_length: Maximum filename length.

    Returns:
        Safe filename string.
    """
    # Remove/replace unsafe characters
    safe = re.sub(r'[<>:"/\|?*]', "_", title)
    safe = re.sub(r'\s+', "_", safe)
    safe = safe.strip("._")
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe or "video"
