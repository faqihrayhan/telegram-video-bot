"""Agent 1: Media Ingestion Agent

Handles YouTube link validation, metadata extraction, and media download.
Includes metadata caching for repeated URLs.
"""
import asyncio
from pathlib import Path
from typing import Optional, Tuple

import yt_dlp

from models import AppConfig
from utils import (
    validate_youtube_url,
    extract_video_id,
    fetch_video_metadata,
    check_duration,
    sanitize_filename,
)
from utils.file_manager import TempFileManager


class DownloadError(Exception):
    """Raised when download fails."""
    pass


class MediaIngestionAgent:
    """Agent responsible for downloading and validating media from YouTube."""

    def __init__(self, config: AppConfig, file_manager: TempFileManager):
        self.config = config
        self.file_manager = file_manager
        self._metadata_cache: dict = {}  # In-memory cache

    async def process(self, url: str, session_id: str) -> Tuple[Path, Path, dict]:
        """Full ingestion pipeline: validate → fetch metadata → download.

        Args:
            url: Raw YouTube URL from user.
            session_id: Unique session identifier.

        Returns:
            Tuple of (video_path, audio_path, metadata_dict).

        Raises:
            ValidationError: If URL invalid or video restricted.
            DownloadError: If download fails.
        """
        # Step 1: Validate URL
        validated_url = validate_youtube_url(url)
        video_id = extract_video_id(validated_url)

        # Step 2: Fetch metadata (with caching)
        metadata = await self._get_metadata(validated_url, video_id)

        # Step 3: Check duration limit
        check_duration(metadata["duration"], self.config.max_video_duration_minutes)

        # Step 4: Download video + audio
        session_dir = self.file_manager.get_session_dir(session_id)
        video_path, audio_path = await self._download_media(
            validated_url, session_dir, metadata
        )

        # Step 5: Verify files
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise DownloadError("❌ File video gagal diunduh atau kosong.")

        return video_path, audio_path, metadata

    async def _get_metadata(self, url: str, video_id: str) -> dict:
        """Get metadata with in-memory caching."""
        if video_id in self._metadata_cache:
            return self._metadata_cache[video_id]

        metadata = await fetch_video_metadata(url)
        self._metadata_cache[video_id] = metadata
        return metadata

    async def _download_media(self, url: str, session_dir: Path, 
                              metadata: dict) -> Tuple[Path, Path]:
        """Download video and extract audio using yt-dlp.

        Args:
            url: Validated YouTube URL.
            session_dir: Directory to save files.
            metadata: Video metadata.

        Returns:
            Tuple of (video_file_path, audio_file_path).
        """
        safe_title = sanitize_filename(metadata["title"])
        video_path = session_dir / f"{safe_title}.mp4"
        audio_path = session_dir / f"{safe_title}.mp3"

        loop = asyncio.get_event_loop()

        def _download():
            # Download best video+audio merged
            ydl_opts_video = {
                "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": str(video_path.with_suffix("").with_suffix(".%(ext)s")),
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                ydl.download([url])

            # Download audio-only for transcription
            ydl_opts_audio = {
                "format": "bestaudio/best",
                "outtmpl": str(audio_path.with_suffix("").with_suffix(".%(ext)s")),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([url])

        try:
            await loop.run_in_executor(None, _download)
        except Exception as e:
            raise DownloadError(f"❌ Gagal mengunduh media: {str(e)}")

        # Find actual downloaded files (yt-dlp may change extension)
        downloaded_video = self._find_file(session_dir, safe_title, [".mp4", ".webm", ".mkv"])
        downloaded_audio = self._find_file(session_dir, safe_title, [".mp3", ".m4a", ".wav"])

        if not downloaded_video:
            raise DownloadError("❌ File video tidak ditemukan setelah download.")
        if not downloaded_audio:
            raise DownloadError("❌ File audio tidak ditemukan setelah download.")

        return downloaded_video, downloaded_audio

    def _find_file(self, directory: Path, prefix: str, extensions: list) -> Optional[Path]:
        """Find file by prefix and possible extensions."""
        for ext in extensions:
            candidate = directory / f"{prefix}{ext}"
            if candidate.exists():
                return candidate
        # Fallback: search directory
        for f in directory.iterdir():
            if f.is_file() and f.stem.startswith(prefix):
                return f
        return None
