"""File lifecycle management and temp directory handling.

Handles dual-mode path resolution, cleanup, and storage management.
"""
import os
import shutil
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

from models import AppConfig


class TempFileManager:
    """Manages temporary files with auto-cleanup support."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.temp_base = config.temp_base_path
        self.auto_cleanup = config.auto_cleanup
        self._ensure_temp_dir()

    def _ensure_temp_dir(self) -> None:
        """Create temp directory if it doesn't exist."""
        self.temp_base.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, session_id: str) -> Path:
        """Get or create a session-specific subdirectory.

        Args:
            session_id: Unique identifier for this processing session.

        Returns:
            Path to session directory.
        """
        session_path = self.temp_base / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path

    def get_temp_path(self, session_id: str, filename: str) -> Path:
        """Get a temp file path within a session.

        Args:
            session_id: Session identifier.
            filename: Desired filename.

        Returns:
            Full path to temp file.
        """
        return self.get_session_dir(session_id) / filename

    def list_session_files(self, session_id: str) -> List[Path]:
        """List all files in a session directory.

        Args:
            session_id: Session identifier.

        Returns:
            List of file paths.
        """
        session_dir = self.temp_base / session_id
        if not session_dir.exists():
            return []
        return [f for f in session_dir.iterdir() if f.is_file()]

    async def cleanup_session(self, session_id: str) -> None:
        """Remove all files for a session.

        Args:
            session_id: Session identifier to clean up.
        """
        session_dir = self.temp_base / session_id
        if not session_dir.exists():
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.rmtree, session_dir, True)

    async def cleanup_all(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up old temp files across all sessions.

        Args:
            max_age_hours: Delete files older than this. If None, uses auto_cleanup setting.

        Returns:
            Number of sessions cleaned up.
        """
        if not self.temp_base.exists():
            return 0

        if max_age_hours is None:
            max_age_hours = 1 if self.auto_cleanup else 24

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        count = 0

        for item in self.temp_base.iterdir():
            if not item.is_dir():
                continue
            try:
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, shutil.rmtree, item, True)
                    count += 1
            except (OSError, PermissionError):
                continue

        return count

    def get_file_size_mb(self, filepath: Path) -> float:
        """Get file size in megabytes.

        Args:
            filepath: Path to file.

        Returns:
            Size in MB.
        """
        if not filepath.exists():
            return 0.0
        return filepath.stat().st_size / (1024 * 1024)

    def check_disk_space(self, required_mb: int = 1000) -> bool:
        """Check if sufficient disk space is available.

        Args:
            required_mb: Required space in MB.

        Returns:
            True if enough space available.
        """
        stat = shutil.disk_usage(self.temp_base)
        available_mb = stat.free / (1024 * 1024)
        return available_mb >= required_mb
