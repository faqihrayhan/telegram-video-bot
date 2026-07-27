"""AI Video Clipper Bot - Main Entry Point with Toggle Menu System

Telegram bot built with aiogram featuring inline keyboard menus for:
- Mode toggle (LOCAL / VPS)
- Transcriber on/off toggle  
- Skill manager / settings panel
- Status display with current configuration

All toggles are fully integrated into the processing pipeline.
"""
import os
import sys
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass, asdict

import aiosqlite
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, 
    InlineKeyboardButton, FSInputFile
)
from aiogram.filters import Command
from aiogram.enums import ParseMode

sys.path.insert(0, str(Path(__file__).parent))

from models import AppConfig
from utils import TempFileManager, ValidationError
from services import (
    MediaIngestionAgent,
    ContentAnalystAgent,
    TranscriberAgent,
    VideoEditorAgent,
    DownloadError,
    AnalysisError,
    TranscriptionError,
    EditError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class UserSettings:
    """Per-user bot settings stored in memory (or SQLite for persistence)."""
    user_id: int
    app_mode: str = "LOCAL"           # LOCAL or VPS
    transcriber_enabled: bool = True   # On / Off
    max_duration: int = 60             # minutes
    max_clips: int = 3                 # 1-5 clips
    whisper_model: str = "whisper-large-v3"  # or whisper-large-v3-turbo
    subtitle_style: str = "default"    # default, capcut, minimal
    auto_cleanup: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class SettingsManager:
    """Manages per-user settings with SQLite persistence."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._cache: Dict[int, UserSettings] = {}

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    app_mode TEXT DEFAULT 'LOCAL',
                    transcriber_enabled INTEGER DEFAULT 1,
                    max_duration INTEGER DEFAULT 60,
                    max_clips INTEGER DEFAULT 3,
                    whisper_model TEXT DEFAULT 'whisper-large-v3',
                    subtitle_style TEXT DEFAULT 'default',
                    auto_cleanup INTEGER DEFAULT 1,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    async def get(self, user_id: int) -> UserSettings:
        if user_id in self._cache:
            return self._cache[user_id]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()

            if row:
                settings = UserSettings(
                    user_id=row[0],
                    app_mode=row[1],
                    transcriber_enabled=bool(row[2]),
                    max_duration=row[3],
                    max_clips=row[4],
                    whisper_model=row[5],
                    subtitle_style=row[6],
                    auto_cleanup=bool(row[7])
                )
            else:
                settings = UserSettings(user_id=user_id)
                await self._save_to_db(settings)

            self._cache[user_id] = settings
            return settings

    async def update(self, user_id: int, **kwargs) -> UserSettings:
        settings = await self.get(user_id)
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        await self._save_to_db(settings)
        self._cache[user_id] = settings
        return settings

    async def _save_to_db(self, settings: UserSettings):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO user_settings 
                (user_id, app_mode, transcriber_enabled, max_duration, max_clips,
                 whisper_model, subtitle_style, auto_cleanup, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                settings.user_id, settings.app_mode,
                int(settings.transcriber_enabled), settings.max_duration,
                settings.max_clips, settings.whisper_model,
                settings.subtitle_style, int(settings.auto_cleanup),
                datetime.now().isoformat()
            ))
            await db.commit()


class FeedbackDatabase:
    """SQLite database for storing user feedback and metadata cache."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

    async def init(self):
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_url TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    segment_index INTEGER NOT NULL,
                    rating INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    reasoning TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metadata_cache (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    duration INTEGER,
                    uploader TEXT,
                    thumbnail TEXT,
                    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    video_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME
                )
            """)
            await db.commit()
        self._initialized = True
        logger.info("Database initialized")

    async def save_feedback(self, video_url: str, video_id: str, 
                            segment_index: int, rating: int,
                            user_id: int, reasoning: Optional[str] = None):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO feedback 
                   (video_url, video_id, segment_index, rating, user_id, timestamp, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_url, video_id, segment_index, rating, user_id,
                 datetime.now().isoformat(), reasoning)
            )
            await db.commit()

    async def get_feedback_stats(self, video_id: str) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT rating, COUNT(*) FROM feedback WHERE video_id = ? GROUP BY rating",
                (video_id,)
            )
            rows = await cursor.fetchall()
            stats = {"thumbs_up": 0, "thumbs_down": 0, "neutral": 0}
            for rating, count in rows:
                if rating == 1: stats["thumbs_up"] = count
                elif rating == -1: stats["thumbs_down"] = count
                else: stats["neutral"] = count
            return stats

    async def save_job(self, job_id: str, user_id: int, video_url: str, status: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO jobs (id, user_id, video_url, status) VALUES (?, ?, ?, ?)",
                (job_id, user_id, video_url, status)
            )
            await db.commit()

    async def update_job_status(self, job_id: str, status: str):
        async with aiosqlite.connect(self.db_path) as db:
            if status == "completed":
                await db.execute(
                    "UPDATE jobs SET status = ?, completed_at = ? WHERE id = ?",
                    (status, datetime.now().isoformat(), job_id)
                )
            else:
                await db.execute(
                    "UPDATE jobs SET status = ? WHERE id = ?",
                    (status, job_id)
                )
            await db.commit()


class ClipperBot:
    """Main bot orchestrator with toggle menu system."""

    def __init__(self):
        self.config = self._load_config()
        self.bot = Bot(token=self.config.telegram_bot_token)
        self.dp = Dispatcher()

        self.file_manager = TempFileManager(self.config)
        self.db = FeedbackDatabase(self.config.db_path)
        self.settings = SettingsManager(self.config.db_path)

        # Shared agents (stateless or configurable via method params)
        self.analyst = ContentAnalystAgent(self.config)
        self.transcriber = TranscriberAgent(self.config)

        self.job_semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        self._setup_handlers()

    def _load_config(self) -> AppConfig:
        return AppConfig(
            app_mode=os.getenv("APP_MODE", "LOCAL"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            max_video_duration_minutes=int(os.getenv("MAX_VIDEO_DURATION_MINUTES", "60")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "500")),
            max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "2")),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            groq_whisper_model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3"),
        )

    def _setup_handlers(self):
        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_help, Command("help"))
        self.dp.message.register(self.cmd_settings, Command("settings"))
        self.dp.message.register(self.handle_youtube_link, F.text)

        # Main menu callbacks
        self.dp.callback_query.register(self.cb_process_video, F.data == "menu:process")
        self.dp.callback_query.register(self.cb_show_settings, F.data == "menu:settings")
        self.dp.callback_query.register(self.cb_show_skills, F.data == "menu:skills")
        self.dp.callback_query.register(self.cb_help, F.data == "menu:help")

        # Settings toggles
        self.dp.callback_query.register(self.cb_toggle_mode, F.data == "toggle:mode")
        self.dp.callback_query.register(self.cb_toggle_transcriber, F.data == "toggle:transcriber")
        self.dp.callback_query.register(self.cb_toggle_cleanup, F.data == "toggle:cleanup")
        self.dp.callback_query.register(self.cb_set_clips, F.data.startswith("clips:"))
        self.dp.callback_query.register(self.cb_set_whisper, F.data.startswith("whisper:"))
        self.dp.callback_query.register(self.cb_back_main, F.data == "back:main")
        self.dp.callback_query.register(self.cb_back_settings, F.data == "back:settings")

        # Feedback
        self.dp.callback_query.register(self.handle_feedback, F.data.startswith("feedback:"))

    # ==================== KEYBOARD BUILDERS ====================

    def _main_menu_keyboard(self, settings: UserSettings) -> InlineKeyboardMarkup:
        mode_icon = "VPS" if settings.app_mode == "VPS" else "LOCAL"
        trans_icon = "ON" if settings.transcriber_enabled else "OFF"

        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="Process Video", callback_data="menu:process"),
            ],
            [
                InlineKeyboardButton(text=f"Mode: {mode_icon}", callback_data="toggle:mode"),
                InlineKeyboardButton(text=f"Transcriber: {trans_icon}", callback_data="toggle:transcriber"),
            ],
            [
                InlineKeyboardButton(text="Skill Manager", callback_data="menu:skills"),
                InlineKeyboardButton(text="Settings", callback_data="menu:settings"),
            ],
            [
                InlineKeyboardButton(text="Help", callback_data="menu:help"),
            ],
        ])

    def _settings_keyboard(self, settings: UserSettings) -> InlineKeyboardMarkup:
        cleanup_icon = "ON" if settings.auto_cleanup else "OFF"

        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="Toggle Mode (LOCAL/VPS)", callback_data="toggle:mode"),
            ],
            [
                InlineKeyboardButton(text="Toggle Transcriber", callback_data="toggle:transcriber"),
            ],
            [
                InlineKeyboardButton(text=f"Auto Cleanup: {cleanup_icon}", callback_data="toggle:cleanup"),
            ],
            [
                InlineKeyboardButton(text="Max Clips", callback_data="clips:show"),
            ],
            [
                InlineKeyboardButton(text="Whisper Model", callback_data="whisper:show"),
            ],
            [
                InlineKeyboardButton(text="Back to Main", callback_data="back:main"),
            ],
        ])

    def _skills_keyboard(self, settings: UserSettings) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="Agent 1: Ingestion", callback_data="skill:ingestion"),
            ],
            [
                InlineKeyboardButton(text="Agent 2: Analyst", callback_data="skill:analyst"),
            ],
            [
                InlineKeyboardButton(text="Agent 3: Transcriber", callback_data="skill:transcriber"),
            ],
            [
                InlineKeyboardButton(text="Agent 4: Editor", callback_data="skill:editor"),
            ],
            [
                InlineKeyboardButton(text="Back to Main", callback_data="back:main"),
            ],
        ])

    def _clips_keyboard(self, current: int) -> InlineKeyboardMarkup:
        buttons = []
        row = []
        for i in range(1, 6):
            label = f"[{i}]" if i == current else str(i)
            row.append(InlineKeyboardButton(text=label, callback_data=f"clips:{i}"))
        buttons.append(row)
        buttons.append([InlineKeyboardButton(text="Back", callback_data="back:settings")])
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    def _whisper_keyboard(self, current: str) -> InlineKeyboardMarkup:
        models = [
            ("Large v3", "whisper-large-v3"),
            ("Turbo", "whisper-large-v3-turbo"),
        ]
        buttons = []
        for label, model in models:
            mark = " [active]" if model == current else ""
            buttons.append([InlineKeyboardButton(text=f"{label}{mark}", callback_data=f"whisper:{model}")])
        buttons.append([InlineKeyboardButton(text="Back", callback_data="back:settings")])
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    def _status_text(self, settings: UserSettings) -> str:
        mode_emoji = "VPS" if settings.app_mode == "VPS" else "LOCAL"
        trans_emoji = "ON" if settings.transcriber_enabled else "OFF"
        cleanup_emoji = "ON" if settings.auto_cleanup else "OFF"

        return (
            f"<b>AI Video Clipper Bot</b>\n"
            f"<code>Status: Online</code>\n\n"
            f"<b>Current Configuration:</b>\n"
            f"  Mode: <code>{mode_emoji}</code>\n"
            f"  Transcriber: <code>{trans_emoji}</code>\n"
            f"  Max Clips: <code>{settings.max_clips}</code>\n"
            f"  Whisper: <code>{settings.whisper_model.split('-')[-1]}</code>\n"
            f"  Cleanup: <code>{cleanup_emoji}</code>\n\n"
            f"Send a YouTube link to start processing."
        )

    # ==================== COMMAND HANDLERS ====================

    async def cmd_start(self, message: Message):
        user_id = message.from_user.id
        settings = await self.settings.get(user_id)

        text = self._status_text(settings)
        keyboard = self._main_menu_keyboard(settings)

        await message.answer(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    async def cmd_help(self, message: Message):
        help_text = (
            "<b>AI Video Clipper Bot - Help</b>\n\n"
            "<b>How to use:</b>\n"
            "1. Send a YouTube link\n"
            "2. Bot analyzes and finds viral segments\n"
            "3. Receive clips with burned-in subtitles\n\n"
            "<b>Menu Options:</b>\n"
            "  Process Video - Start processing a link\n"
            "  Mode - Toggle LOCAL (quality) / VPS (speed)\n"
            "  Transcriber - Turn subtitle generation on/off\n"
            "  Skill Manager - View agent pipeline status\n"
            "  Settings - Configure max clips, whisper model, cleanup\n\n"
            "<b>Tips:</b>\n"
            "  VPS mode = faster render, auto cleanup\n"
            "  LOCAL mode = better quality, manual cleanup\n"
            "  Whisper Turbo = 2.8x cheaper, similar quality"
        )
        await message.answer(help_text, parse_mode=ParseMode.HTML)

    async def cmd_settings(self, message: Message):
        user_id = message.from_user.id
        settings = await self.settings.get(user_id)
        keyboard = self._settings_keyboard(settings)

        text = (
            f"<b>Settings Panel</b>\n\n"
            f"Mode: <code>{settings.app_mode}</code>\n"
            f"Transcriber: <code>{'ON' if settings.transcriber_enabled else 'OFF'}</code>\n"
            f"Max Clips: <code>{settings.max_clips}</code>\n"
            f"Whisper Model: <code>{settings.whisper_model}</code>\n"
            f"Auto Cleanup: <code>{'ON' if settings.auto_cleanup else 'OFF'}</code>"
        )
        await message.answer(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    # ==================== CALLBACK HANDLERS ====================

    async def cb_process_video(self, callback: CallbackQuery):
        await callback.message.edit_text(
            "<b>Process Video</b>\n\n"
            "Send a YouTube link to start processing.\n"
            "Supported formats:\n"
            "  youtube.com/watch?v=...\n"
            "  youtube.com/shorts/...\n"
            "  youtu.be/...",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Back", callback_data="back:main")]
            ])
        )
        await callback.answer()

    async def cb_show_settings(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        keyboard = self._settings_keyboard(settings)

        text = (
            f"<b>Settings Panel</b>\n\n"
            f"Mode: <code>{settings.app_mode}</code>\n"
            f"Transcriber: <code>{'ON' if settings.transcriber_enabled else 'OFF'}</code>\n"
            f"Max Clips: <code>{settings.max_clips}</code>\n"
            f"Whisper Model: <code>{settings.whisper_model}</code>\n"
            f"Auto Cleanup: <code>{'ON' if settings.auto_cleanup else 'OFF'}</code>"
        )
        await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await callback.answer()

    async def cb_show_skills(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        keyboard = self._skills_keyboard(settings)

        text = (
            "<b>Skill Manager</b>\n\n"
            "<b>Agent Pipeline Status:</b>\n"
            "  1. Media Ingestion  - Ready\n"
            "  2. Content Analyst  - Ready\n"
            "  3. Transcriber    - " + ("Active" if settings.transcriber_enabled else "Disabled") + "\n"
            "  4. Video Editor   - Ready\n\n"
            "Click an agent to view details."
        )
        await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await callback.answer()

    async def cb_help(self, callback: CallbackQuery):
        text = (
            "<b>Help & Documentation</b>\n\n"
            "<b>Quick Start:</b>\n"
            "1. Toggle settings via the menu below\n"
            "2. Send any YouTube link\n"
            "3. Wait for processing (status updates at each step)\n"
            "4. Receive clips and rate them\n\n"
            "<b>Mode Differences:</b>\n"
            "  LOCAL: Better quality, manual cleanup, D: drive temp\n"
            "  VPS:   Ultrafast render, auto cleanup, /tmp/ temp\n\n"
            "<b>Transcriber Toggle:</b>\n"
            "  ON:  Generates word-level subtitles (costs API credits)\n"
            "  OFF: Skips transcription, clips without subtitles"
        )
        await callback.message.edit_text(
            text, parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Back", callback_data="back:main")]
            ])
        )
        await callback.answer()

    # ==================== TOGGLE HANDLERS ====================

    async def cb_toggle_mode(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        new_mode = "VPS" if settings.app_mode == "LOCAL" else "LOCAL"
        await self.settings.update(user_id, app_mode=new_mode)

        updated = await self.settings.get(user_id)
        keyboard = self._main_menu_keyboard(updated)
        text = self._status_text(updated)

        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception:
            pass
        await callback.answer(f"Mode switched to {new_mode}")

    async def cb_toggle_transcriber(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        new_state = not settings.transcriber_enabled
        await self.settings.update(user_id, transcriber_enabled=new_state)

        updated = await self.settings.get(user_id)
        keyboard = self._main_menu_keyboard(updated)
        text = self._status_text(updated)

        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception:
            pass
        status = "ON" if new_state else "OFF"
        await callback.answer(f"Transcriber {status}")

    async def cb_toggle_cleanup(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        new_state = not settings.auto_cleanup
        await self.settings.update(user_id, auto_cleanup=new_state)

        updated = await self.settings.get(user_id)
        keyboard = self._settings_keyboard(updated)

        text = (
            f"<b>Settings Panel</b>\n\n"
            f"Mode: <code>{updated.app_mode}</code>\n"
            f"Transcriber: <code>{'ON' if updated.transcriber_enabled else 'OFF'}</code>\n"
            f"Max Clips: <code>{updated.max_clips}</code>\n"
            f"Whisper Model: <code>{updated.whisper_model}</code>\n"
            f"Auto Cleanup: <code>{'ON' if updated.auto_cleanup else 'OFF'}</code>"
        )
        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception:
            pass
        status = "ON" if new_state else "OFF"
        await callback.answer(f"Auto Cleanup {status}")

    async def cb_set_clips(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        data = callback.data

        if data == "clips:show":
            settings = await self.settings.get(user_id)
            keyboard = self._clips_keyboard(settings.max_clips)
            await callback.message.edit_text(
                "<b>Select Max Clips</b>\n\n"
                "How many clips to generate per video?",
                parse_mode=ParseMode.HTML, reply_markup=keyboard
            )
            await callback.answer()
            return

        clips = int(data.split(":")[1])
        await self.settings.update(user_id, max_clips=clips)

        updated = await self.settings.get(user_id)
        keyboard = self._settings_keyboard(updated)
        text = (
            f"<b>Settings Panel</b>\n\n"
            f"Mode: <code>{updated.app_mode}</code>\n"
            f"Transcriber: <code>{'ON' if updated.transcriber_enabled else 'OFF'}</code>\n"
            f"Max Clips: <code>{updated.max_clips}</code>\n"
            f"Whisper Model: <code>{updated.whisper_model}</code>\n"
            f"Auto Cleanup: <code>{'ON' if updated.auto_cleanup else 'OFF'}</code>"
        )
        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception:
            pass
        await callback.answer(f"Max clips set to {clips}")

    async def cb_set_whisper(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        data = callback.data

        if data == "whisper:show":
            settings = await self.settings.get(user_id)
            keyboard = self._whisper_keyboard(settings.whisper_model)
            await callback.message.edit_text(
                "<b>Select Whisper Model</b>\n\n"
                "Large v3: Best accuracy\n"
                "Turbo: 2.8x cheaper, slightly faster",
                parse_mode=ParseMode.HTML, reply_markup=keyboard
            )
            await callback.answer()
            return

        model = data.split(":", 1)[1]
        await self.settings.update(user_id, whisper_model=model)

        updated = await self.settings.get(user_id)
        keyboard = self._settings_keyboard(updated)
        text = (
            f"<b>Settings Panel</b>\n\n"
            f"Mode: <code>{updated.app_mode}</code>\n"
            f"Transcriber: <code>{'ON' if updated.transcriber_enabled else 'OFF'}</code>\n"
            f"Max Clips: <code>{updated.max_clips}</code>\n"
            f"Whisper Model: <code>{updated.whisper_model}</code>\n"
            f"Auto Cleanup: <code>{'ON' if updated.auto_cleanup else 'OFF'}</code>"
        )
        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except Exception:
            pass
        await callback.answer(f"Whisper model: {model}")

    async def cb_back_main(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        keyboard = self._main_menu_keyboard(settings)
        text = self._status_text(settings)

        await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await callback.answer()

    async def cb_back_settings(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        settings = await self.settings.get(user_id)
        keyboard = self._settings_keyboard(settings)

        text = (
            f"<b>Settings Panel</b>\n\n"
            f"Mode: <code>{settings.app_mode}</code>\n"
            f"Transcriber: <code>{'ON' if settings.transcriber_enabled else 'OFF'}</code>\n"
            f"Max Clips: <code>{settings.max_clips}</code>\n"
            f"Whisper Model: <code>{settings.whisper_model}</code>\n"
            f"Auto Cleanup: <code>{'ON' if settings.auto_cleanup else 'OFF'}</code>"
        )
        await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await callback.answer()

    # ==================== VIDEO PROCESSING (FULLY INTEGRATED) ====================

    async def handle_youtube_link(self, message: Message):
        """Handle incoming YouTube URL."""
        url = message.text.strip()
        user_id = message.from_user.id

        if not any(domain in url.lower() for domain in ["youtube.com", "youtu.be"]):
            await message.answer("Send a valid YouTube link.")
            return

        settings = await self.settings.get(user_id)
        session_id = str(uuid.uuid4())[:8]
        job_id = str(uuid.uuid4())

        asyncio.create_task(
            self._process_job(job_id, session_id, user_id, url, message, settings)
        )

    async def _process_job(self, job_id: str, session_id: str, 
                           user_id: int, url: str, message: Message, 
                           settings: UserSettings):
        """Process video with FULL user settings integration into the pipeline."""
        async with self.job_semaphore:

            # ===== BUILD USER-SPECIFIC CONFIG =====
            user_config = self.config.model_copy(update={
                "app_mode": settings.app_mode,
                "groq_whisper_model": settings.whisper_model,
            })

            # Log user settings for debugging
            logger.info(
                f"Job {job_id} | User {user_id} | "
                f"Mode={settings.app_mode} | "
                f"Transcriber={settings.transcriber_enabled} | "
                f"MaxClips={settings.max_clips} | "
                f"Whisper={settings.whisper_model} | "
                f"Cleanup={settings.auto_cleanup}"
            )

            # Build user-specific services
            user_file_manager = TempFileManager(user_config)
            user_ingestion = MediaIngestionAgent(user_config, user_file_manager)
            user_editor = VideoEditorAgent(user_config, user_file_manager)

            await self.db.save_job(job_id, user_id, url, "processing")

            status_msg = await message.answer(
                f"<b>Starting process...</b>\n"
                f"<code>Mode: {settings.app_mode} | "
                f"Transcriber: {'ON' if settings.transcriber_enabled else 'OFF'} | "
                f"Clips: {settings.max_clips}</code>",
                parse_mode=ParseMode.HTML
            )

            try:
                # ===== AGENT 1: INGESTION (with user config) =====
                await self._update_status(status_msg, 
                    f"Agent 1/4: Downloading video...\n"
                    f"<code>Mode: {settings.app_mode}</code>")

                video_path, audio_path, metadata = await user_ingestion.process(url, session_id)

                title = metadata.get("title", "Unknown")
                duration = metadata.get("duration", 0)

                await self._update_status(
                    status_msg,
                    f"Download complete!\n"
                    f"<b>{title[:50]}{'...' if len(title) > 50 else ''}</b>\n"
                    f"Duration: {duration/60:.1f} min\n"
                    f"Temp: <code>{user_config.temp_base_path}</code>\n\n"
                    f"Agent 2/4: Analyzing content..."
                )

                # ===== AGENT 2: ANALYSIS =====
                analysis = await self.analyst.analyze(video_path, audio_path, metadata)

                if not analysis.segments:
                    await self._update_status(status_msg, "No viral segments found.")
                    return

                # Apply user max clips setting
                max_clips = min(settings.max_clips, len(analysis.segments))
                segments = analysis.segments[:max_clips]

                segments_text = "\n".join([
                    f"  {i+1}. {seg.start_time:.1f}s - {seg.end_time:.1f}s "
                    for i, seg in enumerate(segments)
                ])

                await self._update_status(
                    status_msg,
                    f"Analysis complete!\n"
                    f"Found {len(analysis.segments)} segments, "
                    f"using top {max_clips} (user limit)\n"
                    f"<code>{segments_text}</code>\n\n"
                    f"Agent 3/4: Processing...",
                    parse_mode=ParseMode.HTML
                )

                # ===== AGENT 3 & 4: TRANSCRIBE + EDIT =====
                rendered_clips = []

                for idx, segment in enumerate(segments):
                    subtitle_path = None

                    if settings.transcriber_enabled:
                        await self._update_status(
                            status_msg,
                            f"Agent 3/4: Transcribing segment {idx+1}/{len(segments)}...\n"
                            f"<code>Model: {settings.whisper_model}</code>"
                        )

                        transcription = await self.transcriber.transcribe(
                            audio_path,
                            model=settings.whisper_model  # <-- USER WHISPER MODEL
                        )

                        segment_words = [
                            w for w in transcription.words
                            if segment.start_time <= w.start <= segment.end_time
                        ]
                        for word in segment_words:
                            word.start -= segment.start_time
                            word.end -= segment.start_time

                        from models import TranscriptionResult
                        segment_transcription = TranscriptionResult(
                            words=segment_words,
                            full_text=" ".join(w.word for w in segment_words),
                            language=transcription.language,
                            duration=segment.duration
                        )

                        subtitle_path = user_file_manager.get_temp_path(
                            session_id, f"subtitle_{idx+1:02d}.ass"
                        )
                        await self.transcriber.generate_subtitle(
                            segment_transcription, subtitle_path
                        )
                    else:
                        await self._update_status(
                            status_msg,
                            "Agent 3/4: Transcriber OFF, skipping subtitles..."
                        )

                    await self._update_status(
                        status_msg,
                        f"Agent 4/4: Rendering clip {idx+1}/{len(segments)}...\n"
                        f"<code>Preset: {user_config.ffmpeg_preset}</code>"
                    )

                    clip_path = await user_editor.render_clip(
                        video_path=video_path,
                        audio_path=audio_path,
                        segment=segment,
                        subtitle_path=subtitle_path,
                        session_id=session_id,
                        segment_index=idx,
                        preset=user_config.ffmpeg_preset  # <-- USER PRESET
                    )

                    rendered_clips.append((clip_path, segment, idx))

                # ===== DELIVERY =====
                await self._update_status(status_msg, "Sending results...")

                video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]

                for clip_path, segment, idx in rendered_clips:
                    keyboard = InlineKeyboardMarkup(inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Good",
                                callback_data=f"feedback:{video_id}:{idx}:1:{job_id}"
                            ),
                            InlineKeyboardButton(
                                text="Bad",
                                callback_data=f"feedback:{video_id}:{idx}:-1:{job_id}"
                            )
                        ]
                    ])

                    trans_status = "Subtitles ON" if settings.transcriber_enabled else "No subtitles"
                    caption = (
                        f"<b>Clip {idx+1}</b>\n"
                        f"{segment.start_time:.1f}s - {segment.end_time:.1f}s\n"
                        f"{trans_status} | {segment.hook_type or 'viral'}\n\n"
                        f"<i>{segment.reasoning}</i>\n\n"
                        f"Rate this clip?"
                    )

                    await message.answer_video(
                        video=FSInputFile(clip_path),
                        caption=caption,
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard
                    )

                await self._update_status(
                    status_msg,
                    f"<b>Done!</b> {len(rendered_clips)} clip(s) generated.\n"
                    f"<code>Mode: {settings.app_mode} | "
                    f"Preset: {user_config.ffmpeg_preset}</code>"
                )
                await self.db.update_job_status(job_id, "completed")

            except ValidationError as e:
                await self._update_status(status_msg, f"{str(e)}")
                await self.db.update_job_status(job_id, "failed_validation")
            except DownloadError as e:
                await self._update_status(status_msg, f"{str(e)}")
                await self.db.update_job_status(job_id, "failed_download")
            except AnalysisError as e:
                await self._update_status(status_msg, f"{str(e)}")
                await self.db.update_job_status(job_id, "failed_analysis")
            except TranscriptionError as e:
                await self._update_status(status_msg, f"{str(e)}")
                await self.db.update_job_status(job_id, "failed_transcription")
            except EditError as e:
                await self._update_status(status_msg, f"{str(e)}")
                await self.db.update_job_status(job_id, "failed_edit")
            except Exception as e:
                logger.exception("Unexpected error in pipeline")
                await self._update_status(
                    status_msg,
                    f"Unexpected error: {str(e)[:200]}\nPlease try again."
                )
                await self.db.update_job_status(job_id, "failed_unknown")
            finally:
                # Cleanup using user setting
                if settings.auto_cleanup:
                    logger.info(f"Job {job_id} | Auto cleanup enabled, cleaning session {session_id}")
                    await user_file_manager.cleanup_session(session_id)
                else:
                    logger.info(f"Job {job_id} | Auto cleanup disabled, keeping files at {user_config.temp_base_path}")

    async def _update_status(self, message, text: str, parse_mode=None):
        try:
            await message.edit_text(text, parse_mode=parse_mode or ParseMode.HTML)
        except Exception:
            pass

    async def handle_feedback(self, callback: CallbackQuery):
        try:
            parts = callback.data.split(":")
            if len(parts) != 5:
                await callback.answer("Invalid feedback data.")
                return

            _, video_id, segment_index, rating, job_id = parts
            segment_index = int(segment_index)
            rating = int(rating)
            user_id = callback.from_user.id

            await self.db.save_feedback(
                video_url=f"https://youtube.com/watch?v={video_id}",
                video_id=video_id,
                segment_index=segment_index,
                rating=rating,
                user_id=user_id
            )

            emoji = "Good" if rating == 1 else "Bad"
            await callback.answer(f"Thanks! {emoji} feedback saved.")
            await callback.message.edit_reply_markup(reply_markup=None)

        except Exception as e:
            logger.error(f"Feedback error: {e}")
            await callback.answer("Failed to save feedback.")

    async def run(self):
        await self.db.init()
        await self.settings.init()
        logger.info(f"Bot started in {self.config.app_mode} mode")
        logger.info(f"Temp directory: {self.config.temp_base_path}")
        logger.info(f"Max concurrent jobs: {self.config.max_concurrent_jobs}")

        await self.dp.start_polling(self.bot)


async def main():
    bot = ClipperBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
