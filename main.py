"""AI Video Clipper Bot - Main Entry Point

Telegram bot built with aiogram that orchestrates 4 specialized agents
to convert YouTube videos into viral-ready 9:16 clips with subtitles.

Features:
- Async job queue with semaphore (VPS-safe concurrent processing)
- User feedback system (SQLite) for continuous improvement
- Status updates at each pipeline stage
- Comprehensive error handling and recovery
"""
import os
import sys
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import aiosqlite
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.enums import ParseMode

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import AppConfig, AppMode, AnalysisResult, TimestampSegment
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FeedbackDatabase:
    """SQLite database for storing user feedback and metadata cache."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

    async def init(self):
        """Initialize database tables."""
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
                    rating INTEGER NOT NULL,  -- -1, 0, 1
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
        """Save user feedback."""
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
        """Get feedback statistics for a video."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT rating, COUNT(*) FROM feedback WHERE video_id = ? GROUP BY rating",
                (video_id,)
            )
            rows = await cursor.fetchall()
            stats = {"thumbs_up": 0, "thumbs_down": 0, "neutral": 0}
            for rating, count in rows:
                if rating == 1:
                    stats["thumbs_up"] = count
                elif rating == -1:
                    stats["thumbs_down"] = count
                else:
                    stats["neutral"] = count
            return stats

    async def save_job(self, job_id: str, user_id: int, video_url: str, status: str):
        """Save job record."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO jobs (id, user_id, video_url, status) VALUES (?, ?, ?, ?)",
                (job_id, user_id, video_url, status)
            )
            await db.commit()

    async def update_job_status(self, job_id: str, status: str):
        """Update job status."""
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
    """Main bot orchestrator with job queue and agent pipeline."""

    def __init__(self):
        self.config = self._load_config()
        self.bot = Bot(token=self.config.telegram_bot_token)
        self.dp = Dispatcher()

        # Initialize components
        self.file_manager = TempFileManager(self.config)
        self.db = FeedbackDatabase(self.config.db_path)

        # Initialize agents
        self.ingestion = MediaIngestionAgent(self.config, self.file_manager)
        self.analyst = ContentAnalystAgent(self.config)
        self.transcriber = TranscriberAgent(self.config)
        self.editor = VideoEditorAgent(self.config, self.file_manager)

        # Job queue with semaphore for concurrent control
        self.job_semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        self._setup_handlers()

    def _load_config(self) -> AppConfig:
        """Load and validate configuration from environment."""
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
        """Register Telegram bot handlers."""
        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_help, Command("help"))
        self.dp.message.register(self.handle_youtube_link, F.text)
        self.dp.callback_query.register(self.handle_feedback, F.data.startswith("feedback:"))

    async def cmd_start(self, message: Message):
        """Handle /start command."""
        welcome_text = (
            "🎬 <b>AI Video Clipper Bot</b>

"
            "Kirim link YouTube, dan bot akan:
"
            "1️⃣ Download video
"
            "2️⃣ Analisis bagian paling viral
"
            "3️⃣ Buat subtitle kata-per-kata
"
            "4️⃣ Render clip 9:16 siap upload

"
            "📎 <b>Cara pakai:</b> Kirim link YouTube (contoh: youtube.com/watch?v=...)
"
            "⏱️ <b>Limit:</b> Maksimal 60 menit per video

"
            "Ketik /help untuk info lebih lanjut."
        )
        await message.answer(welcome_text, parse_mode=ParseMode.HTML)

    async def cmd_help(self, message: Message):
        """Handle /help command."""
        help_text = (
            "📖 <b>Panduan Penggunaan</b>

"
            "<b>1. Kirim Link YouTube</b>
"
            "   • youtube.com/watch?v=...
"
            "   • youtube.com/shorts/...
"
            "   • youtu.be/...

"
            "<b>2. Tunggu Proses</b>
"
            "   Bot akan mengirim status tiap tahap.

"
            "<b>3. Terima Hasil</b>
"
            "   Video clip 9:16 dengan subtitle burn-in.

"
            "<b>4. Beri Feedback</b>
"
            "   👍 kalau bagus, 👎 kalau kurang.

"
            "<b>Tips:</b>
"
            "• Video dengan dialog/jelas lebih bagus hasilnya
"
            "• Podcast, interview, dan educational content = ⭐⭐⭐"
        )
        await message.answer(help_text, parse_mode=ParseMode.HTML)

    async def handle_youtube_link(self, message: Message):
        """Handle incoming YouTube URL from user."""
        url = message.text.strip()
        user_id = message.from_user.id

        # Validate URL quickly
        if not any(domain in url.lower() for domain in ["youtube.com", "youtu.be"]):
            await message.answer(
                "❌ Kirim link YouTube yang valid.
"
                "Contoh: https://youtube.com/watch?v=..."
            )
            return

        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        job_id = str(uuid.uuid4())

        # Send to job queue
        asyncio.create_task(
            self._process_job(job_id, session_id, user_id, url, message)
        )

    async def _process_job(self, job_id: str, session_id: str, 
                           user_id: int, url: str, message: Message):
        """Process video through the 4-agent pipeline with semaphore."""
        async with self.job_semaphore:
            await self.db.save_job(job_id, user_id, url, "processing")

            status_msg = await message.answer("⏳ <b>Memulai proses...</b>", parse_mode=ParseMode.HTML)

            try:
                # ===== AGENT 1: INGESTION =====
                await self._update_status(status_msg, "📥 Agent 1/4: Mendownload video...")
                video_path, audio_path, metadata = await self.ingestion.process(url, session_id)

                title = metadata.get("title", "Unknown")
                duration = metadata.get("duration", 0)

                await self._update_status(
                    status_msg,
                    f"✅ Download selesai!
"
                    f"📹 <b>{title[:50]}{'...' if len(title) > 50 else ''}</b>
"
                    f"⏱️ Durasi: {duration/60:.1f} menit

"
                    f"🧠 Agent 2/4: Menganalisis konten viral..."
                )

                # ===== AGENT 2: ANALYSIS =====
                analysis = await self.analyst.analyze(video_path, audio_path, metadata)

                if not analysis.segments:
                    await self._update_status(status_msg, "❌ Tidak menemukan segment yang cocok.")
                    return

                segments_text = "
".join([
                    f"  {i+1}. {seg.start_time:.1f}s - {seg.end_time:.1f}s "
                    f"({seg.hook_type or 'viral'})"
                    for i, seg in enumerate(analysis.segments[:3])
                ])

                await self._update_status(
                    status_msg,
                    f"✅ Analisis selesai!
"
                    f"🎯 Menemukan {len(analysis.segments)} segment viral:
"
                    f"<code>{segments_text}</code>

"
                    f"📝 Agent 3/4: Mentranskrip audio...",
                    parse_mode=ParseMode.HTML
                )

                # Process each segment
                rendered_clips = []

                for idx, segment in enumerate(analysis.segments[:3]):
                    # ===== AGENT 3: TRANSCRIPTION =====
                    await self._update_status(
                        status_msg,
                        f"📝 Agent 3/4: Transkripsi segment {idx+1}/{len(analysis.segments[:3])}..."
                    )

                    # For transcription, we use the full audio but the timing will be offset
                    # In production, you'd extract just the segment audio
                    transcription = await self.transcriber.transcribe(audio_path)

                    # Filter words within segment timeframe
                    segment_words = [
                        w for w in transcription.words
                        if segment.start_time <= w.start <= segment.end_time
                    ]

                    # Adjust word timings relative to segment start
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

                    # Generate subtitle
                    subtitle_path = self.file_manager.get_temp_path(
                        session_id, f"subtitle_{idx+1:02d}.ass"
                    )
                    await self.transcriber.generate_subtitle(
                        segment_transcription, subtitle_path
                    )

                    # ===== AGENT 4: VIDEO EDITING =====
                    await self._update_status(
                        status_msg,
                        f"✂️ Agent 4/4: Render clip {idx+1}/{len(analysis.segments[:3])}..."
                    )

                    clip_path = await self.editor.render_clip(
                        video_path=video_path,
                        audio_path=audio_path,
                        segment=segment,
                        subtitle_path=subtitle_path,
                        session_id=session_id,
                        segment_index=idx
                    )

                    rendered_clips.append((clip_path, segment, idx))

                # ===== DELIVERY =====
                await self._update_status(status_msg, "📤 Mengirim hasil...")

                video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]

                for clip_path, segment, idx in rendered_clips:
                    # Build feedback keyboard
                    keyboard = InlineKeyboardMarkup(inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="👍 Bagus",
                                callback_data=f"feedback:{video_id}:{idx}:1:{job_id}"
                            ),
                            InlineKeyboardButton(
                                text="👎 Kurang",
                                callback_data=f"feedback:{video_id}:{idx}:-1:{job_id}"
                            )
                        ]
                    ])

                    caption = (
                        f"🎬 <b>Clip {idx+1}</b>
"
                        f"⏱️ {segment.start_time:.1f}s - {segment.end_time:.1f}s
"
                        f"🎯 {segment.hook_type or 'viral'} | "
                        f"confidence: {segment.confidence:.0%}

"
                        f"<i>{segment.reasoning}</i>

"
                        f"Bagus nggak hasilnya?"
                    )

                    await message.answer_video(
                        video=FSInputFile(clip_path),
                        caption=caption,
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard
                    )

                await self._update_status(
                    status_msg,
                    f"✅ <b>Selesai!</b> {len(rendered_clips)} clip berhasil dibuat.
"
                    f"Kasih rating ya biar AI makin pintar 🙏"
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
                    f"❌ Error tidak terduga: {str(e)[:200]}
"
                    f"Coba lagi atau hubungi admin."
                )
                await self.db.update_job_status(job_id, "failed_unknown")

            finally:
                # Cleanup temp files if VPS mode
                if self.config.auto_cleanup:
                    await self.file_manager.cleanup_session(session_id)

    async def _update_status(self, message, text: str, parse_mode=None):
        """Update status message."""
        try:
            await message.edit_text(text, parse_mode=parse_mode or ParseMode.HTML)
        except Exception:
            pass

    async def handle_feedback(self, callback: CallbackQuery):
        """Handle thumbs up/down feedback."""
        try:
            # Parse callback data: feedback:video_id:segment_index:rating:job_id
            parts = callback.data.split(":")
            if len(parts) != 5:
                await callback.answer("Data feedback tidak valid.")
                return

            _, video_id, segment_index, rating, job_id = parts
            segment_index = int(segment_index)
            rating = int(rating)
            user_id = callback.from_user.id

            # Save to database
            await self.db.save_feedback(
                video_url=f"https://youtube.com/watch?v={video_id}",
                video_id=video_id,
                segment_index=segment_index,
                rating=rating,
                user_id=user_id
            )

            # Get stats
            stats = await self.db.get_feedback_stats(video_id)

            emoji = "👍" if rating == 1 else "👎"
            await callback.answer(f"Terima kasih! Feedback {emoji} tersimpan.")

            # Update button to show thanks
            await callback.message.edit_reply_markup(reply_markup=None)

        except Exception as e:
            logger.error(f"Feedback error: {e}")
            await callback.answer("Gagal menyimpan feedback.")

    async def run(self):
        """Start the bot."""
        await self.db.init()
        logger.info(f"Bot started in {self.config.app_mode} mode")
        logger.info(f"Temp directory: {self.config.temp_base_path}")
        logger.info(f"Max concurrent jobs: {self.config.max_concurrent_jobs}")

        await self.dp.start_polling(self.bot)


# Import FSInputFile for sending local files
from aiogram.types import FSInputFile


async def main():
    """Main entry point."""
    bot = ClipperBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
