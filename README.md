# AI Video Clipper Bot

A Telegram bot that automatically converts YouTube videos into viral-ready 9:16 portrait clips with burned-in subtitles. Built on a modular 4-agent architecture with per-user toggle controls, retry logic, circuit breakers, and a feedback-driven improvement loop.

## Architecture

The system is structured as a sequential pipeline of four specialized agents, each with a single responsibility and structured data interfaces between stages.

```
User (Telegram) -> Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 -> User
```

| Agent | Role | Technology |
|-------|------|------------|
| Media Ingestion | Download and validate YouTube content | yt-dlp |
| Content Analyst | Identify viral segments via multimodal reasoning | Google Gemini 2.0 Flash |
| Transcriber | Speech-to-text with word-level timing | Groq Whisper |
| Video Editor | Cut, crop 9:16, burn subtitles | FFmpeg |

## Features

- **Inline Toggle Menu** — Users can switch settings directly from Telegram without editing config files
- **Per-User Settings** — Each user has independent configuration stored in SQLite
- **Mode Toggle (LOCAL / VPS)** — Switch between quality-focused local rendering and speed-focused VPS rendering
- **Transcriber Toggle (ON / OFF)** — Enable or disable subtitle generation to save API costs
- **Whisper Model Selector** — Choose between Large v3 (accuracy) or Turbo (2.8x cheaper)
- **Max Clips Selector** — Limit how many clips to generate per video (1-5)
- **Retry and Circuit Breaker** on the Gemini API. Falls back to a heuristic middle-segment selector if the AI service is unavailable
- **Concurrent Job Queue** using asyncio semaphores to limit simultaneous FFmpeg renders
- **User Feedback Loop** via inline Telegram buttons. Ratings are stored in SQLite for future prompt fine-tuning
- **Metadata Caching** to avoid repeated fetches for the same YouTube URL
- **Dynamic ASS Subtitles** with karaoke-style word highlighting

## Project Structure

```
telegram-video-bot/
├── .env                    # API keys and default mode switch
├── .env.example            # Template
├── requirements.txt        # Python dependencies
├── main.py                 # Bot entry point, job queue, toggle menu, and feedback handlers
│
├── models/                 # Pydantic schemas for structured agent communication
│   ├── config.py           # AppConfig, AppMode (LOCAL / VPS), model_copy()
│   ├── analysis.py         # AnalysisResult, TimestampSegment, FeedbackEntry
│   └── subtitle.py         # TranscriptionResult, WordTiming, SubtitleStyle
│
├── services/               # The four agent implementations
│   ├── downloader.py       # Agent 1: yt-dlp download with metadata cache
│   ├── ai_analyzer.py      # Agent 2: Gemini analysis with circuit breaker
│   ├── transcriber.py      # Agent 3: Groq Whisper transcription (model override support)
│   └── video_editor.py     # Agent 4: FFmpeg pipeline (preset override support)
│
├── utils/                  # Shared utilities
│   ├── validators.py       # URL validation, duration checks, filename sanitization
│   ├── file_manager.py     # Temp directory lifecycle and disk space monitoring
│   └── ass_builder.py      # ASS subtitle generator with word-highlight styling
│
└── temp/                   # Auto-created by config.py (LOCAL: D:/, VPS: /tmp/)
```

## Requirements

- Python 3.10+
- FFmpeg
- Telegram Bot Token (from BotFather)
- Google Gemini API Key
- Groq API Key

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/faqihrayhan/telegram-video-bot.git
cd telegram-video-bot
```

### 2. Install FFmpeg

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg -y

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 3. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
APP_MODE=LOCAL
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
MAX_VIDEO_DURATION_MINUTES=120
MAX_CONCURRENT_JOBS=2
```

### 5. Run the bot

```bash
python main.py
```

## Usage

### Main Menu

Send `/start` to open the inline menu:

```
Process Video
Mode: LOCAL          Transcriber: ON
Skill Manager        Settings
Help
```

### Toggle Controls

| Button | Function |
|--------|----------|
| **Mode** | Toggle between LOCAL (quality) and VPS (speed) |
| **Transcriber** | Toggle subtitle generation ON/OFF |
| **Skill Manager** | View agent pipeline status |
| **Settings** | Configure max clips, whisper model, auto cleanup |

### Settings Panel

```
Toggle Mode (LOCAL/VPS)
Toggle Transcriber
Auto Cleanup: ON
Max Clips
Whisper Model
Back to Main
```

- **Max Clips**: Choose 1-5 clips per video
- **Whisper Model**: Large v3 (best accuracy) or Turbo (2.8x cheaper)
- **Auto Cleanup**: Automatically delete temp files after processing

### Sending a Video

1. Toggle your preferred settings via the menu
2. Send any YouTube link directly in chat
3. The bot displays real-time status updates at each pipeline stage
4. Receive clips and rate them with inline buttons

## Configuration Modes

### Local Mode

```
Mode: LOCAL
```

- Temp directory: `D:/telegram-video-bot-temp/`
- FFmpeg preset: `medium` (better compression)
- Auto cleanup: disabled by default (manual)

### VPS Mode

```
Mode: VPS
```

- Temp directory: `/tmp/telegram-video-bot/`
- FFmpeg preset: `ultrafast` (faster render)
- Auto cleanup: enabled after delivery

Switching modes requires no code changes — users toggle directly from the Telegram menu.

## Cost Optimization

### Frame Sampling for Long Videos

Gemini analyzes video by extracting frames. For videos longer than 30 minutes, the bot automatically applies **frame sampling** (1 frame every 5 seconds instead of every 1 second) to reduce token usage:

| Video Duration | Frame Rate | Frames Analyzed | Token Savings |
|---------------|------------|-----------------|---------------|
| 10 minutes | 1 fps (default) | 600 frames | — |
| 45 minutes | 1 fps | 2,700 frames | Baseline |
| 45 minutes | 1 per 5s (auto) | 540 frames | **80% fewer tokens** |
| 2 hours | 1 fps | 7,200 frames | Baseline |
| 2 hours | 1 per 5s (auto) | 1,440 frames | **80% fewer tokens** |

Frame sampling is **automatic** — no user action required. The bot detects video duration and applies the appropriate sampling rate.

### Toggle Impact on Cost

| Toggle | Savings per 60-min Video | How to Use |
|--------|--------------------------|------------|
| **Transcriber OFF** | Rp 1,776 (54% total) | Click "Transcriber: ON" in menu |
| **Whisper Turbo** | Rp 1,136 (35% total) | Settings -> Whisper Model -> Turbo |
| **Mode VPS/LOCAL** | No cost difference | Only affects render speed |

### Cost Breakdown per Video

| Video Duration | Transcriber ON (v3) | Transcriber ON (Turbo) | Transcriber OFF |
|---------------|---------------------|------------------------|-----------------|
| 10 minutes | Rp 549 | Rp 360 | Rp 253 |
| 25 minutes | Rp 1,364 | Rp 891 | Rp 624 |
| 60 minutes | Rp 3,267 | Rp 2,131 | Rp 1,491 |
| 120 minutes | Rp 6,534 | Rp 4,262 | Rp 2,982 |

> Kurs: $1 = Rp 16,000. Gemini analysis is the fixed cost; Whisper transcription is the variable cost.

### Recommended Configurations

| Use Case | Setting | Cost (60 min) | Cost (120 min) |
|----------|---------|---------------|----------------|
| Budget mode | Transcriber OFF | Rp 1,491 | Rp 2,982 |
| Balanced | Turbo + ON | Rp 2,131 | Rp 4,262 |
| Maximum quality | v3 + ON + LOCAL | Rp 3,267 | Rp 6,534 |

## Error Handling

| Failure | Handling |
|---------|----------|
| Gemini API timeout / rate limit | Retry 3 times with exponential backoff; fallback to heuristic segment selection |
| Gemini circuit breaker open | Immediate fallback to middle-segment heuristic |
| Video exceeds duration limit | Rejected before download |
| FFmpeg render timeout (> 10 min) | Cancelled with error message |
| Insufficient disk space | Checked before render; rejected with warning |

## Feedback Database

User ratings are stored in SQLite (`temp/clipper.db`) with the following schema:

```sql
CREATE TABLE feedback (
    video_id TEXT,
    segment_index INTEGER,
    rating INTEGER,        -- -1 (dislike) or 1 (like)
    user_id INTEGER,
    timestamp TEXT,
    reasoning TEXT
);
```

This data can be used for:
- Fine-tuning the Gemini analysis prompt
- Segment quality analysis
- Confidence score recalibration

## Deployment

### VPS Deployment

Recommended providers: DigitalOcean, Vultr, Hetzner, or AWS Lightsail. Minimum 2 GB RAM is advised because FFmpeg rendering is memory-intensive.

1. Provision an Ubuntu 22.04 server.
2. Install Python, pip, venv, and FFmpeg.
3. Upload the project via `scp` or `git clone`.
4. Set `APP_MODE=VPS` in `.env`.
5. Run with a process manager such as `systemd` or `tmux` to keep the bot alive.

Example `systemd` service file:

```ini
[Unit]
Description=AI Video Clipper Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/telegram-video-bot
ExecStart=/root/telegram-video-bot/venv/bin/python /root/telegram-video-bot/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
systemctl daemon-reload
systemctl enable clipper-bot
systemctl start clipper-bot
journalctl -u clipper-bot -f
```

## Important Notes

- Gemini video uploads are temporary and stored in Google Cloud. Large videos may hit quota limits; consider frame sampling for content longer than 30 minutes.
- Groq Whisper has an approximate audio file size limit of 25 MB. Long videos may require audio splitting.
- The `ultrafast` FFmpeg preset produces larger files but renders faster, which is ideal for VPS environments. Use `medium` for better compression when running locally.
- Keep `MAX_CONCURRENT_JOBS` low (1 to 2) on small VPS instances to avoid CPU and memory exhaustion.
- Per-user settings are stored in SQLite and persist across bot restarts.

## Roadmap

- Face-tracking smart crop (follow the speaker instead of center crop)
- Multiple subtitle themes (CapCut, YouTube Shorts, TikTok)
- Auto-caption generation for non-speech content
- Batch processing for multiple URLs
- Web dashboard for job monitoring and feedback analytics
- Admin panel to view user settings and usage statistics

## License

MIT License
