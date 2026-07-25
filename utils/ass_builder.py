"""ASS subtitle generator with CapCut-style word highlighting.

Generates Advanced SubStation Alpha (.ass) subtitle files with
karaoke-style word highlighting effects.
"""
import math
from pathlib import Path
from typing import List

from models import TranscriptionResult, WordTiming, SubtitleStyle


class ASSBuilder:
    """Builder for styled ASS subtitle files."""

    # ASS color format: &HAABBGGRR
    DEFAULT_STYLE = SubtitleStyle()

    def __init__(self, style: SubtitleStyle = None):
        self.style = style or self.DEFAULT_STYLE

    def _escape_ass_text(self, text: str) -> str:
        """Escape special characters for ASS format."""
        text = text.replace("\", "\\")
        text = text.replace("{", "\{")
        text = text.replace("}", "\}")
        return text

    def _generate_header(self) -> str:
        """Generate ASS file header with style definitions."""
        header = f"""[Script Info]
Title: AI Clipper Subtitle
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{self.style.font_name},{self.style.font_size},{self.style.primary_color},{self.style.secondary_color},{self.style.outline_color},{self.style.back_color},{1 if self.style.bold else 0},0,0,0,100,100,0,0,1,{self.style.outline},{self.style.shadow},{self.style.alignment},10,10,{self.style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        return header

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

    def _build_word_highlight_line(self, words: List[WordTiming], 
                                    line_start: float,
                                    line_end: float) -> str:
        """Build a single dialogue line with word-by-word highlighting.

        Uses ASS override tags to highlight current word in secondary color
        while keeping other words in primary color.
        """
        if not words:
            return ""

        # Build text with per-word color tags
        text_parts = []
        for i, word in enumerate(words):
            escaped = self._escape_ass_text(word.word)

            if self.style.highlight_current_word:
                # Use \k (karaoke) tag for word highlighting
                # Format: {\k<duration>}word
                duration_cs = int((word.end - word.start) * 100)
                if duration_cs < 1:
                    duration_cs = 1

                # Start with primary color, switch to secondary via karaoke
                part = f"{{\k{duration_cs}}}{escaped}"
            else:
                part = escaped

            text_parts.append(part)

        text = " ".join(text_parts)

        start_time = self._seconds_to_ass_time(line_start)
        end_time = self._seconds_to_ass_time(line_end)

        return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"

    def _group_words_into_lines(self, words: List[WordTiming], 
                                 max_chars_per_line: int = 28,
                                 max_words_per_line: int = 6) -> List[List[WordTiming]]:
        """Group words into display lines for better readability.

        Args:
            words: List of word timings.
            max_chars_per_line: Maximum characters per line.
            max_words_per_line: Maximum words per line.

        Returns:
            List of word groups (lines).
        """
        lines = []
        current_line = []
        current_chars = 0

        for word in words:
            word_len = len(word.word)

            if (current_chars + word_len > max_chars_per_line or 
                len(current_line) >= max_words_per_line):
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_chars = word_len
            else:
                current_line.append(word)
                current_chars += word_len + 1  # +1 for space

        if current_line:
            lines.append(current_line)

        return lines

    def build_ass(self, transcription: TranscriptionResult, 
                  output_path: Path,
                  max_chars_per_line: int = 28) -> Path:
        """Generate complete ASS file from transcription.

        Args:
            transcription: Transcription result with word timings.
            output_path: Where to save the .ass file.
            max_chars_per_line: Max characters per subtitle line.

        Returns:
            Path to generated ASS file.
        """
        lines = self._group_words_into_lines(
            transcription.words, 
            max_chars_per_line=max_chars_per_line
        )

        ass_content = self._generate_header()

        for line_words in lines:
            if not line_words:
                continue
            line_start = line_words[0].start
            line_end = line_words[-1].end

            # Add small buffer
            line_start = max(0, line_start - 0.1)
            line_end = line_end + 0.3

            dialogue = self._build_word_highlight_line(line_words, line_start, line_end)
            if dialogue:
                ass_content += dialogue + "\n"

        output_path.write_text(ass_content, encoding="utf-8")
        return output_path

    def build_ass_simple(self, transcription: TranscriptionResult,
                          output_path: Path) -> Path:
        """Generate simple ASS without word highlighting (fallback mode).

        Args:
            transcription: Transcription result.
            output_path: Where to save the .ass file.

        Returns:
            Path to generated ASS file.
        """
        ass_content = self._generate_header()

        # Group words into simple lines
        lines = self._group_words_into_lines(transcription.words)

        for line_words in lines:
            if not line_words:
                continue
            start = self._seconds_to_ass_time(max(0, line_words[0].start - 0.1))
            end = self._seconds_to_ass_time(line_words[-1].end + 0.3)
            text = " ".join(self._escape_ass_text(w.word) for w in line_words)

            ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"

        output_path.write_text(ass_content, encoding="utf-8")
        return output_path
