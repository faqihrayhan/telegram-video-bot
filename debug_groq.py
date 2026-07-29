"""Debug script: cek response mentah dari Groq, bukan lewat pipeline bot."""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

if len(sys.argv) < 2:
    print("Usage: python3 debug_groq.py <path_ke_file_audio>")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open(AUDIO_FILE, "rb") as f:
    response = client.audio.transcriptions.create(
        file=f,
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        language=None,
    )

print("=== Tipe response ===")
print(type(response))
print()
print("=== response.words ===")
words = getattr(response, "words", "!!! TIDAK ADA ATRIBUT 'words' !!!")
print(f"Tipe: {type(words)}")
print(f"Jumlah: {len(words) if hasattr(words, '__len__') else 'N/A'}")
print(f"Isi (5 pertama): {words[:5] if hasattr(words, '__getitem__') else words}")
print()
print("=== response.text ===")
print(getattr(response, "text", "TIDAK ADA")[:300])
print()
print("=== Full JSON dump ===")
if hasattr(response, "model_dump"):
    print(json.dumps(response.model_dump(), indent=2, default=str)[:3000])
else:
    print("Bukan pydantic model, tipe:", type(response))
