"""Generate the spoken status fillers once.

Run: PYTHONPATH=. uv run --env-file .env python scripts/make_fillers.py
"""

from openai import OpenAI

from app.voice import FILLER_DIR, FILLER_TEXT, speak


def main() -> None:
    client = OpenAI()
    FILLER_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in FILLER_TEXT.items():
        path = FILLER_DIR / f"{name}.mp3"
        path.write_bytes(speak(text, client))
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
