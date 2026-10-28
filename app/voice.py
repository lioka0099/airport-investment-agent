"""Voice: speech in (STT), spoken TL;DR out (TTS), and pre-generated status fillers."""

import re
from pathlib import Path

from airport_agent import config

FILLER_DIR = Path(__file__).parent / "assets" / "fillers"
FILLER_TEXT = {
    "finding": "Let me find those airports.",
    "numbers": "Running the numbers.",
    "checking": "Double-checking the figures.",
}
FILLER_FOR = {
    "resolve_airports": "finding",
    "rank_airports": "numbers", "compare_airports": "numbers", "airport_profile": "numbers",
    "route_mix": "numbers", "concentration_risk": "numbers",
    "verifying": "checking",
}
MAX_SPOKEN_CHARS = 600


def filler_for(event: dict, played: set[str]) -> Path | None:
    name = FILLER_FOR.get(event.get("tool") or event.get("stage"))
    if name is None or name in played:
        return None
    played.add(name)
    return FILLER_DIR / f"{name}.mp3"


def speakable(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹*_`#>]", "", text)
    return " ".join(text.split())[:MAX_SPOKEN_CHARS]


def transcribe(audio: bytes, client) -> str:
    result = client.audio.transcriptions.create(model=config.STT_MODEL,
                                                file=("question.wav", audio))
    return result.text.strip()


def speak(text: str, client) -> bytes:
    return client.audio.speech.create(model=config.TTS_MODEL, voice=config.TTS_VOICE,
                                      input=speakable(text), response_format="mp3").read()
