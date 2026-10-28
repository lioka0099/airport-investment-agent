"""Voice: speech in (STT) and a spoken answer out (TTS)."""

import re

from airport_agent import config

MAX_SPOKEN_CHARS = 600
# Auto language detection heard "Compare airports" as Spanish "Comprar AirPods".
STT_LANGUAGE = "en"
STT_HINT = "US airport investment questions, e.g. Compare LAX and SNA. Airports: BOS, JFK, SFO."


def speakable(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹*_`#>]", "", text)
    return " ".join(text.split())[:MAX_SPOKEN_CHARS]


def transcribe(audio: bytes, client) -> str:
    result = client.audio.transcriptions.create(model=config.STT_MODEL,
                                                file=("question.wav", audio),
                                                language=STT_LANGUAGE, prompt=STT_HINT)
    return result.text.strip()


def speak(text: str, client) -> bytes:
    return client.audio.speech.create(model=config.TTS_MODEL, voice=config.TTS_VOICE,
                                      input=speakable(text), response_format="mp3").read()
