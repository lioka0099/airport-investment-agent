"""Voice: speech in (STT) and a spoken answer out (TTS), with a provider fallback."""

import io
import logging
import re
import wave

import numpy as np
from google.genai import types

from airport_agent import config

log = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 600
# Auto language detection heard "Compare airports" as Spanish "Comprar AirPods".
STT_LANGUAGE = "en"
STT_HINT = "US airport investment questions, e.g. Compare LAX and SNA. Airports: BOS, JFK, SFO."
# On silence the model "transcribes" the hint itself; nobody says this phrase, so it marks an echo.
HINT_ECHO = "airport investment questions, e.g."
# ponytail: a fixed loudness gate on 16-bit samples (max 32767). Room noise sits in the low
# hundreds, speech peaks in the thousands. Tune it if quiet speakers get dropped.
SILENCE_LEVEL = 500
# gpt-4o-mini-tts improvises tone and pace unless told; fixed instructions keep every answer alike.
VOICE_STYLE = ("Speak as a calm, friendly airport investment analyst: warm, clear, steady pace, "
               "neutral American accent, same tone every time.")


def is_silent(audio: bytes) -> bool:
    """True when the recording never gets louder than room noise. Not a WAV: let STT decide."""
    try:
        with wave.open(io.BytesIO(audio)) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except (wave.Error, EOFError):
        return False
    # 99th percentile, not max: one click or bump shouldn't count as speech
    return samples.size == 0 or np.percentile(np.abs(samples.astype(np.int32)), 99) < SILENCE_LEVEL


def speakable(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹*_`#>]", "", text)
    return " ".join(text.split())[:MAX_SPOKEN_CHARS]


def _openai_stt(audio: bytes, client, model: str) -> str:
    return client.audio.transcriptions.create(model=model, file=("question.wav", audio),
                                              language=STT_LANGUAGE, prompt=STT_HINT).text


def _gemini_stt(audio: bytes, client, model: str) -> str:
    cfg = types.GenerateContentConfig(audio_transcription_config=types.AudioTranscriptionConfig(
        language_codes=[f"{STT_LANGUAGE}-US"]))
    r = client.models.generate_content(
        model=model, contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav")],
        config=cfg)
    return "".join(p.audio_transcription.text for p in r.candidates[0].content.parts
                   if p.audio_transcription)


def _openai_tts(text: str, client, model: str, voice: str) -> tuple[bytes, str]:
    audio = client.audio.speech.create(model=model, voice=voice, input=text,
                                       instructions=VOICE_STYLE, response_format="mp3").read()
    return audio, "audio/mp3"


def _gemini_tts(text: str, client, model: str, voice: str) -> tuple[bytes, str]:
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    part = client.models.generate_content(model=model, contents=text, config=cfg)
    data = part.candidates[0].content.parts[0].inline_data
    return data.data, data.mime_type


STT = {"openai": _openai_stt, "google": _gemini_stt}
TTS = {"openai": _openai_tts, "google": _gemini_tts}


def _first(attempts):
    """Return the first provider's result that works; raise the last error if none does."""
    error = RuntimeError("No voice provider has an API key.")
    for provider, call in attempts:
        try:
            return call()
        except Exception as exc:
            # type only: the message may hold the key
            log.warning("Voice %s failed: %s", provider, type(exc).__name__)
            error = exc
    raise error


def transcribe(audio: bytes, clients: dict) -> str:
    """clients: {provider: SDK client} for the providers that have a key. "" means nothing said."""
    if is_silent(audio):
        return ""  # no API call, and nothing for the model to make up
    text = _first((p, lambda p=p, m=m: STT[p](audio, clients[p], m).strip())
                  for p, m in config.STT_CHAIN if p in clients)
    return "" if HINT_ECHO in text else text


def speak(text: str, clients: dict) -> tuple[bytes, str]:
    """Returns (audio, mime type): providers answer in different formats."""
    text = speakable(text)
    return _first((p, lambda p=p, m=m, v=v: TTS[p](text, clients[p], m, v))
                  for p, m, v in config.TTS_CHAIN if p in clients)
