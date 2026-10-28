import io
import wave
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app import voice


def test_speakable_strips_marks_markdown_and_links():
    text = "**BOS** leads at 83.2¹ (see [BTS](https://x.gov)) and `code` https://y.com"
    assert voice.speakable(text) == "BOS leads at 83.2 (see BTS) and code"
    assert len(voice.speakable("a " * 1000)) <= 600


def wav(amplitude: int) -> bytes:
    """Half a second of a 16 kHz tone; amplitude 0 is silence."""
    t = np.arange(8000) / 16000
    samples = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


class FakeOpenAI:
    def __init__(self, down=False, heard="  Compare LAX and SNA \n"):
        self.sent, self.down, self.heard = {}, down, heard
        self.audio = NS(transcriptions=NS(create=self._stt), speech=NS(create=self._tts))

    def _stt(self, **kw):
        if self.down:
            raise ConnectionError("down")
        self.sent["stt"] = kw
        return NS(text=self.heard)

    def _tts(self, **kw):
        if self.down:
            raise ConnectionError("down")
        self.sent["tts"] = kw
        return NS(read=lambda: b"mp3")


class FakeGemini:
    def __init__(self):
        self.sent = []
        self.models = NS(generate_content=self._generate)

    def _generate(self, **kw):
        self.sent.append(kw)
        if kw["model"].endswith("tts"):
            part = NS(inline_data=NS(data=b"wav", mime_type="audio/wav"))
        else:
            part = NS(audio_transcription=NS(text="Compare BOS and JFK."))
        return NS(candidates=[NS(content=NS(parts=[part]))])


def test_openai_is_used_first_with_english_fixed():
    c = FakeOpenAI()
    clients = {"openai": c, "google": FakeGemini()}
    assert voice.transcribe(b"wav", clients) == "Compare LAX and SNA"
    assert c.sent["stt"]["model"] == "gpt-4o-transcribe"
    assert c.sent["stt"]["language"] == "en"  # no auto-detect: it once heard Spanish
    assert voice.speak("Hi¹ **there**", clients) == (b"mp3", "audio/mp3")
    assert c.sent["tts"]["input"] == "Hi there" and c.sent["tts"]["voice"] == "alloy"


def test_gemini_takes_over_when_openai_fails():
    g = FakeGemini()
    clients = {"openai": FakeOpenAI(down=True), "google": g}
    assert voice.transcribe(b"wav", clients) == "Compare BOS and JFK."
    assert voice.speak("Hi **there**", clients) == (b"wav", "audio/wav")
    assert [s["model"] for s in g.sent] == ["gemini-3.5-transcribe", "gemini-3.8-flash-tts"]
    assert g.sent[1]["contents"] == "Hi there"


def test_all_providers_down_raises_and_no_keys_raises():
    with pytest.raises(ConnectionError):
        voice.transcribe(b"wav", {"openai": FakeOpenAI(down=True)})
    with pytest.raises(RuntimeError):
        voice.speak("hi", {})


def test_silence_skips_the_providers():
    c = FakeOpenAI()
    assert voice.transcribe(wav(0), {"openai": c}) == ""
    assert voice.transcribe(wav(200), {"openai": c}) == ""  # room noise
    assert "stt" not in c.sent
    assert voice.transcribe(wav(8000), {"openai": c}) == "Compare LAX and SNA"


def test_an_echoed_hint_counts_as_nothing_said():
    echo = f"context: ### {voice.STT_HINT}"
    assert voice.transcribe(wav(8000), {"openai": FakeOpenAI(heard=echo)}) == ""
