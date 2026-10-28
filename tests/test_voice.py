from types import SimpleNamespace as NS

import pytest

from app import voice


def test_speakable_strips_marks_markdown_and_links():
    text = "**BOS** leads at 83.2¹ (see [BTS](https://x.gov)) and `code` https://y.com"
    assert voice.speakable(text) == "BOS leads at 83.2 (see BTS) and code"
    assert len(voice.speakable("a " * 1000)) <= 600


class FakeOpenAI:
    def __init__(self, down=False):
        self.sent, self.down = {}, down
        self.audio = NS(transcriptions=NS(create=self._stt), speech=NS(create=self._tts))

    def _stt(self, **kw):
        if self.down:
            raise ConnectionError("down")
        self.sent["stt"] = kw
        return NS(text="  Compare LAX and SNA \n")

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
