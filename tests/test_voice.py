from types import SimpleNamespace

from app import voice


def test_speakable_strips_marks_markdown_and_links():
    text = "**BOS** leads at 83.2¹ (see [BTS](https://x.gov)) and `code` https://y.com"
    assert voice.speakable(text) == "BOS leads at 83.2 (see BTS) and code"
    assert len(voice.speakable("a " * 1000)) <= 600


class FakeClient:
    def __init__(self):
        self.sent = {}
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._stt),
            speech=SimpleNamespace(create=self._tts))

    def _stt(self, **kw):
        self.sent["stt"] = kw
        return SimpleNamespace(text="  Compare LAX and SNA \n")

    def _tts(self, **kw):
        self.sent["tts"] = kw
        return SimpleNamespace(read=lambda: b"mp3")


def test_transcribe_and_speak_use_configured_models():
    c = FakeClient()
    assert voice.transcribe(b"wav", c) == "Compare LAX and SNA"
    assert c.sent["stt"]["model"] == "gpt-4o-transcribe"
    assert c.sent["stt"]["file"] == ("question.wav", b"wav")
    assert c.sent["stt"]["language"] == "en"  # no auto-detect: it once heard Spanish
    assert voice.speak("Hi¹ **there**", c) == b"mp3"
    assert c.sent["tts"]["input"] == "Hi there" and c.sent["tts"]["voice"] == "alloy"
