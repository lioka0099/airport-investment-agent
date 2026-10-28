import pytest

from airport_agent.security.sanitize import InputRejected, clean_user_input, filter_output

SRC = "https://data.bts.gov/x"


def test_strips_zero_width_and_control_chars():
    assert clean_user_input("Rank\u200b BOS\x07 \u202eplease  ") == "Rank BOS please"


def test_keeps_newlines():
    assert clean_user_input("line1\nline2") == "line1\nline2"


def test_nfkc_normalizes_lookalikes():
    assert clean_user_input("\uff33\uff26\uff2f") == "SFO"


@pytest.mark.parametrize("bad", ["", "   \u200b ", "x" * 2001])
def test_rejects_empty_and_too_long(bad):
    with pytest.raises(InputRejected):
        clean_user_input(bad)


def test_removes_images_even_from_allowed_hosts():
    out = filter_output(f"See ![x](https://evil.com/p.png?d=secret) and ![y]({SRC})", {SRC})
    assert "evil.com" not in out and "![" not in out


def test_keeps_allowed_links_and_unwraps_others():
    out = filter_output(f"[BTS]({SRC}) and [click](https://evil.com/?q=data)", {SRC})
    assert f"[BTS]({SRC})" in out
    assert "click" in out and "evil.com" not in out


def test_removes_bare_and_autolinks():
    out = filter_output("go to https://evil.com/a and <https://evil.com/b>", set())
    assert "evil.com" not in out
    assert out.count("[link removed]") == 2


def test_allowed_bare_url_survives():
    assert SRC in filter_output(f"source: {SRC}", {SRC})


@pytest.mark.parametrize(
    "bad",
    [
        "![[x]](https://evil.com/p.png?d=1)",
        "[[x]](https://evil.com/?d=1)",
        "[a\\]b](https://evil.com/?d=1)",
    ],
)
def test_bracket_bypasses_are_neutralized(bad):
    out = filter_output(bad, set())
    assert "evil.com" not in out and "](" not in out


def test_bracket_bypass_keeps_text():
    assert filter_output("[[x]](https://evil.com/?d=1)", set()) == "[x]"


def test_uppercase_scheme_is_removed():
    assert "EVIL" not in filter_output("see HTTPS://EVIL.COM/x", set())


def test_trailing_punctuation_does_not_hide_allowed_url():
    out = filter_output("see http://allowed.example/x.", {"http://allowed.example/x"})
    assert out == "see http://allowed.example/x."
