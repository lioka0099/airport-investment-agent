from airport_agent import config
from airport_agent.agent.grounding import format_value, missing_refs, parse_answer, render

MANIFEST = {
    "sources": {
        "t100": {"period": "2025-08 to 2026-07", "retrieved_at": "2026-10-28T08:00:00"}
    }
}
FACTS = {
    "SFO.load_factor": {"value": 0.8823, "unit": "pct", "source": "t100"},
    "SFO.passengers_l12m": {"value": 25123456.0, "unit": "count", "source": "t100"},
    "SFO.avg_taxi_out": {"value": 25.04, "unit": "minutes", "source": "ontime"},
    "SFO.score_overall": {"value": 83.24, "unit": "score", "source": "computed"},
}


def test_parse_answer_sections():
    p = parse_answer(
        "TL;DR: SFO is full.\n\nBody line.\n\nFOLLOW-UPS:\n- Why?\n* Compare with OAK?\n"
    )
    assert p.tldr == "SFO is full."
    assert p.body == "Body line."
    assert p.follow_ups == ["Why?", "Compare with OAK?"]


def test_parse_answer_without_markers_keeps_everything_in_body():
    p = parse_answer("Just text.")
    assert (p.tldr, p.body, p.follow_ups) == ("", "Just text.", [])


def test_missing_refs():
    assert missing_refs("{{SFO.load_factor}} {{SFO.nope}} {{XYZ.a}} {{SFO.nope}}", FACTS) == [
        "SFO.nope", "XYZ.a"]


def test_format_value_by_unit():
    assert format_value({"value": 0.8823, "unit": "pct"}) == "88.2%"
    assert format_value({"value": 25123456.0, "unit": "count"}) == "25,123,456"
    assert format_value({"value": 25.04, "unit": "minutes"}) == "25.0 min"
    assert format_value({"value": 1.537, "unit": "ratio"}) == "1.54×"
    assert format_value({"value": 1, "unit": "rank"}) == "#1"
    assert format_value({"value": 0.78, "unit": "pctl"}) == "78th percentile"
    assert format_value({"value": 0.82, "unit": "pctl"}) == "82nd percentile"
    assert format_value({"value": 0.113, "unit": "pctl"}) == "11th percentile"
    assert format_value({"value": 2603.3, "unit": "hhi"}) == "2,603"
    assert format_value({"value": 2400, "unit": "miles"}) == "2,400 mi"
    assert format_value({"value": "Both", "unit": "text"}) == "Both"


def test_render_substitutes_and_cites_per_source():
    r = render(
        "TL;DR: LF {{SFO.load_factor}}, {{SFO.passengers_l12m}} pax, taxi {{SFO.avg_taxi_out}}.",
        FACTS,
        MANIFEST,
    )
    assert r.text == "TL;DR: LF 88.2%¹, 25,123,456¹ pax, taxi 25.0 min²."
    assert [c["source"] for c in r.citations] == ["t100", "ontime"]
    assert r.citations[0]["period"] == "2025-08 to 2026-07"
    assert r.citations[1]["period"] is None
    assert r.refs == ["SFO.load_factor", "SFO.passengers_l12m", "SFO.avg_taxi_out"]
    assert r.allowed_urls == sorted(
        {config.SOURCES["t100"]["url"], config.SOURCES["ontime"]["url"]}
    )
    assert all(r.allowed_urls)


def test_render_allowed_urls_empty_when_source_has_no_url():
    r = render("Score {{SFO.score_overall}}.", FACTS, MANIFEST)
    assert r.citations[0]["source"] == "computed"
    assert r.allowed_urls == []


def test_render_flags_orphan_digits_but_not_list_numbers_or_codes():
    r = render("1. SFO ({{SFO.score_overall}}) beats B6 hubs by 42% since 2019.\n2. Next.",
               FACTS, MANIFEST)
    assert r.orphans == ["42%", "2019"]


def test_render_marks_unresolved():
    r = render("Score {{SFO.nope}}.", FACTS, MANIFEST)
    assert r.text == "Score n/a." and r.unresolved == ["SFO.nope"]


def test_render_null_or_unitless_fact_is_unresolved_not_a_crash():
    facts = {
        "SFO.a": {"value": None, "unit": "pct", "source": "t100"},
        "SFO.b": {"value": 5, "source": "t100"},
    }
    r = render("{{SFO.a}} and {{SFO.b}}", facts, MANIFEST)
    assert r.text.startswith("n/a and ")
    assert r.unresolved == ["SFO.a"]
    assert r.refs == ["SFO.b"]


def test_render_malformed_placeholders_are_unresolved():
    r = render("{{sfo.load_factor}} {{SFO.Load_Factor}} {{SFO.load factor}}", FACTS, MANIFEST)
    assert r.text == "n/a n/a n/a"
    assert r.unresolved == ["{{sfo.load_factor}}", "{{SFO.Load_Factor}}", "{{SFO.load factor}}"]


def test_render_unresolved_is_unique_in_first_seen_order():
    r = render("{{SFO.z}} {{SFO.y}} {{SFO.z}} {{bad}} {{bad}}", FACTS, MANIFEST)
    assert r.unresolved == ["SFO.z", "SFO.y", "{{bad}}"]


def test_parse_answer_accepts_loose_headings():
    p = parse_answer("tl;dr: Short.\n\nBody.\n\nFollow-ups:\n- One?\n- Two?")
    assert (p.tldr, p.body, p.follow_ups) == ("Short.", "Body.", ["One?", "Two?"])
    p = parse_answer("TL;DR: S.\n\nBody.\n\nFOLLOW UPS:\n- A?")
    assert p.follow_ups == []  # not a recognised heading: left alone
    p = parse_answer("TL;DR: S.\n\nBody.\n\nFollowups: - A?\n- B?")
    assert p.follow_ups == ["A?", "B?"] and p.body == "Body."


def test_unit_typed_by_the_writer_is_not_doubled():
    text = "{{SFO.avg_taxi_out}}-minute taxi, {{SFO.load_factor}}% full, {{SFO.avg_taxi_out}} wait"
    r = render(text, FACTS, MANIFEST)
    assert r.text == "25.0¹-minute taxi, 88.2²% full, 25.0 min¹ wait"
