"""Grounding: the LLM writes {{CODE.metric}} placeholders; code fills in the numbers.

This makes numbers impossible to fabricate or misattribute: a placeholder either names a fact a
tool returned (and gets that exact value plus a citation) or it is caught as missing.
"""

import re
from dataclasses import dataclass, field

from airport_agent import config

PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\.([a-z0-9_]+)\}\}")
ANY_BRACES = re.compile(r"\{\{[^}]*\}\}")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
DIGITS = re.compile(r"(?<![A-Za-z_\d.,])\d+(?:[.,]\d+)*%?")
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
# If the writer typed the unit right after a placeholder anyway, drop ours (rule 1 is a prompt
# rule, so code guarantees it): unit -> (suffix format_value adds, what the writer may type).
TYPED_UNIT = {
    "pct": ("%", re.compile(r"\s*(%|percent\b)", re.I)),
    "minutes": (" min", re.compile(r"[\s-]*(minutes?|min)\b", re.I)),
    "miles": (" mi", re.compile(r"[\s-]*(miles?|mi)\b", re.I)),
    "pctl": (" percentile", re.compile(r"[\s-]*percentile\b", re.I)),
}
FOLLOW_UPS = re.compile(r"(?im)^[ \t]*follow-?ups:[ \t]*")  # own line, or a first item after it
TLDR = re.compile(r"(?i)tl;dr:")
BULLET = re.compile(r"^\s*[-*•]\s+(.*\S)")


@dataclass
class ParsedAnswer:
    tldr: str
    body: str
    follow_ups: list[str]


@dataclass
class Rendered:
    text: str
    refs: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    allowed_urls: list[str] = field(default_factory=list)


def parse_answer(text: str) -> ParsedAnswer:
    text = text.strip()
    follow_ups: list[str] = []
    if len(parts := FOLLOW_UPS.split(text, maxsplit=1)) == 2:
        text, tail = parts
        follow_ups = [m.group(1) for line in tail.splitlines() if (m := BULLET.match(line))]
    tldr, body = "", text.strip()
    if m := TLDR.match(body):
        first, _, rest = body.partition("\n")
        tldr, body = first[m.end():].strip(), rest.strip()
    return ParsedAnswer(tldr=tldr, body=body, follow_ups=follow_ups[:3])


def missing_refs(text: str, facts: dict[str, dict]) -> list[str]:
    return sorted({f"{c}.{m}" for c, m in PLACEHOLDER.findall(text)} - facts.keys())


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_value(fact: dict) -> str:
    v, unit = fact["value"], fact.get("unit")
    if unit == "text" or isinstance(v, str):
        return str(v)
    return {
        "pct": lambda: f"{v:.1%}",
        "pctl": lambda: f"{_ordinal(round(v * 100))} percentile",
        "count": lambda: f"{v:,.0f}",
        "minutes": lambda: f"{v:.1f} min",
        "score": lambda: f"{v:.1f}",
        "rank": lambda: f"#{int(v)}",
        "ratio": lambda: f"{v:.2f}×",
        "miles": lambda: f"{v:,.0f} mi",
        "hhi": lambda: f"{v:,.0f}",
    }.get(unit, lambda: f"{v:,}")()


def render(text: str, facts: dict[str, dict], manifest: dict) -> Rendered:
    out = Rendered(text="")
    numbers: dict[str, int] = {}  # source -> footnote number, in order of first use

    def substitute(match: re.Match) -> str:
        ref = f"{match.group(1)}.{match.group(2)}"
        fact = facts.get(ref)
        if fact is None or fact.get("value") is None:
            out.unresolved.append(ref)
            return "n/a"
        out.refs.append(ref)
        source = fact["source"]
        if source not in numbers:
            numbers[source] = len(numbers) + 1
            meta = config.SOURCES.get(source, {"label": source, "url": ""})
            run = manifest.get("sources", {}).get(source, {})
            out.citations.append({"n": numbers[source], "source": source, "label": meta["label"],
                                  "url": meta["url"], "period": run.get("period"),
                                  "retrieved_at": run.get("retrieved_at")})
        value = format_value(fact)
        suffix, typed = TYPED_UNIT.get(fact.get("unit"), (None, None))
        if typed and typed.match(match.string, match.end()):
            value = value.removesuffix(suffix)
        return value + str(numbers[source]).translate(SUPERSCRIPT)

    def leftover(match: re.Match) -> str:  # malformed placeholder: never shown to the user
        out.unresolved.append(match.group(0))
        return "n/a"

    out.text = ANY_BRACES.sub(leftover, PLACEHOLDER.sub(substitute, text))
    out.unresolved = list(dict.fromkeys(out.unresolved))
    # Orphans: digits the LLM typed itself (not from a placeholder). The reviewer judges them.
    typed = LIST_MARKER.sub("", PLACEHOLDER.sub("", text))
    out.orphans = DIGITS.findall(typed)
    out.allowed_urls = sorted({c["url"] for c in out.citations if c["url"]})
    return out
