"""Live FAA NAS status. Returns structured, whitelisted fields only.

Boundary B3: the free-text <Reason> (a NOTAM) is attacker-reachable text, so it never leaves
this module. We derive one keyword flag from it in code and drop the rest.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET  # expat >= 2.4 (bundled with 3.13) blocks entity expansion
from collections.abc import Callable

import httpx

from airport_agent import config

log = logging.getLogger(__name__)
CODE_RE = re.compile(r"^[A-Z]{3}$")
TIME_RE = re.compile(r"^[A-Za-z0-9 :.,/-]{1,40}$")
TRENDS = {"Increasing", "Decreasing", "Steady"}
DIRECTIONS = {"Arrival", "Departure"}
PARTIAL_MARKERS = ("NON SKED", "TRANSIENT GA", "GA ACFT")


def _minutes(text: str | None) -> int | None:
    if not text:
        return None
    hours = re.search(r"(\d+)\s*hour", text)
    mins = re.search(r"(\d+)\s*min", text)
    if not hours and not mins:
        return None
    return (int(hours.group(1)) * 60 if hours else 0) + (int(mins.group(1)) if mins else 0)


def _safe_time(text: str | None) -> str | None:
    text = (text or "").strip()
    return text if TIME_RE.match(text) else None


def _first(el: ET.Element, tag: str) -> str | None:
    found = next(el.iter(tag), None)
    return found.text if found is not None else None


def parse_status(xml_text: str) -> tuple[str, list[dict]]:
    root = ET.fromstring(xml_text)
    updated = _safe_time(root.findtext("Update_Time")) or ""
    events = []
    for delay_type in root.iter("Delay_type"):
        name = (delay_type.findtext("Name") or "").strip()
        for el in delay_type.iter():
            code = (el.findtext("ARPT") or "").strip().upper()
            if not CODE_RE.match(code):
                continue
            reason = el.findtext("Reason") or ""
            ad = next(el.iter("Arrival_Departure"), None)
            direction = ad.get("Type") if ad is not None else None
            trend = (_first(el, "Trend") or "").strip()
            events.append({
                "airport": code,
                "type": name if TIME_RE.match(name) else "Delay",
                "scope": ("partial (non-scheduled / general aviation only)"
                          if any(m in reason for m in PARTIAL_MARKERS) else "full"),
                "direction": direction if direction in DIRECTIONS else None,
                "start": _safe_time(el.findtext("Start")),
                "reopen": _safe_time(el.findtext("Reopen")),
                "end": _safe_time(el.findtext("End_Time")),
                "min_delay_min": _minutes(_first(el, "Min")),
                "avg_delay_min": _minutes(_first(el, "Avg")),
                "max_delay_min": _minutes(_first(el, "Max")),
                "trend": trend if trend in TRENDS else None,
            })
    return updated, events


class FaaStatusClient:
    def __init__(self, url: str = config.FAA_STATUS_URL, timeout: float = config.FAA_TIMEOUT_S,
                 cache_s: float = config.FAA_CACHE_S, fetch: Callable[[], str] | None = None,
                 clock=time.monotonic):
        self.cache_s = cache_s
        self.clock = clock
        self.fetch = fetch or (lambda: httpx.get(url, timeout=timeout).raise_for_status().text)
        self._cached: tuple[float, str, list[dict]] | None = None

    def _events(self) -> tuple[str, list[dict]]:
        # ponytail: no lock; two threads may both refresh, which is harmless
        if self._cached and self.clock() - self._cached[0] < self.cache_s:
            return self._cached[1], self._cached[2]
        updated, events = parse_status(self.fetch())
        self._cached = (self.clock(), updated, events)
        return updated, events

    def status(self, codes: list[str]) -> dict:
        try:
            updated, events = self._events()
        except Exception as exc:  # live data is optional: report, never crash the turn
            log.warning("FAA status failed: %s", type(exc).__name__)
            return {"available": False, "reason": "FAA status service unreachable"}
        return {"available": True, "updated": updated,
                "airports": {c: [e for e in events if e["airport"] == c] for c in codes}}
