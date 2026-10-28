"""Pure helpers for the chat UI: labels, charts, tables, badges. No Streamlit calls here."""

import json

import altair as alt
import pandas as pd

from airport_agent.agent.grounding import SUPERSCRIPT
from airport_agent.security.sanitize import filter_output

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

STATUS_LABELS = {
    "resolve_airports": "Finding the airports…",
    "rank_airports": "Scoring airports…",
    "compare_airports": "Comparing airports…",
    "airport_profile": "Building the airport profile…",
    "route_mix": "Analyzing routes…",
    "concentration_risk": "Checking airline concentration…",
    "live_status": "Checking live FAA status…",
    "thinking": "Thinking…",
    "fixing": "Fixing number references…",
    "verifying": "Double-checking the answer…",
}
MINUTE_METRICS = {"avg_taxi_out": "Taxi-out (runway queue)", "avg_taxi_in": "Taxi-in (gate wait)",
                  "nas_min_per_flight": "NAS delay per arrival"}
CONFIDENCE_ORDER = ["low", "medium", "high"]


def status_label(event: dict) -> str:
    return STATUS_LABELS.get(event.get("tool") or event.get("stage"), "Working…")


def md_safe(text: str) -> str:
    return text.replace(r"\$", "$").replace("$", r"\$")  # already-escaped $ stays escaped once


def chart(tool_data: list[dict]) -> alt.Chart | None:
    for t in reversed(tool_data):
        d = t.get("data") or {}
        if t["tool"] == "rank_airports" and d.get("ranked"):
            return alt.Chart(pd.DataFrame(d["ranked"])).mark_bar().encode(
                x=alt.X("score:Q", title="Opportunity Score (0-100)",
                        scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("code:N", sort="-x", title=None),
                color=alt.Color("classification:N", title="Bottleneck"),
                tooltip=["code", "name", "score", "classification", "confidence"])
        if t["tool"] == "compare_airports" and d.get("airports"):
            df = (pd.DataFrame(d["airports"])
                  .melt(id_vars=["code"], value_vars=list(MINUTE_METRICS), var_name="metric",
                        value_name="minutes").dropna())
            df["metric"] = df["metric"].map(MINUTE_METRICS)
            return alt.Chart(df).mark_bar().encode(
                x=alt.X("minutes:Q", title="Minutes per flight"),
                y=alt.Y("metric:N", title=None), yOffset="code:N",
                color=alt.Color("code:N", title="Airport"), tooltip=["code", "metric", "minutes"])
        if t["tool"] == "airport_profile" and d.get("trend"):
            return alt.Chart(pd.DataFrame(d["trend"])).mark_line(point=True).encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("passengers:Q", title="Passengers per month"),
                tooltip=["month", "passengers", alt.Tooltip("load_factor:Q", format=".1%")])
        if t["tool"] == "route_mix" and d.get("buckets"):
            df = pd.DataFrame({"distance": list(d["buckets"]),
                               "share": list(d["buckets"].values())})
            return alt.Chart(df).mark_bar().encode(
                x=alt.X("distance:N", sort=None, title="Flight distance"),
                y=alt.Y("share:Q", title="Share of departures", axis=alt.Axis(format="%")),
                tooltip=["distance", alt.Tooltip("share:Q", format=".1%")])
        if t["tool"] == "concentration_risk" and d.get("shares"):
            df = pd.DataFrame({"carrier": list(d["shares"]), "share": list(d["shares"].values())})
            return alt.Chart(df.head(8)).mark_bar().encode(
                x=alt.X("share:Q", title="Share of departures", axis=alt.Axis(format="%")),
                y=alt.Y("carrier:N", sort="-x", title=None),
                tooltip=["carrier", alt.Tooltip("share:Q", format=".1%")])
    return None


def tables(tool_data: list[dict]) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for t in tool_data:
        d, code = t.get("data") or {}, filter_output(str(t["args"].get("code", "")), set())
        if t["tool"] == "rank_airports" and d.get("ranked"):
            out.append(("Ranking", pd.DataFrame(d["ranked"])))
        elif t["tool"] == "compare_airports" and d.get("airports"):
            out.append(("Comparison", pd.DataFrame(d["airports"])))
        elif t["tool"] == "airport_profile" and d.get("trend"):
            out.append((f"{code} monthly traffic", pd.DataFrame(d["trend"])))
        elif t["tool"] == "route_mix" and d.get("top_destinations"):
            out.append((f"{code} routes", pd.DataFrame(d["top_destinations"])))
        elif t["tool"] == "concentration_risk" and d.get("shares"):
            out.append((f"{code} carrier shares",
                        pd.DataFrame({"carrier": list(d["shares"]),
                                      "share": list(d["shares"].values())})))
    return out


def confidence(tool_data: list[dict]) -> str | None:
    levels = [r["confidence"] for t in tool_data if t["tool"] == "rank_airports"
              for r in (t.get("data") or {}).get("ranked", []) if r.get("confidence")]
    known = [c for c in levels if c in CONFIDENCE_ORDER]
    return min(known, key=CONFIDENCE_ORDER.index) if known else None


def verification_badge(v: dict) -> str:
    if v["status"] == "verified":
        return f"✓ Verified by a second AI ({v['reviewer']})"
    # Issues are LLM text (reviewer, or echoes of the draft): no links or images allowed.
    return "⚠️ Not verified: " + "; ".join(md_safe(filter_output(i, set())) for i in v["issues"])


def sources_md(citations: list[dict]) -> str:
    lines = []
    for c in citations:
        mark = str(c["n"]).translate(SUPERSCRIPT)
        label = f"[{c['label']}]({c['url']})" if c["url"] else c["label"]
        extra = [p for p in (c.get("period"),
                             c.get("retrieved_at") and f"retrieved {c['retrieved_at'][:10]}") if p]
        lines.append(f"{mark} {label}" + (f" — {', '.join(extra)}" if extra else ""))
    return "  \n".join(lines)


def _span(args: dict) -> str:
    # Tool args can hold model-chosen text; a backtick would close the code span.
    return json.dumps(args).replace("`", "'")


def steps_md(tool_data: list[dict]) -> str:
    return "\n".join(f"{i}. `{t['tool']}` with `{_span(t['args'])}`"
                     for i, t in enumerate(tool_data, 1))
