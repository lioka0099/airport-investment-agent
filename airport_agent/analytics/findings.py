"""Structured findings: the conclusions code is allowed to hand the LLM.

The LLM explains these; it does not invent comparisons of its own.
"""

import math

import pandas as pd

from airport_agent import config


def significance(a: float, b: float) -> str:
    hi, lo = max(abs(a), abs(b)), min(abs(a), abs(b))
    if hi == 0:
        return "similar"
    if lo == 0:
        return "large"
    rel = (hi - lo) / lo
    if rel < config.SIGNIFICANCE_SIMILAR:
        return "similar"
    if rel > config.SIGNIFICANCE_LARGE:
        return "large"
    return "moderate"


def compare_findings(rows: pd.DataFrame, metrics: list[str]) -> list[dict]:
    out = []
    for metric in metrics:
        values = rows[metric].dropna()
        if len(values) < 2:
            continue
        leader, other = values.idxmax(), values.idxmin()
        hi, lo = float(values[leader]), float(values[other])
        out.append({"type": "comparison", "metric": metric, "leader": leader, "other": other,
                    "ratio": round(hi / lo, 2) if lo else None,
                    "significance": significance(hi, lo)})
    return out


def rank_findings(ranked: pd.DataFrame) -> list[dict]:
    out = []
    for code, r in ranked.iterrows():
        out.append({"type": "rank", "airport": code, "position": int(r["rank"]),
                    "of": len(ranked), "stable": bool(r["stable"]),
                    "rank_range": [int(r["rank_min"]), int(r["rank_max"])]})
        out.append({"type": "classification", "airport": code, "value": r["classification"]})
    if len(ranked) >= 2:
        first, second = ranked["score"].iloc[0], ranked["score"].iloc[1]
        both_valid = not (math.isnan(first) or math.isnan(second))
        if both_valid and significance(first, second) == "similar":
            out.append({"type": "close_call", "airports": list(ranked.index[:2])})
    return out


def dominant_cause_finding(code: str, mix: dict[str, float]) -> dict | None:
    if not mix:
        return None
    cause = max(mix, key=mix.get)
    return {"type": "dominant_cause", "airport": code, "cause": cause,
            "share": round(mix[cause], 3)}
