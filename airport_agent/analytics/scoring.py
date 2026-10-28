"""Opportunity Score: percentile components, weighted sum, bottleneck class, sensitivity.

Percentiles (not min-max) so that giants like ATL do not compress everyone else.
"""

import math
from dataclasses import dataclass

import pandas as pd

from airport_agent import config

COMPONENTS = ("demand", "growth", "terminal", "airside", "scale")
FOCUS_CLASSES: dict[str, set[str] | None] = {
    "overall": None,
    "terminal": {"Terminal-constrained", "Both"},
    "airside": {"Airside-constrained", "Both"},
}
UNKNOWN = "Unknown (no On-Time data)"
METRIC_COLUMNS = ["name", "state", "passengers_l12m", "load_factor", "growth_yoy", "avg_taxi_in",
                  "avg_taxi_out", "nas_min_per_flight", "ontime_coverage", "months_t100"]


@dataclass
class Ranking:
    ranked: pd.DataFrame
    excluded: list[dict]
    weights: dict[str, float]


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = {**config.WEIGHTS, **(weights or {})}
    unknown = set(merged) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown weight components: {sorted(unknown)}")
    if any(v < 0 for v in merged.values()):
        raise ValueError("Weights must be non-negative")
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return {k: merged[k] / total for k in COMPONENTS}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def classify(terminal_strain: float | None, airside_strain: float | None) -> str:
    if _missing(terminal_strain) or _missing(airside_strain):
        return UNKNOWN
    terminal = terminal_strain >= config.STRAIN_HIGH_PCT
    airside = airside_strain >= config.STRAIN_HIGH_PCT
    if terminal and airside:
        return "Both"
    if terminal:
        return "Terminal-constrained"
    if airside:
        return "Airside-constrained"
    return "Unconstrained"


def confidence(months_t100: int, ontime_coverage: float | None) -> str:
    if _missing(ontime_coverage):
        return "low"
    full_history = months_t100 >= config.CONFIDENCE_FULL_MONTHS
    if full_history and ontime_coverage >= config.CONFIDENCE_HIGH_COVERAGE:
        return "high"
    if ontime_coverage >= config.CONFIDENCE_MEDIUM_COVERAGE:
        return "medium"
    return "low"


def _components(peers: pd.DataFrame) -> pd.DataFrame:
    def pct(col: str) -> pd.Series:
        return peers[col].rank(pct=True)  # NaN stays NaN

    airside = pd.concat([pct("avg_taxi_out"), pct("nas_min_per_flight")], axis=1).mean(axis=1)
    return pd.DataFrame({
        "demand": pct("load_factor"),
        "growth": pct("growth_yoy"),
        "terminal": pct("avg_taxi_in"),
        "airside": airside,
        "scale": pct("log_passengers"),
    })


def _score(comp: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights)[list(COMPONENTS)]
    used = (comp.notna() * w).sum(axis=1).replace(0, float("nan"))
    return (comp.fillna(0) * w).sum(axis=1) / used * 100


def _ranks(scores: pd.Series) -> pd.Series:
    return scores.rank(ascending=False, method="min").fillna(len(scores)).astype(int)


def _sensitivity(comp: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    scenarios = [_ranks(_score(comp, weights))] + [
        _ranks(_score(comp, normalize_weights({**weights, c: weights[c] * f})))
        for c in COMPONENTS for f in config.SENSITIVITY_FACTORS
    ]
    all_ranks = pd.concat(scenarios, axis=1)
    return all_ranks.min(axis=1), all_ranks.max(axis=1)


def score_airports(metrics: pd.DataFrame, codes: list[str] | None = None,
                   weights: dict[str, float] | None = None, focus: str = "overall") -> Ranking:
    if focus not in FOCUS_CLASSES:
        raise ValueError(f"focus must be one of {list(FOCUS_CLASSES)}")
    w = normalize_weights(weights)
    peers = metrics[metrics["passengers_l12m"] >= config.PEER_MIN_PASSENGERS]
    comp = _components(peers)

    wanted = list(dict.fromkeys(codes)) if codes else list(peers.index)
    excluded = []
    for code in wanted:
        if code not in metrics.index:
            excluded.append({"code": code, "reason": "no T-100 traffic in the snapshot"})
        elif code not in peers.index:
            pax = int(metrics.loc[code, "passengers_l12m"])
            excluded.append({"code": code, "reason": (
                f"below the {config.PEER_MIN_PASSENGERS:,} passenger peer threshold "
                f"({pax:,} in the last 12 months)")})
    subset = [c for c in wanted if c in peers.index]

    out = peers.loc[subset, METRIC_COLUMNS].join(comp.loc[subset])
    out["classification"] = [
        classify(t, a) for t, a in zip(out["terminal"], out["airside"], strict=True)]
    out["confidence"] = [confidence(m, c) for m, c in
                         zip(out["months_t100"], out["ontime_coverage"], strict=True)]

    allowed = FOCUS_CLASSES[focus]
    if allowed:
        drop = ~out["classification"].isin(allowed)
        excluded += [{"code": c, "reason": f"classified {cls}, not a {focus} candidate"}
                     for c, cls in out.loc[drop, "classification"].items()]
        out = out[~drop]

    sub = out[list(COMPONENTS)]
    out = out.assign(score=_score(sub, w))
    out["rank"] = _ranks(out["score"])
    out["rank_min"], out["rank_max"] = _sensitivity(sub, w)
    out["stable"] = out["rank_min"] == out["rank_max"]
    return Ranking(ranked=out.sort_values("score", ascending=False), excluded=excluded, weights=w)
