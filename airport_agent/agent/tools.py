"""The 7 read-only tools the agent can call. Each returns a ToolResult envelope.

Boundary B2: tool arguments come from the LLM, so they are untrusted input. Pydantic checks
shape and ranges; each tool also checks that airport codes exist. run_tool() converts every
failure into an error envelope the LLM can recover from.
"""

import difflib
import logging
import math
from typing import Annotated, Literal

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from airport_agent import config
from airport_agent.agent.envelope import Fact, ToolResult
from airport_agent.analytics import findings, scoring
from airport_agent.analytics import metrics as metrics_mod
from airport_agent.data.faa_status import FaaStatusClient
from airport_agent.data.store import Store

log = logging.getLogger(__name__)

Code = Annotated[str, BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v),
                 StringConstraints(pattern=r"^[A-Z]{3}$")]

# metric -> (unit, source)
METRIC_FACTS = {
    "name": ("text", "ourairports"), "city": ("text", "ourairports"),
    "state": ("text", "ourairports"),
    "passengers_l12m": ("count", "t100"), "seats_l12m": ("count", "t100"),
    "load_factor": ("pct", "t100"), "growth_yoy": ("pct", "t100"), "vs_2019": ("pct", "t100"),
    "seats_gap": ("count", "t100"), "intl_share": ("pct", "t100"),
    "avg_taxi_out": ("minutes", "ontime"), "avg_taxi_in": ("minutes", "ontime"),
    "nas_min_per_flight": ("minutes", "ontime"), "dep_del15_rate": ("pct", "ontime"),
    "cancel_rate": ("pct", "ontime"), "ontime_coverage": ("pct", "computed"),
    "score": ("score", "computed"), "rank": ("rank", "computed"),
    "classification": ("text", "computed"), "confidence": ("text", "computed"),
    **{f"{c}_pctl": ("pctl", "computed") for c in scoring.COMPONENTS},
}
PCTL_KEYS = [f"{c}_pctl" for c in scoring.COMPONENTS]
RANK_KEYS = ["name", "score", "rank", "classification", "confidence", "passengers_l12m",
             "load_factor", "growth_yoy", "avg_taxi_in", "avg_taxi_out", "nas_min_per_flight",
             *PCTL_KEYS]
COMPARE_METRICS = ["passengers_l12m", "load_factor", "growth_yoy", "avg_taxi_out", "avg_taxi_in",
                   "nas_min_per_flight", "dep_del15_rate", "cancel_rate"]
PROFILE_KEYS = ["name", "city", "state", "passengers_l12m", "seats_l12m", "load_factor",
                "growth_yoy", "vs_2019", "seats_gap", "intl_share", "avg_taxi_out", "avg_taxi_in",
                "nas_min_per_flight", "dep_del15_rate", "cancel_rate", "ontime_coverage"]

PEER_NOTE = (f"Scores use percentiles among US airports with at least "
             f"{config.PEER_MIN_PASSENGERS:,} passengers in the last 12 months.")
ONTIME_NOTE = ("Taxi times, delays and routes come from BTS On-Time data, which covers domestic "
               "flights of reporting carriers only (no international or cargo flights).")
AIRSIDE_CAVEAT = ("Airside strain mixes runway capacity with airspace/ATC limits (part of NAS "
                  "delay), so not all of it can be fixed by building at the airport.")
NOT_RETURN = "The score ranks opportunity (demand pressing on capacity), not financial return."
FOCUS_NOTES = {
    "terminal": "Terminal focus keeps airports classified Terminal-constrained or Both. For Both, "
                "terminal work alone will not unlock growth; a combined project is needed.",
    "airside": "Airside focus keeps airports classified Airside-constrained or Both.",
}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _facts(code: str, values: dict, keys: list[str]) -> dict[str, Fact]:
    out = {}
    for key in keys:
        v = values.get(key)
        if _missing(v):
            continue
        unit, source = METRIC_FACTS[key]
        out[f"{code}.{key}"] = Fact(value=str(v) if unit == "text" else float(v), unit=unit,
                                    source=source)
    return out


def _with_pctl(row: pd.Series) -> dict:
    return row.to_dict() | {f"{c}_pctl": row[c] for c in scoring.COMPONENTS}


def base_facts(manifest: dict) -> dict[str, Fact]:
    facts = {
        "CONFIG.peer_min_passengers": Fact(value=config.PEER_MIN_PASSENGERS, unit="count",
                                           source="computed"),
        "CONFIG.lf_target": Fact(value=config.LF_TARGET, unit="pct", source="computed"),
        "CONFIG.strain_high_pctl": Fact(value=config.STRAIN_HIGH_PCT, unit="pctl",
                                        source="computed"),
        "CONFIG.long_haul_mi": Fact(value=config.LONG_HAUL_MI, unit="miles", source="computed"),
        "DATA.period_start": Fact(value=manifest["l12m_start"][:7], unit="text", source="t100"),
        "DATA.period_end": Fact(value=manifest["l12m_end"][:7], unit="text", source="t100"),
    }
    for c, w in config.WEIGHTS.items():
        facts[f"CONFIG.weight_{c}"] = Fact(value=w, unit="pct", source="computed")
    return facts


def run_tool(tool: StructuredTool, args: dict) -> ToolResult:
    try:
        return tool.invoke(args)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
        return ToolResult(error={"code": "invalid_arguments", "message": detail})
    except Exception:
        log.exception("Tool %s failed", tool.name)
        return ToolResult(error={"code": "tool_failed",
                                 "message": "The tool failed unexpectedly. Try fewer airports or "
                                            "a different question."})


# ---------- argument schemas ----------

class ResolveArgs(BaseModel):
    query: str = Field(min_length=1, max_length=100,
                       description="Airport name, city, metro (e.g. 'LA'), US state, region "
                                   "(e.g. 'New England') or IATA code")
    states: list[str] | None = Field(
        None, max_length=60,
        description="Only for a region that is not built in (e.g. 'Great Lakes'): the 2-letter "
                    "US state codes it covers. Shown to the user as an assumption.")


class RankArgs(BaseModel):
    codes: list[Code] | None = Field(None, max_length=60,
                                     description="IATA codes to rank; omit to rank all US peers")
    focus: Literal["overall", "terminal", "airside"] = "overall"
    top_n: int = Field(10, ge=1, le=50)
    weights: dict[str, float] | None = Field(
        None, description="Optional weight overrides for demand, growth, terminal, airside, scale")


class CompareArgs(BaseModel):
    codes: list[Code] = Field(min_length=2, max_length=5)

    @field_validator("codes")
    @classmethod
    def no_duplicates(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("airport codes must be distinct")
        return v


class CodeArgs(BaseModel):
    code: Code


class RouteArgs(BaseModel):
    code: Code
    long_haul_threshold_mi: int = Field(config.LONG_HAUL_MI, ge=500, le=6000)


class LiveArgs(BaseModel):
    codes: list[Code] = Field(min_length=1, max_length=10)


def build_tools(store: Store, faa: FaaStatusClient) -> list[StructuredTool]:
    m = metrics_mod.airport_metrics(store)
    known = list(m.index)
    manifest = store.manifest()
    coverage = {
        "period": f"{manifest['l12m_start'][:7]} to {manifest['l12m_end'][:7]}",
        "ontime_missing_months": manifest["sources"].get("ontime", {}).get("missing_months", []),
    }

    def unknown(codes: list[str]) -> ToolResult | None:
        missing = [c for c in codes if c not in m.index]
        if not missing:
            return None
        return ToolResult(error={
            "code": "unknown_airport", "airports": missing,
            "suggestions": {c: difflib.get_close_matches(c, known, n=3, cutoff=0.3)
                            for c in missing},
            "message": "Unknown airport code. Call resolve_airports to find the right code."})

    def by_pax(codes) -> list[str]:
        return sorted((c for c in codes if c in m.index),
                      key=lambda c: -m.loc[c, "passengers_l12m"])

    def match_rows(codes: list[str]) -> list[dict]:
        return [{"code": c, "name": m.loc[c, "name"], "city": m.loc[c, "city"],
                 "state": m.loc[c, "state"]} for c in codes]

    # ---------- tools ----------

    def resolve_airports(query: str, states: list[str] | None = None) -> ToolResult:
        q = " ".join(query.lower().split())
        llm_states = [s.upper() for s in states or []]
        states, alternatives, assumptions = None, [], []
        if q in config.REGIONS:  # the built-in definition wins over the LLM's
            states = config.REGIONS[q]
            how = f"the {query} region ({', '.join(states)})"
        elif llm_states:
            bad_states = [s for s in llm_states if s not in config.US_STATES.values()]
            if bad_states:
                bad = ", ".join(bad_states)
                return ToolResult(error={"code": "invalid_arguments",
                                         "message": f"Not US state codes: {bad}."})
            states = llm_states
            how = (f"{', '.join(states)} (the assistant's reading of '{query}'; not a built-in "
                   "region, so correct it if you meant different states)")
        elif q in config.US_STATES:
            states, how = [config.US_STATES[q]], f"the state of {query.title()}"
        elif q in config.METROS:
            metro = config.METROS[q]
            codes = [c for c in metro["primary"] if c in m.index]
            alternatives = [c for c in metro["alternatives"] if c in m.index]
            how = f"{', '.join(codes)} (the main airport for '{query}')"
            if alternatives:
                assumptions.append(f"Interpreted '{query}' as {', '.join(codes)}. Other airports "
                                   f"in the area: {', '.join(alternatives)}. Ask to include them "
                                   "if you meant the whole metro area.")
        elif q.upper() in m.index:
            codes, how = [q.upper()], f"airport code {q.upper()}"
        elif len(q) == 2 and q.upper() in config.US_STATES.values():
            states, how = [q.upper()], f"the state {q.upper()}"
        else:
            hit = (m["name"].str.lower().str.contains(q, regex=False)
                   | m["city"].str.lower().str.contains(q, regex=False))
            codes, how = by_pax(m.index[hit])[:5], f"airports matching '{query}'"
            if len(codes) > 1:
                assumptions.append("Several airports match; the busiest are listed first.")

        small: list[str] = []
        if states is not None:
            in_states = by_pax(m.index[m["state"].isin(states)])
            codes = [c for c in in_states
                     if m.loc[c, "passengers_l12m"] >= config.PEER_MIN_PASSENGERS]
            small = [c for c in in_states if c not in codes]
            assumptions.append(f"Interpreted '{query}' as {how}.")
            if small:
                assumptions.append(
                    f"{len(small)} smaller airports (under {config.PEER_MIN_PASSENGERS:,} "
                    f"passengers) are not scored: {', '.join(small[:10])}"
                    f"{' and more' if len(small) > 10 else ''}.")
        if not codes:
            return ToolResult(error={"code": "not_found",
                                     "message": f"No US airport matches '{query}'.",
                                     "suggestions": small[:5]})
        facts = {}
        for c in codes + alternatives:
            facts |= _facts(c, m.loc[c].to_dict(), ["name", "passengers_l12m"])
        return ToolResult(
            data={"interpretation": how, "matches": match_rows(codes),
                  "alternatives": alternatives, "small_airports": small},
            facts=facts, assumptions=assumptions, data_coverage=coverage)

    def rank_airports(codes: list[str] | None = None, focus: str = "overall", top_n: int = 10,
                      weights: dict[str, float] | None = None) -> ToolResult:
        if codes and (err := unknown(codes)):
            return err
        try:
            ranking = scoring.score_airports(m, codes, weights, focus)
        except ValueError as exc:
            return ToolResult(error={"code": "invalid_arguments", "message": str(exc)})
        shown = ranking.ranked.head(top_n)
        facts, rows = {}, []
        for code, r in shown.iterrows():
            facts |= _facts(code, _with_pctl(r), RANK_KEYS)
            rows.append({
                "code": code, "name": r["name"], "state": r["state"],
                "score": round(float(r["score"]), 1), "rank": int(r["rank"]),
                "rank_range": [int(r["rank_min"]), int(r["rank_max"])],
                "stable": bool(r["stable"]), "classification": r["classification"],
                "confidence": r["confidence"],
                **{c: None if _missing(r[c]) else round(float(r[c]), 3)
                   for c in scoring.COMPONENTS},
            })
        facts |= {f"WEIGHTS.{k}": Fact(value=v, unit="pct", source="computed")
                  for k, v in ranking.weights.items()}
        shown_codes = set(shown.index)
        fnd = [f for f in findings.rank_findings(ranking.ranked)
               if f.get("airport") in shown_codes or f["type"] == "close_call"]
        assumptions = [PEER_NOTE, "Percentiles are national; the rank is within the airports "
                       "you asked about.", NOT_RETURN, AIRSIDE_CAVEAT, ONTIME_NOTE]
        if focus in FOCUS_NOTES:
            assumptions.insert(1, FOCUS_NOTES[focus])
        return ToolResult(
            data={"focus": focus, "ranked": rows, "excluded": ranking.excluded,
                  "weights": ranking.weights},
            facts=facts, findings=fnd, assumptions=assumptions,
            data_coverage=coverage | {"low_confidence": [r["code"] for r in rows
                                                         if r["confidence"] == "low"]})

    def compare_airports(codes: list[str]) -> ToolResult:
        if err := unknown(codes):
            return err
        rows = m.loc[codes]
        facts = {}
        for c in codes:
            facts |= _facts(c, rows.loc[c].to_dict(),
                            ["name", *COMPARE_METRICS, "ontime_coverage"])
        fnd = findings.compare_findings(rows, COMPARE_METRICS)
        for f in fnd:
            if f["ratio"]:
                key = f"{f['leader']}.{f['metric']}_ratio_vs_{f['other'].lower()}"
                facts[key] = Fact(value=f["ratio"], unit="ratio", source="computed")
        ranking = scoring.score_airports(m, codes)
        for c, r in ranking.ranked.iterrows():
            facts |= _facts(c, r.to_dict(), ["classification", "confidence"])
            fnd.append({"type": "classification", "airport": c, "value": r["classification"]})
        assumptions = [
            "Congestion is measured by taxi-out (runway queue), taxi-in (waiting for a gate), "
            "NAS delay minutes per arrival, and the share of departures delayed 15+ minutes.",
            ONTIME_NOTE, AIRSIDE_CAVEAT, PEER_NOTE,
            *[f"{e['code']} has no classification: {e['reason']}." for e in ranking.excluded],
        ]
        table = [{"code": c, "name": rows.loc[c, "name"],
                  **{k: None if _missing(rows.loc[c, k]) else float(rows.loc[c, k])
                     for k in COMPARE_METRICS}} for c in codes]
        return ToolResult(data={"airports": table, "metrics": COMPARE_METRICS}, facts=facts,
                          findings=fnd, assumptions=assumptions, data_coverage=coverage)

    def airport_profile(code: str) -> ToolResult:
        if err := unknown([code]):
            return err
        row = m.loc[code]
        facts = _facts(code, row.to_dict(), PROFILE_KEYS)
        mix = metrics_mod.delay_cause_mix(store, code)
        facts |= {f"{code}.cause_{k}_share": Fact(value=v, unit="pct", source="ontime")
                  for k, v in mix.items() if not _missing(v)}
        fnd, assumptions = [], [
            f"Unmet demand is estimated as the extra seats needed to bring the load factor down "
            f"to the {config.LF_TARGET:.0%} target: max(0, passengers / target - seats). Seats "
            "are aircraft seats, not terminal capacity.",
            "Delay-cause minutes are attributed to the arrival airport.", ONTIME_NOTE]
        national = scoring.score_airports(m)
        if code in national.ranked.index:
            r = national.ranked.loc[code]
            facts |= _facts(code, _with_pctl(r), ["score", "classification", "confidence",
                                                  *PCTL_KEYS])
            facts[f"{code}.national_rank"] = Fact(value=int(r["rank"]), unit="rank",
                                                  source="computed")
            facts["PEERS.count"] = Fact(value=len(national.ranked), unit="count",
                                        source="computed")
            fnd += [f for f in findings.rank_findings(national.ranked) if f.get("airport") == code]
            assumptions.append(PEER_NOTE)
        else:
            assumptions.append(
                f"{code} is below the peer threshold, so it has no Opportunity Score.")
        if dominant := findings.dominant_cause_finding(code, mix):
            fnd.append(dominant)
        if not _missing(row["seats_gap"]) and row["seats_gap"] > 0:
            fnd.append({"type": "unmet_demand", "airport": code,
                        "load_factor_above_target": True})
        trend = metrics_mod.monthly_trend(store, code)
        trend["month"] = pd.to_datetime(trend["month"]).dt.strftime("%Y-%m")
        return ToolResult(
            data={"airport": match_rows([code])[0], "delay_cause_mix": mix,
                  "trend": trend.to_dict("records")},
            facts=facts, findings=fnd, assumptions=assumptions, data_coverage=coverage)

    def route_mix(code: str, long_haul_threshold_mi: int = config.LONG_HAUL_MI) -> ToolResult:
        if err := unknown([code]):
            return err
        rm = metrics_mod.route_mix(store, code, long_haul_threshold_mi)
        assumptions = [f"Long haul = flights of at least {long_haul_threshold_mi:,} miles "
                       "(about 5+ hours). Ask for a different threshold if you use another "
                       "definition.", ONTIME_NOTE]
        if rm["total_departures"] == 0:
            return ToolResult(data=rm, assumptions=[f"No On-Time route data for {code}.",
                                                    *assumptions], data_coverage=coverage)
        facts = {
            f"{code}.long_haul_pct": Fact(value=rm["long_haul_pct"], unit="pct", source="ontime"),
            f"{code}.long_haul_departures": Fact(value=rm["long_haul_departures"], unit="count",
                                                 source="ontime"),
            f"{code}.total_departures": Fact(value=rm["total_departures"], unit="count",
                                             source="ontime"),
            f"{code}.long_haul_threshold_mi": Fact(value=long_haul_threshold_mi, unit="miles",
                                                   source="computed"),
        }
        for low, high, label in config.DISTANCE_BUCKETS:
            key = f"share_{low}_{high}" if high is not None else f"share_{low}_plus"
            facts[f"{code}.{key}"] = Fact(value=rm["buckets"][label], unit="pct", source="ontime")
        facts |= _facts(code, m.loc[code].to_dict(), ["intl_share"])
        assumptions.append(f"For international traffic, see {code}.intl_share (T-100 share of "
                           "passengers on international flights).")
        fnd = [{"type": "route_mix", "airport": code,
                "long_haul_destinations": [d["dest"] for d in rm["long_haul_destinations"]]}]
        return ToolResult(data=rm, facts=facts, findings=fnd, assumptions=assumptions,
                          data_coverage=coverage)

    def concentration_risk(code: str) -> ToolResult:
        if err := unknown([code]):
            return err
        cs = metrics_mod.carrier_shares(store, code)
        assumptions = [
            "Concentration uses the Herfindahl-Hirschman Index (0-10,000) on operating-carrier "
            f"departure shares: under {config.HHI_LOW:,} low, {config.HHI_LOW:,}-"
            f"{config.HHI_HIGH:,} moderate, over {config.HHI_HIGH:,} high (US DOJ bands).",
            "Regional carriers flying for a major airline count separately, which can understate "
            "concentration.", ONTIME_NOTE]
        if cs["hhi"] is None:
            return ToolResult(data=cs, assumptions=[f"No On-Time carrier data for {code}.",
                                                    *assumptions], data_coverage=coverage)
        facts = {
            f"{code}.hhi": Fact(value=cs["hhi"], unit="hhi", source="computed"),
            f"{code}.concentration_band": Fact(value=cs["band"], unit="text", source="computed"),
            f"{code}.top_carrier": Fact(value=cs["top_carrier"], unit="text", source="ontime"),
            f"{code}.top_carrier_share": Fact(value=cs["top_share"], unit="pct", source="ontime"),
        }
        for carrier, share in list(cs["shares"].items())[:5]:
            facts[f"{code}.share_{carrier.lower()}"] = Fact(value=share, unit="pct",
                                                            source="ontime")
        fnd = [{"type": "concentration", "airport": code, "band": cs["band"],
                "top_carrier": cs["top_carrier"]}]
        return ToolResult(data=cs, facts=facts, findings=fnd, assumptions=assumptions,
                          data_coverage=coverage)

    def live_status(codes: list[str]) -> ToolResult:
        status = faa.status(codes)
        assumptions = ["Live status shows current conditions only and does not change the score.",
                       "Free-text NOTAM reasons are not shown (they are untrusted input)."]
        if not status["available"]:
            return ToolResult(data=status, assumptions=[
                "Live FAA status is unavailable right now; the historical analysis is unaffected.",
                *assumptions])
        facts = {}
        for code, events in status["airports"].items():
            summary = "; ".join(
                f"{e['type']} ({e['scope']})" + (f" until {e['reopen']}" if e["reopen"] else "")
                + (f", average delay {e['avg_delay_min']} min" if e["avg_delay_min"] else "")
                for e in events) or "no active delays or closures"
            facts[f"{code}.live_status"] = Fact(value=summary, unit="text", source="faa_live")
        return ToolResult(data=status, facts=facts, assumptions=assumptions)

    specs = [
        (resolve_airports, ResolveArgs,
         "Turn a place (airport name, city, metro like 'LA', US state, or region like 'New "
         "England') into IATA codes. Call this first whenever the user names places, not codes. "
         "For a region that is not built in, pass the states you believe it covers."),
        (rank_airports, RankArgs,
         "Rank airports by Opportunity Score (0-100) with bottleneck classification, confidence "
         "and rank stability. focus='terminal' for terminal expansion, 'airside' for runway/"
         "airfield projects, 'overall' otherwise."),
        (compare_airports, CompareArgs,
         "Compare 2-5 airports side by side on traffic, load factor, growth and congestion "
         "(taxi-out, taxi-in, NAS delay, delay rate), with computed comparison findings."),
        (airport_profile, CodeArgs,
         "Full profile of one airport: all KPIs, unmet demand (seats gap), delay-cause mix, "
         "national rank and classification, and the monthly trend. Use for 'why' questions."),
        (route_mix, RouteArgs,
         "Route distance mix for one airport: long-haul share (default threshold 2,400 miles), "
         "distance buckets, top destinations."),
        (concentration_risk, CodeArgs,
         "Airline concentration at one airport: carrier shares, HHI, and risk band."),
        (live_status, LiveArgs,
         "Current FAA delays and closures for up to 10 airports. Live context only."),
    ]
    return [StructuredTool.from_function(func=f, name=f.__name__, description=d, args_schema=a)
            for f, a, d in specs]
