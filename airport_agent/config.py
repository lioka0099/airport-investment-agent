"""Every business assumption lives here: weights, thresholds, models, lookup tables.

Changing a number in this file changes the behaviour of the whole system, and nothing
else in the codebase hard-codes these values.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "snapshot"
CHAT_DB_PATH = ROOT / "chat_history.sqlite"

# --- Scoring (see DESIGN.md "Scoring methodology") ---
PEER_MIN_PASSENGERS = 500_000          # last-12-month passengers to enter the peer group
WEIGHTS = {
    "demand": 0.35,    # load factor
    "growth": 0.25,    # passengers L12M vs prior 12M
    "terminal": 0.125, # taxi-in (waiting for a gate)
    "airside": 0.125,  # taxi-out + NAS delay per flight
    "scale": 0.15,     # log passengers
}
STRAIN_HIGH_PCT = 0.66                 # percentile at or above which strain counts as "high"
SENSITIVITY_FACTORS = (0.9, 1.1)       # each weight scaled by these, then renormalized

# --- Metrics ---
LF_TARGET = 0.80                       # healthy load factor used for seats_gap
LONG_HAUL_MI = 2400                    # ~5h+ flight
DISTANCE_BUCKETS = [                   # (low inclusive, high exclusive or None, label)
    (0, 500, "<500 mi"),
    (500, 1500, "500-1,500 mi"),
    (1500, 2400, "1,500-2,400 mi"),
    (2400, None, ">=2,400 mi"),
]

# --- Findings ---
SIGNIFICANCE_SIMILAR = 0.10            # relative difference below this -> "similar"
SIGNIFICANCE_LARGE = 0.50              # relative difference above this -> "large"
HHI_LOW = 1500                         # DOJ market-concentration bands
HHI_HIGH = 2500

# --- Confidence ---
CONFIDENCE_HIGH_COVERAGE = 0.5         # On-Time departures / T-100 departures
CONFIDENCE_MEDIUM_COVERAGE = 0.2
CONFIDENCE_FULL_MONTHS = 12           # T-100 months needed for "high" confidence

# --- Agent limits ---
TOOL_BUDGET = 12
MAX_GROUND_FIXES = 1
MAX_REVIEW_ROUNDS = 2
MAX_INPUT_CHARS = 2000
SESSION_MAX_MESSAGES = 30
HISTORY_MAX_TOKENS = 12_000

# --- LLM providers: (provider, model). Order = fallback order. ---
AGENT_CHAIN = [("openai", "gpt-5.4-mini"), ("anthropic", "claude-haiku-5-5")]
REVIEWER_CHAIN = [
    ("anthropic", "claude-haiku-5-5"),
    ("google", "gemini-3.6-flash"),
    ("openai", "gpt-5.4-mini"),
]
LLM_TIMEOUT_S = 45
CIRCUIT_OPEN_S = 60
RETRY_AFTER_MAX_S = 5                  # 429 with longer Retry-After -> fall back, don't wait
LLM_MAX_RETRIES = 2                    # retries per provider for transient errors
STT_MODEL = "gpt-4o-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

# --- Live FAA status ---
FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"
FAA_TIMEOUT_S = 5
FAA_CACHE_S = 300

# --- Source registry (labels and links shown in citations) ---
SOURCES = {
    "t100": {
        "label": "BTS T-100 Segment Summary by Origin Airport",
        "url": "https://data.bts.gov/Aviation/AFF-T100-Segment-Summary-By-Origin-Airport/r495-tyji",
    },
    "ontime": {
        "label": "BTS Airline On-Time Performance",
        "url": "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ",
    },
    "ourairports": {
        "label": "OurAirports airport database",
        "url": "https://ourairports.com/data/",
    },
    "faa_live": {
        "label": "FAA NAS Status (live)",
        "url": "https://nasstatus.faa.gov/",
    },
    "computed": {
        "label": "Opportunity Score methodology (computed from T-100 + On-Time)",
        "url": "",
    },
}

# --- Geography ---
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "district of columbia": "DC", "puerto rico": "PR",
}

# US Census regions and divisions, plus common informal groupings.
REGIONS = {
    "new england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "mid-atlantic": ["NJ", "NY", "PA"],
    "northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "south atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "southeast": ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
    "midwest": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "southwest": ["AZ", "NM", "OK", "TX"],
    "mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific": ["AK", "CA", "HI", "OR", "WA"],
    "west coast": ["CA", "OR", "WA"],
}

# Metro aliases. "primary" is assumed unless the user asks for the whole area.
METROS = {
    "la": {"primary": ["LAX"], "alternatives": ["BUR", "LGB", "ONT", "SNA"]},
    "los angeles": {"primary": ["LAX"], "alternatives": ["BUR", "LGB", "ONT", "SNA"]},
    "santa ana": {"primary": ["SNA"], "alternatives": []},
    "orange county": {"primary": ["SNA"], "alternatives": []},
    "sf": {"primary": ["SFO"], "alternatives": ["OAK", "SJC"]},
    "san francisco": {"primary": ["SFO"], "alternatives": ["OAK", "SJC"]},
    "bay area": {"primary": ["SFO", "OAK", "SJC"], "alternatives": []},
    "nyc": {"primary": ["JFK"], "alternatives": ["LGA", "EWR"]},
    "new york city": {"primary": ["JFK"], "alternatives": ["LGA", "EWR"]},
    "chicago": {"primary": ["ORD"], "alternatives": ["MDW"]},
    "washington dc": {"primary": ["IAD"], "alternatives": ["DCA", "BWI"]},
    "dc": {"primary": ["IAD"], "alternatives": ["DCA", "BWI"]},
    "dallas": {"primary": ["DFW"], "alternatives": ["DAL"]},
    "houston": {"primary": ["IAH"], "alternatives": ["HOU"]},
    "miami": {"primary": ["MIA"], "alternatives": ["FLL"]},
    "anchorage": {"primary": ["ANC"], "alternatives": []},
    "boston": {"primary": ["BOS"], "alternatives": []},
}
