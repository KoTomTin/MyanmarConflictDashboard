# components/loaders.py
#
# Each public loader delegates to an lru_cache'd private function keyed on the
# source file's mtime, so a long-running server picks up pipeline updates on
# the next request instead of serving the old frame until process restart.
from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
GEO  = ROOT / "data" / "shapes"

ACLED_MAIN_PARQUET   = DATA / "acled_cleaned.parquet"
ACTOR_LEVEL_PARQUET  = DATA / "acled_actor_level.parquet"
ALLY_PAIRS_PARQUET   = DATA / "acled_actor_ally_pairs.parquet"
MONTHLY_TSP_PARQUET  = DATA / "monthly_township.parquet"
LAST_UPDATED_FILE    = DATA / "last_updated.txt"
LAST_CHECKED_FILE    = DATA / "last_checked.json"
BOUNDARIES_GEOJSON   = GEO / "boundaries.geojson"
WEB_BOUNDARIES_GEOJSON = GEO / "boundaries_web.geojson"
NEIGHBOR_BORDERS_GEOJSON = GEO / "neighbor_borders.geojson"

def _mtime(p: Path) -> float:
    try:
        return os.path.getmtime(p)
    except FileNotFoundError:
        return 0.0

# ---- Geo ----
def geojson_source_path() -> Path:
    """The boundary file the app serves — also exposed over HTTP at /geo/ so
    choropleth traces can reference it by URL instead of embedding ~1.2 MB
    of geometry in every figure JSON."""
    return WEB_BOUNDARIES_GEOJSON if WEB_BOUNDARIES_GEOJSON.exists() else BOUNDARIES_GEOJSON


def load_geojson() -> dict:
    return _load_geojson(_mtime(geojson_source_path()))

@lru_cache(maxsize=1)
def _load_geojson(version: float) -> dict:
    with open(geojson_source_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def load_neighbor_borders() -> dict:
    return _load_neighbor_borders(_mtime(NEIGHBOR_BORDERS_GEOJSON))

@lru_cache(maxsize=1)
def _load_neighbor_borders(version: float) -> dict:
    if not NEIGHBOR_BORDERS_GEOJSON.exists():
        return {"type": "FeatureCollection", "features": []}
    with open(NEIGHBOR_BORDERS_GEOJSON, "r", encoding="utf-8") as f:
        return json.load(f)

# Low-cardinality string columns — convert to category to save ~100 MB RAM
_MAIN_CAT_COLS = [
    "disorder_type", "event_type", "sub_event_type", "key_event", "detailed_event",
    "inter1", "primary_actor_type", "inter2", "secondary_actor_type",
    "civilian_targeting", "admin1", "admin2", "admin3", "Tsp_Pcode",
]
_ACTOR_CAT_COLS  = ["type1", "type2", "Tsp_Pcode"]
_ALLY_CAT_COLS   = ["type1", "type2"]
_MONTHLY_CAT_COLS = ["Tsp_Pcode", "admin1", "key_event"]  # month excluded — used in >= / <= range comparisons

# ---- ACLED main ----
def load_acled_main() -> pd.DataFrame:
    return _load_acled_main(_mtime(ACLED_MAIN_PARQUET))

@lru_cache(maxsize=1)
def _load_acled_main(version: float) -> pd.DataFrame:
    df = pd.read_parquet(ACLED_MAIN_PARQUET)
    # Parquet preserves dtypes — event_date is already datetime
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()
    for col in ["key_event", "detailed_event", "primary_actor", "secondary_actor",
                "admin1", "admin2", "admin3", "Tsp_Pcode", "civilian_targeting"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # Missing values must stay missing — astype(str) turns NaN into the
            # literal "nan", which leaks into hover text and defeats dropna().
            df[col] = s.where(~s.isin(("nan", "None", "NaT", "")))
    if "fatalities" in df.columns:
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    # Convert low-cardinality columns to category — cuts ~65 MB RAM
    for col in _MAIN_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# ---- Actor level ----
def load_actor_level() -> pd.DataFrame:
    return _load_actor_level(_mtime(ACTOR_LEVEL_PARQUET))

@lru_cache(maxsize=1)
def _load_actor_level(version: float) -> pd.DataFrame:
    df = pd.read_parquet(ACTOR_LEVEL_PARQUET)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        # Precomputed once here — deriving it per callback cost ~300 ms/request
        df["month"] = df["event_date"].dt.to_period("M").astype(str)
    for col in _ACTOR_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# ---- Ally pairs ----
def load_ally_pairs() -> pd.DataFrame:
    return _load_ally_pairs(_mtime(ALLY_PAIRS_PARQUET))

@lru_cache(maxsize=1)
def _load_ally_pairs(version: float) -> pd.DataFrame:
    df = pd.read_parquet(ALLY_PAIRS_PARQUET)
    for col in _ALLY_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# ---- Monthly township aggregation (pre-built by pipeline, Massacres recoding already applied) ----
def load_monthly_township() -> pd.DataFrame:
    """Read pre-aggregated monthly township counts from parquet.
    Columns: Tsp_Pcode, admin1, month, key_event, events, fatalities."""
    return _load_monthly_township(_mtime(MONTHLY_TSP_PARQUET))

@lru_cache(maxsize=1)
def _load_monthly_township(version: float) -> pd.DataFrame:
    df = pd.read_parquet(MONTHLY_TSP_PARQUET)
    for col in _MONTHLY_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_last_updated() -> str:
    return _load_last_updated(_mtime(LAST_UPDATED_FILE))

@lru_cache(maxsize=1)
def _load_last_updated(version: float) -> str:
    try:
        return LAST_UPDATED_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def load_last_checked() -> dict:
    return _load_last_checked(max(_mtime(LAST_CHECKED_FILE), _mtime(LAST_UPDATED_FILE)))

@lru_cache(maxsize=1)
def _load_last_checked(version: float) -> dict:
    if LAST_CHECKED_FILE.exists():
        try:
            payload = json.loads(LAST_CHECKED_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    fallback = load_last_updated()
    if not fallback:
        return {}

    try:
        date_str = pd.to_datetime(fallback).strftime("%d %b %Y")
    except Exception:
        date_str = fallback

    return {
        "display": date_str,
        "date_display": date_str,
        "time_display": "",
        "timezone_label": "Yangon time",
        "cadence_note": "Scheduled ACLED check: daily, ~07:00 Yangon time",
        "recorded_time": False,
    }
