# components/loaders.py
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
BOUNDARIES_GEOJSON   = GEO / "boundaries.geojson"

def _mtime(p: Path) -> float:
    try:
        return os.path.getmtime(p)
    except FileNotFoundError:
        return 0.0

# ---- Geo ----
@lru_cache(maxsize=1)
def load_geojson(version: float | None = None) -> dict:
    version = version or _mtime(BOUNDARIES_GEOJSON)
    with open(BOUNDARIES_GEOJSON, "r", encoding="utf-8") as f:
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
@lru_cache(maxsize=1)
def load_acled_main(version: float | None = None) -> pd.DataFrame:
    version = version or _mtime(ACLED_MAIN_PARQUET)
    df = pd.read_parquet(ACLED_MAIN_PARQUET)
    # Parquet preserves dtypes — event_date is already datetime
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()
    for col in ["key_event", "detailed_event", "primary_actor", "secondary_actor",
                "admin1", "admin2", "admin3", "Tsp_Pcode", "civilian_targeting"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "fatalities" in df.columns:
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    # Recode mass civilian killings as "Massacres"
    if "civilian_targeting" in df.columns and "fatalities" in df.columns:
        massacre_mask = (df["civilian_targeting"] == "Yes") & (df["fatalities"] >= 5)
        df.loc[massacre_mask, "key_event"] = "Massacres"
    # Convert low-cardinality columns to category — cuts ~65 MB RAM
    for col in _MAIN_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# ---- Actor level ----
@lru_cache(maxsize=1)
def load_actor_level(version: float | None = None) -> pd.DataFrame:
    version = version or _mtime(ACTOR_LEVEL_PARQUET)
    df = pd.read_parquet(ACTOR_LEVEL_PARQUET)
    for col in _ACTOR_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# ---- Ally pairs ----
@lru_cache(maxsize=1)
def load_ally_pairs(version: float | None = None) -> pd.DataFrame:
    version = version or _mtime(ALLY_PAIRS_PARQUET)
    df = pd.read_parquet(ALLY_PAIRS_PARQUET)
    for col in _ALLY_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# ---- Monthly township aggregation (rebuilt from acled_main so Massacres recoding applies) ----
@lru_cache(maxsize=1)
def load_monthly_township(version: float | None = None) -> pd.DataFrame:
    """Aggregate acled_main (incl. Massacres recoding) into monthly township counts.
    Columns: Tsp_Pcode, month, key_event, admin1, events, fatalities."""
    version = version or _mtime(ACLED_MAIN_PARQUET)
    acled = load_acled_main()  # lru_cached — free second call
    df = acled.copy()
    df["month"] = df["event_date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby(["Tsp_Pcode", "month", "key_event", "admin1"], observed=True)
        .agg(events=("event_date", "count"), fatalities=("fatalities", "sum"))
        .reset_index()
    )
    for col in _MONTHLY_CAT_COLS:
        if col in monthly.columns:
            monthly[col] = monthly[col].astype("category")
    return monthly