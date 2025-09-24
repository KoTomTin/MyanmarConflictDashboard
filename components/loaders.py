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

ACLED_MAIN_CSV   = DATA / "acled_cleaned.csv"
ANOMALY_CSV      = DATA / "anomaly_detection.csv"
TIME_SERIES_CSV  = DATA / "time_series.csv"
BOUNDARIES_GEOJSON = GEO / "boundaries.geojson"

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

# ---- ACLED main ----
REQUIRED_ACLED = {
    "event_id_cnty","event_date","key_event","detailed_event",
    "actor1","primary_actor","primary_actor_type","secondary_actor",
    "admin1","admin2","admin3","Tsp_Pcode","fatalities","population_size",
}

@lru_cache(maxsize=1)
def load_acled_main(version: float | None = None) -> pd.DataFrame:
    version = version or _mtime(ACLED_MAIN_CSV)
    df = pd.read_csv(ACLED_MAIN_CSV, low_memory=False)
    missing = REQUIRED_ACLED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ACLED CSV: {sorted(missing)}")
    # Basic normalization
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["event_date"]).copy()
    # Standardize some text columns
    for col in ["key_event","detailed_event","primary_actor","secondary_actor",
                "admin1","admin2","admin3","Tsp_Pcode"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Numeric safety
    for col in ["fatalities","population_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df

# ---- Anomaly ----
REQUIRED_ANOM = {"admin1","admin2","admin3","Tsp_Pcode","map_category","flag_description"}

@lru_cache(maxsize=1)
def load_anomaly(version: float | None = None) -> pd.DataFrame:
    version = version or _mtime(ANOMALY_CSV)
    df = pd.read_csv(ANOMALY_CSV, low_memory=False)
    missing = REQUIRED_ANOM - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in anomaly CSV: {sorted(missing)}")
    for c in REQUIRED_ANOM:
        df[c] = df[c].astype(str).str.strip()
    return df

# ---- Time series ----
REQUIRED_TS = {"region", "event_date", "actual_month"}

@lru_cache(maxsize=1)
def load_time_series(version: float | None = None) -> pd.DataFrame:
    version = version or _mtime(TIME_SERIES_CSV)
    df = pd.read_csv(TIME_SERIES_CSV, low_memory=False)

    missing = REQUIRED_TS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in time_series CSV: {sorted(missing)}")

    # normalize
    df["region"] = df["region"].astype(str).str.strip()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["event_date"]).copy()

    df["actual_month"] = pd.to_numeric(df["actual_month"], errors="coerce").fillna(0.0)

    # convenience columns
    df["year"] = df["event_date"].dt.year.astype(int)
    df["month_num"] = df["event_date"].dt.month.astype(int)
    # keep an "admin1" alias for pages that expect that name
    df["admin1"] = df["region"]

    return df