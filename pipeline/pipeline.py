"""
Myanmar Conflict Dashboard — Data Pipeline
==========================================
Run this script to:
  1. Process the historical Excel file (ACLED_Data_Jan2026.xlsx)
  2. Fetch new events from the ACLED API (from Feb 2026 onwards, or since last update)
  3. Clean and recode all data (key_event, detailed_event, actors, etc.)
  4. Match Tsp_Pcode from MIMU township lookup
  5. Save three Parquet datasets (compact, type-preserving):
       data/processed/acled_cleaned.parquet     — event-level (1 row per event, ~3-4 MB)
       data/processed/acled_actor_level.parquet — slim relationship table (event_id + actor cols)
       data/processed/acled_actor_ally_pairs.parquet — slim edge table (event_id + pair cols)

  NOTE: notes column is used during processing but NOT stored (reduces file by ~22 MB).

Usage:
  python pipeline/pipeline.py                         # first-time full run
  python pipeline/pipeline.py --update-only           # only fetch new API data and append
  python pipeline/pipeline.py --export-csv            # also write CSVs to data/verify/ for inspection
  python pipeline/pipeline.py --update-only --export-csv
"""

import os
import sys
import re
import json
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[1]
OTHERS_DIR      = ROOT / "Others"
DATA_DIR        = ROOT / "data" / "processed"
EXCEL_FILE      = ROOT / "ACLED_Data_Jan2026.xlsx"
PCODE_FILE      = OTHERS_DIR / "tsp_pcode_Sep25.xlsx"
OUTPUT_PARQUET      = DATA_DIR / "acled_cleaned.parquet"
ACTOR_LEVEL_PARQUET = DATA_DIR / "acled_actor_level.parquet"
ALLY_PAIRS_PARQUET  = DATA_DIR / "acled_actor_ally_pairs.parquet"
LAST_UPDATED             = DATA_DIR / "last_updated.txt"
LAST_CHECKED             = DATA_DIR / "last_checked.json"
MONTHLY_TSP_PARQUET      = DATA_DIR / "monthly_township.parquet"
SYNC_STATE_FILE          = DATA_DIR / "acled_sync_state.json"

VERIFY_DIR = ROOT / "data" / "verify"   # CSV exports for local inspection — gitignored

DATA_DIR.mkdir(parents=True, exist_ok=True)

SYNC_OVERLAP_SECONDS = 48 * 60 * 60
YANGON_TZ = ZoneInfo("Asia/Yangon")

# ── Credentials ────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env")
EMAIL    = os.getenv("ACLED_EMAIL")
PASSWORD = os.getenv("ACLED_PASSWORD")

# ── Final columns to keep in acled_cleaned ─────────────────────────────────────
FINAL_COLS = [
    "event_id_cnty", "event_date",
    "disorder_type", "event_type", "sub_event_type",
    "key_event", "detailed_event",
    "actor1", "primary_actor", "inter1", "primary_actor_type",
    "assoc_actor_1",
    "actor2", "secondary_actor", "inter2", "secondary_actor_type",
    "assoc_actor_2",
    "civilian_targeting",
    "admin1", "admin2", "admin3", "Tsp_Pcode",
    "fatalities",
    # notes intentionally excluded — used during processing, not stored.
    # The two notes-derived boolean signals ARE stored so the recode can be
    # re-applied to the parquet without flipping notes-dependent categories.
    "notes_mentions_drone", "notes_mentions_displacement",
]

# ── Actor recoding constants ───────────────────────────────────────────────────
PDF_PATTERN = r"[Pp]eople['']s [Dd]efens[ec]e? [Ff]orc"
PSH_PATTERN = r"[Pp]yu [Ss]aw [Hh]tee"

# Regime umbrella: any actor string containing these terms → "Myanmar Military Regime"
REGIME_TERMS = {
    "Police Forces of Myanmar (2021-)",
    "Government of Myanmar (2021-)",
    "State Administration Council",
    "Military Forces of Myanmar (2021-) Border Guard Force",
    "Military Forces of Myanmar (2021-)",
    "Military Forces of Myanmar (2021-) People's Militia Force",
}

# Compiled regex for splitting multi-value assoc_actor cells
_ASSOC_SPLIT = re.compile(r"\s*;\s*|\s*,\s*")


# ══════════════════════════════════════════════════════════════════════════════
# 1. AUTHENTICATION / HTTP
# ══════════════════════════════════════════════════════════════════════════════

# ACLED's gateway started rejecting the default python-requests User-Agent from
# datacenter IPs (GitHub runners) with an nginx 415 in July 2026 — identify
# ourselves properly. The retrying session also covers transient 429/5xx.
_HTTP_HEADERS = {
    "User-Agent": "MyanmarConflictDashboard/1.0 (+https://myanmarconflictdashboard.com; ACLED data pipeline)",
    "Accept": "application/json",
}
_HTTP_SESSION: requests.Session | None = None


def _http() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        from urllib3.util.retry import Retry
        s = requests.Session()
        retry = Retry(
            total=4, backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.headers.update(_HTTP_HEADERS)
        _HTTP_SESSION = s
    return _HTTP_SESSION


def get_token():
    """Request a 24-hour OAuth Bearer token from ACLED."""
    print("  Requesting ACLED OAuth token...")
    resp = _http().post(
        "https://acleddata.com/oauth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username":   EMAIL,
            "password":   PASSWORD,
            "grant_type": "password",
            "client_id":  "acled",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token request failed ({resp.status_code}): {resp.text[:200]}")
    print("  Token received OK.")
    return resp.json()["access_token"]


def utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def read_sync_state() -> dict:
    if not SYNC_STATE_FILE.exists():
        return {}
    try:
        return json.loads(SYNC_STATE_FILE.read_text())
    except Exception:
        return {}


def write_sync_state(state: dict):
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. API FETCH — paginated
# ══════════════════════════════════════════════════════════════════════════════

def fetch_myanmar_api(
    token,
    start_date: str | None = None,
    end_date: str | None = None,
    start_timestamp: int | None = None,
) -> pd.DataFrame:
    """
    Fetch Myanmar events using either:
      - event_date BETWEEN start_date and end_date, or
      - timestamp >= start_timestamp
    Handles pagination automatically (5 000 rows per page).
    """
    all_rows = []
    offset   = 0
    page     = 1
    headers  = {"Authorization": f"Bearer {token}"}

    if start_timestamp is not None:
        print(f"  Fetching Myanmar events uploaded/updated since timestamp {start_timestamp} ...")
    else:
        print(f"  Fetching Myanmar events {start_date} → {end_date} ...")

    while True:
        params = {
            "country": "Myanmar",
            "limit": 5000,
            "offset": offset,
        }
        if start_timestamp is not None:
            params["timestamp"] = start_timestamp
        else:
            params["event_date"] = f"{start_date}|{end_date}"
            params["event_date_where"] = "BETWEEN"

        resp = _http().get(
            "https://acleddata.com/api/acled/read",
            headers=headers,
            params=params,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json().get("data", [])
        all_rows.extend(data)
        print(f"    Page {page}: {len(data)} rows (total so far: {len(all_rows)})")

        if len(data) < 5000:
            break
        offset += 5000
        page   += 1

    if not all_rows:
        print("  No new rows returned.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # API returns ISO format YYYY-MM-DD — parse explicitly to avoid dayfirst ambiguity.
    # Keep as datetime (not string) so clean_dataframe's dayfirst=True is a no-op on them.
    df["event_date"] = pd.to_datetime(df["event_date"], format="%Y-%m-%d", errors="coerce")
    today = pd.Timestamp(date.today())
    before = len(df)
    df = df[df["event_date"] <= today].copy()
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped} rows with future/invalid dates.")

    return df


def fetch_deleted_event_ids(token, start_timestamp: int) -> set[str]:
    """
    Fetch deleted Myanmar event IDs from the deleted endpoint using deleted_timestamp.
    ACLED documents this as the correct way to keep local datasets aligned with removals.
    """
    deleted_ids = set()
    offset      = 0
    page        = 1
    headers     = {"Authorization": f"Bearer {token}"}

    print(f"  Fetching deleted Myanmar event IDs since timestamp {start_timestamp} ...")

    while True:
        resp = _http().get(
            "https://acleddata.com/api/deleted/read",
            headers=headers,
            params={
                "event_id_cnty": "MMR",
                "deleted_timestamp": start_timestamp,
                "limit": 5000,
                "offset": offset,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Deleted endpoint error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json().get("data", [])
        print(f"    Deleted page {page}: {len(data)} rows")

        for row in data:
            event_id = str(row.get("event_id_cnty", "")).strip()
            if event_id:
                deleted_ids.add(event_id)

        if len(data) < 5000:
            break
        offset += 5000
        page   += 1

    return deleted_ids


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOWNSHIP PCODE MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def build_pcode_lookup() -> pd.DataFrame:
    """
    Build a district+township → Tsp_Pcode lookup table from MIMU file.
    Matching key: admin2.lower() + '|' + admin3.lower()
    Falls back to deriving the lookup from the existing cleaned parquet
    if the source xlsx is missing.
    """
    if PCODE_FILE.exists():
        pcode = pd.read_excel(PCODE_FILE)
        pcode["match_key"] = (
            pcode["District/SAZ_Name_Eng"].str.strip().str.lower()
            + "|"
            + pcode["Township_Name_Eng"].str.strip().str.lower()
        )
        return pcode[["match_key", "Tsp_Pcode"]].drop_duplicates("match_key").copy()

    # Fallback: rebuild lookup from the existing cleaned parquet
    print("  Warning: pcode xlsx not found — rebuilding lookup from existing parquet.")
    df = pd.read_parquet(OUTPUT_PARQUET, columns=["admin2", "admin3", "Tsp_Pcode"])
    df = df.dropna(subset=["admin2", "admin3", "Tsp_Pcode"])
    df["match_key"] = (
        df["admin2"].astype(str).str.strip().str.lower()
        + "|"
        + df["admin3"].astype(str).str.strip().str.lower()
    )
    return df[["match_key", "Tsp_Pcode"]].drop_duplicates("match_key").copy()


def fix_blank_admin3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard-code admin3 for events where ACLED only records city-level location.
    Also directly assign Tsp_Pcode for these cases because ACLED's admin2 is
    the city name, not the MIMU district name (e.g. 'Yangon' vs 'Yangon-West').
    """
    df = df.copy()
    if "Tsp_Pcode" not in df.columns:
        df["Tsp_Pcode"] = np.nan
    # Ensure object dtype so string assignment doesn't raise FutureWarning
    df["Tsp_Pcode"] = df["Tsp_Pcode"].astype(object)

    blank = df["admin3"].isna() | (df["admin3"].astype(str).str.strip() == "")

    # Mandalay city → Aungmyaythazan (MMR010001)
    mask_mdy = blank & (df["admin1"] == "Mandalay")
    df.loc[mask_mdy, "admin3"]    = "Aungmyaythazan"
    df.loc[mask_mdy, "Tsp_Pcode"] = "MMR010001"

    # Nay Pyi Taw → Poke Ba Thi Ri (MMR018005)
    mask_npt = blank & (df["admin1"] == "Nay Pyi Taw")
    df.loc[mask_npt, "admin3"]    = "Poke Ba Thi Ri"
    df.loc[mask_npt, "Tsp_Pcode"] = "MMR018005"

    # Yangon city → Hlaing (MMR013040)
    mask_ygn = blank & (df["admin1"] == "Yangon")
    df.loc[mask_ygn, "admin3"]    = "Hlaing"
    df.loc[mask_ygn, "Tsp_Pcode"] = "MMR013040"

    # Any remaining blanks
    still_blank = df["admin3"].isna() | (df["admin3"].astype(str).str.strip() == "")
    df.loc[still_blank, "admin3"] = "Unknown"
    return df


def add_tsp_pcode(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Tsp_Pcode into df via district+township match key.
    Rows pre-assigned by fix_blank_admin3 are skipped.
    """
    df = fix_blank_admin3(df)
    # Only match rows that don't already have a Tsp_Pcode
    needs_match = df["Tsp_Pcode"].isna()
    df["match_key"] = (
        df["admin2"].astype(str).str.strip().str.lower()
        + "|"
        + df["admin3"].astype(str).str.strip().str.lower()
    )
    matched = df.loc[needs_match, ["match_key"]].merge(lookup, on="match_key", how="left")
    df.loc[needs_match, "Tsp_Pcode"] = matched["Tsp_Pcode"].values
    df = df.drop(columns=["match_key"])
    n_unmatched = df["Tsp_Pcode"].isna().sum()
    if n_unmatched:
        print(f"  ⚠  {n_unmatched} rows could not be matched to a Tsp_Pcode.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. KEY EVENT RECODING
# ══════════════════════════════════════════════════════════════════════════════

def _notes_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return the two notes-derived boolean signals (drone mention, displacement
    mention) used by the key_event / detailed_event recodes.

    The notes column is not persisted to parquet, so re-running the recode on
    stored data used to silently flip notes-dependent categories (state-forces
    Drone Strike → Air Strike, Displacement → Others) on every --update-only
    run. Resolution order:
      1. notes present (fresh API/Excel data)      → derive from notes
      2. persisted flag columns present            → reuse them
      3. legacy parquet without flags              → reconstruct from
         detailed_event, which was derived from notes at full-rebuild time
         and is never re-derived without them.
    """
    if "notes" in df.columns:
        notes = df["notes"].astype(str).str.lower()
        drone = notes.str.contains("drone", na=False)
        disp  = notes.str.contains("displacement", na=False)
    elif "notes_mentions_drone" in df.columns and "notes_mentions_displacement" in df.columns:
        drone = df["notes_mentions_drone"].fillna(False).astype(bool)
        disp  = df["notes_mentions_displacement"].fillna(False).astype(bool)
    else:
        de = (
            df["detailed_event"].astype(str)
            if "detailed_event" in df.columns
            else pd.Series("", index=df.index)
        )
        drone = de.eq("Drone strike")
        disp  = de.eq("Displacement")
    return drone, disp


def create_key_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode key_event with 12 categories.
    Requires: event_type, sub_event_type, inter1, and either notes or the
    persisted notes_mentions_* flag columns (see _notes_flags).

    Priority order (np.select first-match):
      1. Arrests
      2. Looting/property destruction
      3. Air Strike        (Air/drone strike, inter1 == State forces, no drone keyword)
      4. Drone Strike      (Air/drone strike, non-state actor or 'drone' in notes)
      5. Shelling/Artillery
      6. IED/Mine
      7. Armed Clash       (all remaining Battles / Explosions ground combat)
      8. Attack on Civilians (Violence against civilians)
      9. Protests
     10. Displacement      (notes heuristic)
     → Others (default)

    Post-recode override applied in clean_dataframe() and main loop:
      civilian_targeting == 'Yes' & fatalities >= 5  →  Massacres
    """
    df    = df.copy()
    et    = df["event_type"].astype(str).str.strip()
    set_  = df["sub_event_type"].astype(str).str.strip()
    i1    = df["inter1"].astype(str).str.strip()  if "inter1" in df.columns else pd.Series("State forces", index=df.index)

    drone_mention, disp_mention = _notes_flags(df)
    df["notes_mentions_drone"]        = drone_mention
    df["notes_mentions_displacement"] = disp_mention

    is_air_drone  = set_ == "Air/drone strike"
    is_drone_flag = drone_mention | ~i1.str.casefold().eq("state forces")
    is_combat_et  = et.isin(["Battles", "Explosions/Remote violence"])

    SHELLING_SUBS = frozenset(["Shelling/artillery/missile attack"])
    IED_SUBS      = frozenset([
        "Remote explosive/landmine/IED",
        "Grenade",
        "Suicide bomb",
        "Landmine",
        "IED",
    ])

    conditions = [
        set_ == "Arrests",
        set_ == "Looting/property destruction",
        is_air_drone & ~is_drone_flag,                             # Air Strike
        is_air_drone &  is_drone_flag,                             # Drone Strike
        is_combat_et & ~is_air_drone & set_.isin(SHELLING_SUBS),  # Shelling/Artillery
        is_combat_et & ~is_air_drone & set_.isin(IED_SUBS),       # IED/Mine
        is_combat_et & ~is_air_drone,                              # Armed Clash (all other ground)
        et == "Violence against civilians",                        # Attack on Civilians
        et == "Protests",
        (set_ == "Other") & disp_mention,
    ]
    choices = [
        "Arrests",
        "Looting/property destruction",
        "Air Strike",
        "Drone Strike",
        "Shelling/Artillery",
        "IED/Mine",
        "Armed Clash",
        "Attack on Civilians",
        "Protests",
        "Displacement",
    ]
    df["key_event"] = np.select(conditions, choices, default="Others")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. DETAILED EVENT RECODING
# ══════════════════════════════════════════════════════════════════════════════

def create_detailed_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set detailed_event (sub_event_type granularity).
    key_event is already finalised by create_key_event — this function does NOT modify it.
    """
    df = df.copy()
    # Resolve the notes-derived flags BEFORE overwriting detailed_event —
    # the legacy fallback in _notes_flags reads the stored detailed_event.
    drone_mention, disp_mention = _notes_flags(df)

    df["detailed_event"] = df["sub_event_type"].astype(str)

    i1 = df["inter1"].astype(str).str.strip() if "inter1" in df.columns else pd.Series("State forces", index=df.index)

    is_air   = df["sub_event_type"].astype(str).str.strip() == "Air/drone strike"
    is_drone = drone_mention | ~i1.str.casefold().eq("state forces")

    df.loc[is_air & is_drone,  "detailed_event"] = "Drone strike"
    df.loc[is_air & ~is_drone, "detailed_event"] = "Air strike"

    is_other = df["sub_event_type"].astype(str).str.strip() == "Other"
    df.loc[is_other & disp_mention, "detailed_event"] = "Displacement"

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. ACTOR RECODING
# ══════════════════════════════════════════════════════════════════════════════

def _recode_type(series: pd.Series) -> pd.Series:
    """Recode inter1/inter2 to new type labels."""
    s = series.astype(str).str.strip()
    out = s.copy()
    out = out.where(~s.str.contains(PDF_PATTERN, regex=True), "People's Defense Force")
    out = out.where(~s.str.contains(PSH_PATTERN, regex=True), "Pyu Saw Htee")
    out = out.where(~s.str.casefold().eq("rebel group"), out).where(
              ~s.str.casefold().eq("rebel group"), "ERO")
    out = out.where(~s.str.casefold().eq("state forces"), out).where(
              ~s.str.casefold().eq("state forces"), "Myanmar Military Regime")
    return out


def _recode_actor(actor_series: pd.Series, type_series: pd.Series) -> pd.Series:
    """Recode actor names to short forms based on their recoded type."""
    a = actor_series.astype(str).str.strip()
    t = type_series.copy()
    out = a.copy()
    # Myanmar Military Regime → collapse all regime actors to umbrella label
    out = out.where(~t.str.contains("Myanmar Military Regime", na=False), "Myanmar Military Regime")
    # ERO → keep only text before first ":" (short form, e.g. "KIO/KIA")
    ero_mask = t == "ERO"
    out = out.where(~ero_mask, a.str.split(":", n=1).str[0].str.strip())
    # PDF → collapse to clean label
    pdf_mask = t == "People's Defense Force"
    out = out.where(~pdf_mask, "People's Defense Force")
    return out


def _normalize_assoc_cell(cell: str) -> str:
    """
    Normalize a multi-value assoc_actor cell to short-form actor names.
    Splits on ';' or ',', applies regime/PDF/PSH checks, keeps before ':' for ERO,
    deduplicates, and rejoins with '; '.
    """
    if pd.isna(cell) or not str(cell).strip():
        return ""
    tokens = [t.strip() for t in _ASSOC_SPLIT.split(str(cell)) if t.strip()]
    result = []
    seen   = set()
    for t in tokens:
        # Regime umbrella check (substring match against known actor strings)
        for term in REGIME_TERMS:
            if term in t:
                t = "Myanmar Military Regime"
                break
        # PDF / PSH pattern check
        if re.search(PDF_PATTERN, t):
            t = "People's Defense Force"
        elif re.search(PSH_PATTERN, t):
            t = "Pyu Saw Htee"
        else:
            # Short form: keep text before ":" for ERO-style names
            t = t.split(":", 1)[0].strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return "; ".join(result)


def create_actor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates actor recoding from data_processing.ipynb.
    Creates: primary_actor, primary_actor_type, secondary_actor, secondary_actor_type
    Also normalizes assoc_actor_1 and assoc_actor_2 to short-form names.
    """
    df = df.copy()

    # Step 1: recode types first (based on original inter1/inter2)
    df["primary_actor_type"]   = _recode_type(df["inter1"])
    df["secondary_actor_type"] = _recode_type(df["inter2"])

    # Step 2: handle PDF/PSH in actor name itself (overrides type if needed)
    a1 = df["actor1"].astype(str).str.strip()
    a2 = df["actor2"].astype(str).str.strip()

    df.loc[a1.str.contains(PDF_PATTERN, regex=True), "primary_actor_type"]   = "People's Defense Force"
    df.loc[a1.str.contains(PSH_PATTERN, regex=True), "primary_actor_type"]   = "Pyu Saw Htee"
    df.loc[a2.str.contains(PDF_PATTERN, regex=True), "secondary_actor_type"] = "People's Defense Force"
    df.loc[a2.str.contains(PSH_PATTERN, regex=True), "secondary_actor_type"] = "Pyu Saw Htee"

    # Step 3: recode actor names to short forms based on final types
    df["primary_actor"]   = _recode_actor(df["actor1"], df["primary_actor_type"])
    df["secondary_actor"] = _recode_actor(df["actor2"], df["secondary_actor_type"])

    # Step 4: normalize assoc_actor columns to short forms
    if "assoc_actor_1" in df.columns:
        df["assoc_actor_1"] = df["assoc_actor_1"].astype(str).apply(_normalize_assoc_cell)
    if "assoc_actor_2" in df.columns:
        df["assoc_actor_2"] = df["assoc_actor_2"].astype(str).apply(_normalize_assoc_cell)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. CIVILIAN TARGETING
# ══════════════════════════════════════════════════════════════════════════════

def recode_civilian_targeting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["civilian_targeting"] = df["civilian_targeting"].astype(str).str.strip()
    df["civilian_targeting"] = df["civilian_targeting"].apply(
        lambda x: "Yes" if x == "Civilian targeting" else "No"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. FULL CLEAN PIPELINE — applied to any raw dataframe
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataframe(df: pd.DataFrame, pcode_lookup: pd.DataFrame) -> pd.DataFrame:
    """Apply all recoding and return the final cleaned dataframe."""
    print("  Parsing dates...")
    df["event_date"] = pd.to_datetime(df["event_date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()

    # Drop future-dated rows (ACLED API quirk can produce DD/MM-swapped dates)
    today = pd.Timestamp(date.today())
    before = len(df)
    df = df[df["event_date"] <= today].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with future dates.")

    # Strip admin1 sub-region suffix if present (e.g. "Bago-East" → "Bago")
    df["admin1"] = df["admin1"].astype(str).str.split("-").str[0].str.strip()

    print("  Matching Tsp_Pcode...")
    df = add_tsp_pcode(df, pcode_lookup)

    print("  Recoding key_event...")
    df = create_key_event(df)

    print("  Recoding detailed_event...")
    df = create_detailed_event(df)

    print("  Recoding actors...")
    df = create_actor_columns(df)

    print("  Recoding civilian_targeting...")
    df = recode_civilian_targeting(df)

    print("  Normalising fatalities...")
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)

    print("  Recoding massacres...")
    massacre_mask = (df["civilian_targeting"] == "Yes") & (df["fatalities"] >= 5)
    df.loc[massacre_mask, "key_event"] = "Massacres"
    if massacre_mask.sum():
        print(f"    {massacre_mask.sum():,} mass-casualty civilian attacks recoded as Massacres.")

    # Keep only required columns that actually exist
    keep = [c for c in FINAL_COLS if c in df.columns]
    df = df[keep].copy()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. BUILD ACTOR-LEVEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_actor_level(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Build actor-level dataset (1 row per actor per event).
    Filters to Armed conflict events only.
    Uses primary_actor / assoc_actor_1 for the dashboard's primary side,
    and secondary_actor / assoc_actor_2 for the dashboard's secondary side.

    Note: these are dashboard analytical side assignments derived from the
    recoded actor fields. They should not be interpreted as a native ACLED
    aggressor/victim indicator.

    Output columns (slim — join with acled_cleaned on event_id_cnty for event properties):
        event_id_cnty, actor_name, type1, type2, allies
    """
    # Filter to combat events (includes Massacres — mass-casualty civilian attacks by armed actors)
    COMBAT_EVENTS = {"Armed Clash", "Shelling/Artillery", "IED/Mine", "Air Strike", "Drone Strike", "Massacres"}
    df = df_cleaned[df_cleaned["key_event"].isin(COMBAT_EVENTS)].copy()

    # Only keep event_id_cnty as the join key — event properties live in acled_cleaned
    keep_cols = ["event_id_cnty"] if "event_id_cnty" in df.columns else []

    def split_assoc(cell):
        if pd.isna(cell) or not str(cell).strip():
            return []
        return [p.strip() for p in _ASSOC_SPLIT.split(str(cell)) if p.strip()]

    df["assoc1_list"] = df["assoc_actor_1"].apply(split_assoc) if "assoc_actor_1" in df.columns else [[] for _ in range(len(df))]
    df["assoc2_list"] = df["assoc_actor_2"].apply(split_assoc) if "assoc_actor_2" in df.columns else [[] for _ in range(len(df))]

    def _valid_actor(name):
        return pd.notna(name) and str(name).strip() not in ("", "nan", "NaN")

    # ── Primary side: primary_actor (main) + assoc_actor_1 (associated) ─────────
    off_main = df[keep_cols + ["primary_actor"]].rename(columns={"primary_actor": "actor_name"}).copy()
    off_main = off_main[off_main["actor_name"].apply(_valid_actor)]
    off_main["type1"] = "main"
    off_main["type2"] = "offend"

    off_assoc = df[keep_cols + ["assoc1_list"]].explode("assoc1_list").rename(columns={"assoc1_list": "actor_name"})
    off_assoc = off_assoc[off_assoc["actor_name"].apply(_valid_actor)]
    off_assoc["type1"] = "associated"
    off_assoc["type2"] = "offend"

    # ── Secondary side: secondary_actor (main) + assoc_actor_2 (associated) ─────
    def_main = df[keep_cols + ["secondary_actor"]].rename(columns={"secondary_actor": "actor_name"}).copy()
    def_main = def_main[def_main["actor_name"].apply(_valid_actor)]
    def_main["type1"] = "main"
    def_main["type2"] = "being_offended"

    def_assoc = df[keep_cols + ["assoc2_list"]].explode("assoc2_list").rename(columns={"assoc2_list": "actor_name"})
    def_assoc = def_assoc[def_assoc["actor_name"].apply(_valid_actor)]
    def_assoc["type1"] = "associated"
    def_assoc["type2"] = "being_offended"

    # ── Combine ──────────────────────────────────────────────────────────────────
    actor_level = pd.concat([off_main, off_assoc, def_main, def_assoc], ignore_index=True)
    actor_level["actor_name"] = actor_level["actor_name"].astype(str).str.strip()
    actor_level = actor_level.drop_duplicates(
        subset=keep_cols + ["actor_name", "type1", "type2"]
    ).reset_index(drop=True)

    # ── Primary side takes precedence: if an actor appears on both sides of the
    # same event, keep only their primary-side record to avoid double-counting.
    offend_keys = (
        actor_level[actor_level["type2"] == "offend"][["event_id_cnty", "actor_name"]]
        .drop_duplicates()
        .assign(_has_offend=True)
    )
    actor_level = actor_level.merge(offend_keys, on=["event_id_cnty", "actor_name"], how="left")
    actor_level = actor_level[
        ~((actor_level["type2"] == "being_offended") & (actor_level["_has_offend"] == True))
    ].drop(columns=["_has_offend"]).reset_index(drop=True)

    # ── Compute allies (actors on the same side in the same event) ───────────────
    # Group by event + side only (faster than all keep_cols)
    actors_by_side = (
        actor_level.groupby(["event_id_cnty", "type2"], dropna=False)["actor_name"]
        .apply(lambda s: sorted(set(x for x in s if isinstance(x, str) and x.strip())))
        .reset_index(name="side_actor_list")
    )
    actor_level = actor_level.merge(actors_by_side, on=["event_id_cnty", "type2"], how="left")

    actor_level["allies"] = actor_level.apply(
        lambda row: "; ".join(
            a for a in (row["side_actor_list"] if isinstance(row["side_actor_list"], list) else [])
            if a != row["actor_name"]
        ),
        axis=1,
    )
    actor_level = actor_level.drop(columns=["side_actor_list"], errors="ignore")

    # Join Tsp_Pcode + event_date from df_cleaned so dashboard skips a 90K-row merge at query time
    event_info = (
        df_cleaned[["event_id_cnty", "Tsp_Pcode", "event_date"]]
        .drop_duplicates("event_id_cnty")
    )
    actor_level = actor_level.merge(event_info, on="event_id_cnty", how="left")

    out_cols = ["event_id_cnty", "actor_name", "type1", "type2", "allies", "Tsp_Pcode", "event_date"]
    return actor_level[[c for c in out_cols if c in actor_level.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# 10. BUILD ACTOR-ALLY PAIRS DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_actor_ally_pairs(actor_level: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the allies column into individual actor–ally pair rows.
    Output (slim): event_id_cnty, actor_name, ally_name, type1, type2
    Event properties (date, region, etc.) live in acled_cleaned — join on event_id_cnty.
    """
    keep = [c for c in [
        "event_id_cnty", "actor_name", "type1", "type2",
    ] if c in actor_level.columns]

    pairs = actor_level[keep + ["allies"]].copy()
    pairs["ally_list"] = pairs["allies"].apply(
        lambda x: [a.strip() for a in str(x).split(";") if a.strip()] if x else []
    )
    pairs = pairs.explode("ally_list").rename(columns={"ally_list": "ally_name"})
    pairs = pairs[
        pairs["ally_name"].notna() & (pairs["ally_name"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    pairs = pairs.drop(columns=["allies"], errors="ignore")

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 11. LAST UPDATED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_last_updated() -> str | None:
    if LAST_UPDATED.exists():
        return LAST_UPDATED.read_text().strip()
    return None


def write_last_updated(d: str):
    LAST_UPDATED.write_text(d)


def write_last_checked(checked_at_utc: datetime):
    checked_utc = checked_at_utc.astimezone(timezone.utc)
    checked_yangon = checked_utc.astimezone(YANGON_TZ)
    payload = {
        "last_checked_utc": checked_utc.isoformat(),
        "last_checked_yangon": checked_yangon.isoformat(),
        "display": checked_yangon.strftime("%d %b %Y · %H:%M Yangon"),
        "date_display": checked_yangon.strftime("%d %b %Y"),
        "time_display": checked_yangon.strftime("%H:%M"),
        "timezone_label": "Yangon time",
        "cadence_note": "Scheduled ACLED check: daily, ~07:00 Yangon time",
        "recorded_time": True,
    }
    LAST_CHECKED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print()
    print("=" * 55)
    print("  Myanmar Conflict Dashboard — Data Summary")
    print("=" * 55)
    print(f"  Total events     : {len(df):,}")
    print(f"  Date range       : {df['event_date'].min().date()} → {df['event_date'].max().date()}")
    print(f"  Townships        : {df['Tsp_Pcode'].nunique()} unique Tsp_Pcodes")
    print(f"  Unmatched pcodes : {df['Tsp_Pcode'].isna().sum():,}")
    print()
    print("  key_event breakdown:")
    for k, v in df["key_event"].value_counts().items():
        print(f"    {k:<35} {v:>6,}")
    print()
    print("  Top 5 primary actors:")
    for k, v in df["primary_actor"].value_counts().head(5).items():
        print(f"    {k:<35} {v:>6,}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(update_only: bool = False, export_csv: bool = False):
    pcode_lookup = build_pcode_lookup()
    today_str    = date.today().isoformat()
    sync_state   = read_sync_state()
    sync_started_at = utc_now_ts()
    sync_completed = False

    if not update_only:
        # ── FULL RUN: process historical Excel ─────────────────────────────
        print("\n[1/4] Processing historical Excel file...")
        print(f"  Reading {EXCEL_FILE.name}...")
        raw = pd.read_excel(EXCEL_FILE)
        # Drop empty unnamed columns that Excel sometimes adds
        raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
        print(f"  Loaded {len(raw):,} rows.")

        df_hist = clean_dataframe(raw, pcode_lookup)
        print(f"  Cleaned: {len(df_hist):,} rows.")

        # ── API TOP-UP ─────────────────────────────────────────────────────
        print("\n[2/4] Fetching new data from ACLED API...")
        # Excel ends 2026-01-31; start API from 2026-02-01
        api_start = "2026-02-01"
        api_end   = today_str

        try:
            token    = get_token()
            df_api   = fetch_myanmar_api(token, api_start, api_end)
            if not df_api.empty:
                df_api_clean = clean_dataframe(df_api, pcode_lookup)
                print(f"  API rows cleaned: {len(df_api_clean):,}")
                df_all = pd.concat([df_hist, df_api_clean], ignore_index=True)
            else:
                df_all = df_hist
            sync_completed = True
        except Exception as e:
            print(f"  ⚠  API fetch failed: {e}")
            print("  Continuing with historical data only.")
            df_all = df_hist

        # Deduplicate
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["event_id_cnty"], keep="last")
        print(f"  After dedup: {len(df_all):,} rows (removed {before - len(df_all)})")

    else:
        # ── UPDATE ONLY: load existing CSV and append new API data ─────────
        if not OUTPUT_PARQUET.exists():
            print("No existing dataset found. Please run without --update-only first.")
            sys.exit(1)

        print("\n[1/4] Loading existing cleaned dataset...")
        df_all = pd.read_parquet(OUTPUT_PARQUET)
        # Parquet preserves datetime dtype — no re-parsing needed
        print(f"  Loaded {len(df_all):,} rows. Latest: {df_all['event_date'].max().date()}")

        # Re-apply key_event recode so taxonomy changes take effect without a
        # full re-run.  Notes column is absent (intentionally not stored), so
        # drone detection falls back to inter1 only — which is the primary
        # signal anyway.
        print("  Re-applying key_event recode to existing dataset...")
        df_all = create_key_event(df_all)

        # Event-date based sync. The earlier timestamp= filter silently stopped
        # returning new rows in early April 2026 — three consecutive weekly runs
        # produced identical row_count and latest_event_date.  Switching to the
        # date-range query (the same path the full rebuild uses) is reliable.
        # We refetch the last 14 days of event_dates so retroactive ACLED edits
        # to recent events are captured.  Deduplication on event_id_cnty handles
        # the resulting overlap.
        latest_event_dt = pd.to_datetime(df_all["event_date"].max()).date()
        fetch_from = latest_event_dt - timedelta(days=14)
        fetch_to = date.today()
        # Track the wall-clock start so the deleted-events query (which still
        # uses Unix timestamps and works correctly) keeps a consistent cursor.
        sync_from_ts = int(
            pd.Timestamp(latest_event_dt - timedelta(days=14))
            .tz_localize("UTC").timestamp()
        )

        print(f"\n[2/4] Fetching ACLED events with event_date {fetch_from} → {fetch_to} ...")
        try:
            token  = get_token()
            df_api = fetch_myanmar_api(token, start_date=fetch_from.isoformat(), end_date=fetch_to.isoformat())
            if not df_api.empty:
                df_api_clean = clean_dataframe(df_api, pcode_lookup)
                df_all = pd.concat([df_all, df_api_clean], ignore_index=True)
                before = len(df_all)
                df_all = df_all.drop_duplicates(subset=["event_id_cnty"], keep="last")
                duplicates_removed = before - len(df_all)
                net_new = len(df_api_clean) - duplicates_removed
                print(f"  API rows fetched: {len(df_api_clean):,}  ·  Net new after dedup: {net_new:,}")
            else:
                print("  No events returned in the date window.")

            deleted_ids = fetch_deleted_event_ids(token, sync_from_ts)
            if deleted_ids:
                before = len(df_all)
                df_all = df_all[~df_all["event_id_cnty"].astype(str).isin(deleted_ids)].copy()
                print(f"  Deleted rows removed: {before - len(df_all):,}")
            else:
                print("  No deleted Myanmar event IDs found in the sync window.")
            sync_completed = True
        except Exception as e:
            # Fail loudly: exiting 0 here made the GitHub Actions run look
            # green while no data landed — the pipeline was silently stale
            # for weeks twice (Apr 2026, May–Jul 2026). Nothing has been
            # saved yet, so aborting leaves the datasets untouched.
            print(f"  ✗  API sync failed: {e}")
            sys.exit(1)

    # Sort by date
    df_all["event_date"] = pd.to_datetime(df_all["event_date"])
    df_all = df_all.sort_values("event_date").reset_index(drop=True)

    # Apply Massacres recode to full dataset (idempotent — handles historical rows
    # loaded from parquet that predate this recoding step in clean_dataframe).
    if "civilian_targeting" in df_all.columns and "fatalities" in df_all.columns:
        massacre_mask = (df_all["civilian_targeting"] == "Yes") & (df_all["fatalities"] >= 5)
        n_recoded = massacre_mask.sum()
        df_all.loc[massacre_mask, "key_event"] = "Massacres"
        if n_recoded:
            print(f"  Applied Massacres recode to {n_recoded:,} events in full dataset.")

    # ── SAVE acled_cleaned ──────────────────────────────────────────────────
    print(f"\n[3/4] Saving acled_cleaned.parquet...")
    df_all.to_parquet(OUTPUT_PARQUET, index=False)
    write_last_updated(today_str)
    size_mb = OUTPUT_PARQUET.stat().st_size / 1e6
    print(f"  Saved {len(df_all):,} rows → {OUTPUT_PARQUET.name} ({size_mb:.1f} MB)")

    # ── BUILD & SAVE actor-level datasets ───────────────────────────────────
    print(f"\n[4/4] Building actor-level and pre-aggregated datasets...")
    df_actor = build_actor_level(df_all)
    df_actor.to_parquet(ACTOR_LEVEL_PARQUET, index=False)
    size_mb = ACTOR_LEVEL_PARQUET.stat().st_size / 1e6
    print(f"  actor_level      : {len(df_actor):,} rows → {ACTOR_LEVEL_PARQUET.name} ({size_mb:.1f} MB)")

    df_pairs = build_actor_ally_pairs(df_actor)
    df_pairs.to_parquet(ALLY_PAIRS_PARQUET, index=False)
    size_mb = ALLY_PAIRS_PARQUET.stat().st_size / 1e6
    print(f"  actor_ally_pairs : {len(df_pairs):,} rows → {ALLY_PAIRS_PARQUET.name} ({size_mb:.1f} MB)")

    # Pre-aggregate monthly township counts (avoids 90K-row groupby in dashboard)
    df_monthly = df_all.copy()
    df_monthly["month"] = df_monthly["event_date"].dt.to_period("M").astype(str)
    monthly_tsp = (
        df_monthly.groupby(["Tsp_Pcode", "admin1", "month", "key_event"])
        .agg(events=("event_id_cnty", "count"), fatalities=("fatalities", "sum"))
        .reset_index()
    )
    monthly_tsp.to_parquet(MONTHLY_TSP_PARQUET, index=False)
    size_mb = MONTHLY_TSP_PARQUET.stat().st_size / 1e6
    print(f"  monthly_township : {len(monthly_tsp):,} rows → {MONTHLY_TSP_PARQUET.name} ({size_mb:.1f} MB)")

    print(f"  last_updated.txt → {today_str}")

    # Persist sync cursor after successful incremental syncs or any full rebuild.
    if sync_completed:
        checked_at_utc = datetime.now(timezone.utc)
        write_last_checked(checked_at_utc)
        write_sync_state({
            "last_sync_timestamp": sync_started_at,
            "last_sync_utc": datetime.fromtimestamp(sync_started_at, tz=timezone.utc).isoformat(),
            "latest_event_date": str(df_all["event_date"].max().date()),
            "row_count": int(len(df_all)),
        })
        print(f"  last_checked.json → {checked_at_utc.astimezone(YANGON_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"  acled_sync_state.json → {sync_started_at}")

    print_summary(df_all)

    if export_csv:
        VERIFY_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[CSV export] Writing to {VERIFY_DIR} ...")
        df_all.to_csv(VERIFY_DIR / "acled_cleaned.csv", index=False)
        df_actor.to_csv(VERIFY_DIR / "acled_actor_level.csv", index=False)
        df_pairs.to_csv(VERIFY_DIR / "acled_actor_ally_pairs.csv", index=False)
        print(f"  acled_cleaned.csv        ({(VERIFY_DIR / 'acled_cleaned.csv').stat().st_size / 1e6:.1f} MB)")
        print(f"  acled_actor_level.csv    ({(VERIFY_DIR / 'acled_actor_level.csv').stat().st_size / 1e6:.1f} MB)")
        print(f"  acled_actor_ally_pairs.csv ({(VERIFY_DIR / 'acled_actor_ally_pairs.csv').stat().st_size / 1e6:.1f} MB)")
        print(f"  → Open these in Excel / Google Sheets to verify. Do NOT commit them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Myanmar Conflict Dashboard Data Pipeline")
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Skip historical Excel; only fetch new API data and append",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also write CSV copies to data/verify/ for local inspection (gitignored)",
    )
    args = parser.parse_args()
    main(update_only=args.update_only, export_csv=args.export_csv)
