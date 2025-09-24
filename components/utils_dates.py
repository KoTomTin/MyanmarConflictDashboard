# components/utils_dates.py
from __future__ import annotations
import calendar
import pandas as pd

def month_bounds(yyyy_mm: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    y, m = map(int, yyyy_mm.split("-"))
    first = pd.Timestamp(year=y, month=m, day=1)
    last  = pd.Timestamp(year=y, month=m, day=calendar.monthrange(y, m)[1])
    return first, last

def month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")

def month_human(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")

def month_options(df: pd.DataFrame, date_col: str = "event_date") -> list[dict]:
    """Return dropdown options from min→max month in the data."""
    dmin = pd.to_datetime(df[date_col]).min().replace(day=1)
    dmax = pd.to_datetime(df[date_col]).max().replace(day=1)
    months = pd.date_range(dmin, dmax, freq="MS")
    return [{"label": month_human(m), "value": month_label(m)} for m in months]

def default_range_earliest_latest(df: pd.DataFrame, date_col: str = "event_date") -> tuple[str,str]:
    dmin = pd.to_datetime(df[date_col]).min().replace(day=1)
    dmax = pd.to_datetime(df[date_col]).max().replace(day=1)
    return month_label(dmin), month_label(dmax)

def default_range_last_n_months(df: pd.DataFrame, n: int = 3, date_col: str = "event_date") -> tuple[str,str]:
    dmin = pd.to_datetime(df[date_col]).min().replace(day=1)
    dmax = pd.to_datetime(df[date_col]).max().replace(day=1)
    start = max(dmin, dmax - pd.DateOffset(months=n-1))
    return month_label(start), month_label(dmax)