# components/tiles.py  (REPLACEMENT)
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.colors import SEQUENTIAL_BLUES_ZERO_GREY, SEQUENTIAL_ORANGE

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _month_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """MS (month starts) inclusive window."""
    s = pd.Timestamp(start.year, start.month, 1)
    e = pd.Timestamp(end.year, end.month, 1)
    return pd.date_range(s, e, freq="MS")

def _pivot_region(df: pd.DataFrame, region: str, start: pd.Timestamp, end: pd.Timestamp):
    """
    Build a (years × 12) matrix Z of monthly counts for a region within [start,end],
    using actual_month directly (no aggregation beyond pivot).
    Returns (Z, years, month_labels).
    """
    sub = df[(df["region"] == region) & (df["event_date"] >= start) & (df["event_date"] <= end)].copy()
    months = _month_series(start, end)
    years = sorted(months.year.unique().tolist())
    if sub.empty:
        Z = np.full((len(years), 12), np.nan)
        # mask outside partial first/last year
        if years:
            if start.year == years[0]:
                Z[0, :start.month-1] = np.nan
            if end.year == years[-1]:
                Z[-1, end.month:] = np.nan
        return Z, years, MONTH_ABBR

    sub["y"] = sub["event_date"].dt.year
    sub["m"] = sub["event_date"].dt.month
    # Pivot directly; if duplicates exist, take the first value (no summation)
    agg = sub.pivot_table(index="y", columns="m", values="actual_month", aggfunc="first")

    # ensure full year×month grid
    for m in range(1,13):
        if m not in agg.columns:
            agg[m] = 0
    agg = agg.reindex(index=years, columns=range(1,13), fill_value=0)
    Z = agg.to_numpy(dtype=float)

    # set values outside partial first/last years to NaN (not drawn)
    if years:
        if start.year == years[0]:
            Z[0, :start.month-1] = np.nan
        if end.year == years[-1]:
            Z[-1, end.month:] = np.nan

    return Z, years, MONTH_ABBR

def _forecast_mask(df_all: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, years: list[int]) -> np.ndarray:
    """
    Highlight the LAST up-to-3 unique months present in the whole dataset (dataset includes
    actual + up to next 3 months). We mark those months (if they fall within [start,end]).
    """
    mask = np.zeros((len(years), 12), dtype=bool)
    # last up-to-3 months in the entire dataset
    all_ms = pd.to_datetime(df_all["event_date"]).dt.to_period("M").unique()
    if len(all_ms) == 0:
        return mask
    all_ms = sorted(all_ms)
    last_three = all_ms[-3:]  # already handles <3 safely
    forecast_ms = [pd.Timestamp(p.start_time) for p in last_three]  # month starts

    for dt in forecast_ms:
        if dt < pd.Timestamp(start.year, start.month, 1) or dt > pd.Timestamp(end.year, end.month, 1):
            continue
        y = dt.year
        m = dt.month
        if y in years:
            r = years.index(y)
            mask[r, m-1] = True
    return mask

def build_region_tiles(
    df_ts: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    region_order: list[str],
    ncols: int = 3
) -> go.Figure:
    """
    df_ts: columns = region, event_date (datetime), actual_month (float)
    start/end: date bounds
    region_order: list of regions in desired display order
    ncols: columns in the tiles grid (default 3)
    """
    n = len(region_order)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=region_order,
        # Tighter spacing to fit 15 tiles comfortably
        vertical_spacing=0.035, horizontal_spacing=0.04
    )

    # Compute a global z-range for consistent colors across regions
    in_range = df_ts[(df_ts["event_date"] >= start) & (df_ts["event_date"] <= end)]
    global_max = float(np.nanmax(in_range["actual_month"])) if len(in_range) else 0.0
    global_max = global_max if np.isfinite(global_max) and global_max > 0 else 1.0

    for idx, region in enumerate(region_order, start=1):
        r = (idx - 1) // ncols + 1
        c = (idx - 1) % ncols + 1

        Z, years, months = _pivot_region(df_ts, region, start, end)
        # Build explicit hover text for robust rendering
        rows = len(years)
        cols = 12
        text_actual = [[""] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = Z[i, j]
                if np.isfinite(val):
                    text_actual[i][j] = f"{months[j]} {years[i]}<br>Actual count: {int(round(val))}"

        # main: monthly counts (blues), consistent color range across panels
        zmin = 0
        zmax = global_max
        fig.add_trace(
            go.Heatmap(
                z=Z, x=months, y=years,
                colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="", len=0.5),
                showscale=(idx == 1),
                text=text_actual,
                hovertemplate="%{text}<extra></extra>",
                hoverongaps=False,
            ),
            row=r, col=c
        )

        # overlay: forecast (last up-to-3 months in dataset), in orange
        pred_mask = _forecast_mask(df_ts, start, end, years)
        if pred_mask.any():
            Zpred = np.where(pred_mask, np.nan_to_num(Z, nan=0.0), np.nan)
            text_pred = [[""] * cols for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    val = Zpred[i, j]
                    if np.isfinite(val):
                        text_pred[i][j] = f"{months[j]} {years[i]}<br>Forecasted count: {int(round(val))}"
            fig.add_trace(
                go.Heatmap(
                    z=Zpred, x=months, y=years,
                    colorscale=SEQUENTIAL_ORANGE,
                    zmin=0, zmax=zmax,
                    showscale=False, opacity=0.85,
                    text=text_pred,
                    hovertemplate="%{text}<extra></extra>",
                    hoverongaps=False,
                ),
                row=r, col=c
            )

        # shared axes: only outer tiles show labels
        show_x = (r == nrows)
        show_y = (c == 1)
        fig.update_xaxes(row=r, col=c, showgrid=False, ticks="", showticklabels=show_x)
        fig.update_yaxes(row=r, col=c, showgrid=False, ticks="", showticklabels=show_y)

    # Slightly smaller subplot title text; keep them above each tile (default positioning)
    fig.update_annotations(font=dict(size=11))

    fig.update_layout(
        template="plotly_white",
        title=None,
        # Squeeze per-row height while leaving enough for titles above top row
        height=max(660, 200 * nrows),
        margin=dict(l=10, r=10, t=26, b=10),
        hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"),
    )
    return fig