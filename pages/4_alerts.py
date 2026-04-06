"""
Page 4 — Township Alerts

Analytical page for township-level anomaly detection.
Uses canonical processed data and compares the latest recent window
against each township's own recent and long-term history.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, callback, Output, Input, State

from components.colors import KEY_EVENT_COLORS
from components.loaders import load_acled_main, load_geojson, load_last_checked
from components.map_utils import apply_tight_geos, ensure_full_geoindex, filter_geo_by_property


COMBAT_EVENTS = (
    "Ground-based attack",
    "Air attack",
    "Drone attack",
    "Massacres",
)

WINDOW_DAYS = 30
LOCAL_LOOKBACK = 6
THRESH_MULT = 2.5
MAD_SCALE = 1.4826
PCT_FOR_EMERGENCE = 0.95

ALERT_COLORS = {
    "No unusual change": "#d7dde5",
    "Activity surge": "#2b6cb0",
    "Fatality surge": "#dc2626",
    "Both surges": "#7c3aed",
}
ALERT_CODES = {
    "No unusual change": 0,
    "Activity surge": 1,
    "Fatality surge": 2,
    "Both surges": 3,
}
ALERT_SORT = {
    "Both surges": 0,
    "Fatality surge": 1,
    "Activity surge": 2,
    "No unusual change": 3,
}

PLOTLY_FONT = "Avenir Next, Segoe UI, Arial, sans-serif"
PLOTLY_DISPLAY = "Iowan Old Style, Palatino Linotype, Book Antiqua, Georgia, serif"
PLOTLY_TEXT = "#415669"
PLOTLY_GRID = "#e6ddd0"
PLOTLY_HOVER_BG = "rgba(255,251,246,0.98)"
PLOTLY_HOVER_BORDER = "#d9cfbf"


def _chart_config(filename: str, *, show_modebar: bool = False) -> dict:
    return {
        "displayModeBar": "hover" if show_modebar else False,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 700,
            "width": 1200,
            "scale": 2,
        },
    }


def _empty_fig(message: str | None = None, *, height: int) -> go.Figure:
    fig = go.Figure()
    if message:
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, family=PLOTLY_FONT, color="#708192"),
        )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _fmt(n) -> str:
    return f"{int(n):,}"


def _safe_int(value) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def _fmt_metric(value) -> str:
    try:
        value = float(value)
    except Exception:
        return "0"
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _relative_to_baseline_text(current, baseline, label: str) -> str:
    try:
        current = float(current)
        baseline = float(baseline)
    except Exception:
        return f"relative to the {label}"

    if baseline <= 0:
        return f"against a near-zero {label}"

    ratio = current / baseline
    pct = (ratio - 1.0) * 100.0
    if ratio >= 2:
        return f"{ratio:.1f}x the {label}"
    return f"{pct:.0f}% above the {label}"


def _date_display(value) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def _window_start(end_date) -> pd.Timestamp:
    return pd.Timestamp(end_date).normalize() - pd.Timedelta(days=WINDOW_DAYS - 1)


def _window_label(end_date) -> str:
    end_date = pd.Timestamp(end_date).normalize()
    start_date = _window_start(end_date)
    return f"{_date_display(start_date)} to {_date_display(end_date)}"


def _build_window_endpoints(start_date, end_date) -> list[pd.Timestamp]:
    end_date = pd.Timestamp(end_date).normalize()
    min_end = pd.Timestamp(start_date).normalize() + pd.Timedelta(days=WINDOW_DAYS - 1)
    ends: list[pd.Timestamp] = []
    cursor = end_date
    while cursor >= min_end:
        ends.append(cursor)
        cursor -= pd.Timedelta(days=WINDOW_DAYS)
    return list(reversed(ends))


def _discrete_alert_scale():
    colors = [ALERT_COLORS["No unusual change"], ALERT_COLORS["Activity surge"],
              ALERT_COLORS["Fatality surge"], ALERT_COLORS["Both surges"]]
    return [
        (0.00, colors[0]), (0.2499, colors[0]),
        (0.25, colors[1]), (0.4999, colors[1]),
        (0.50, colors[2]), (0.7499, colors[2]),
        (0.75, colors[3]), (1.00, colors[3]),
    ]


def _geo_lookup(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        rows.append({
            "Tsp_Pcode": props.get("TS_PCODE"),
            "township": props.get("TS"),
            "admin1_geo": props.get("ST"),
        })
    df = pd.DataFrame(rows).dropna(subset=["Tsp_Pcode"]).drop_duplicates("Tsp_Pcode")
    return df


def _rolling_local_median(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    return prev.rolling(LOCAL_LOOKBACK, min_periods=1).median()


def _rolling_local_mad(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    med = prev.rolling(LOCAL_LOOKBACK, min_periods=1).median()
    mad = (prev - med).abs().rolling(LOCAL_LOOKBACK, min_periods=1).median()
    return mad * MAD_SCALE


def _expanding_global_median(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    return prev.expanding(min_periods=1).median()


def _expanding_global_mad(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    med = prev.expanding(min_periods=1).median()
    mad = (prev - med).abs().expanding(min_periods=1).median()
    return mad * MAD_SCALE


def _percentile_int(series: pd.Series, p: float) -> int:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 0
    value = np.nanpercentile(series.astype(float), p * 100.0, method="nearest")
    return int(max(0, round(value)))


def _scope_text(local_flag: int, global_flag: int) -> str | None:
    if local_flag and global_flag:
        return "recent windows and longer-term township history"
    if local_flag:
        return "recent windows"
    if global_flag:
        return "longer-term township history"
    return None


def _flag_with_reason(curr_val, base_med, base_mad, *, scope: str, metric: str, thresholds: dict):
    if pd.isna(curr_val) or pd.isna(base_med):
        return 0, "NONE"

    if float(base_med) == 0:
        thr = thresholds[f"thr_{scope}_zero_{metric}"]
        return int(curr_val >= thr), "EMERGENCE_ZERO" if curr_val >= thr else "NONE"
    if float(base_med) == 1:
        thr = thresholds[f"thr_{scope}_one_{metric}"]
        return int(curr_val >= thr), "EMERGENCE_ONE" if curr_val >= thr else "NONE"

    thr_mad = float(base_med) + THRESH_MULT * float(base_mad or 0)
    return int(curr_val > thr_mad), "MAD" if curr_val > thr_mad else "NONE"


def _build_flag_description(row: pd.Series) -> str:
    parts = []
    event_scope = _scope_text(row["flag_local_event"], row["flag_global_event"])
    fatal_scope = _scope_text(row["flag_local_fatal"], row["flag_global_fatal"])
    if fatal_scope:
        parts.append(f"Reported fatality estimates are unusually high compared with {fatal_scope}.")
    if event_scope:
        parts.append(f"Armed conflict activity is unusually high compared with {event_scope}.")
    if not parts:
        return "No unusual change flagged in the latest alert window."
    return " ".join(parts)


def _threshold_value(base_med, base_mad, *, scope: str, metric: str, thresholds: dict) -> float:
    try:
        base_med = float(base_med)
    except Exception:
        return 0.0
    base_mad = float(base_mad or 0)
    if base_med == 0:
        return float(thresholds[f"thr_{scope}_zero_{metric}"])
    if base_med == 1:
        return float(thresholds[f"thr_{scope}_one_{metric}"])
    return float(base_med + THRESH_MULT * base_mad)


def _assign_category(row: pd.Series) -> str:
    event_flag = max(int(row["flag_local_event"]), int(row["flag_global_event"]))
    fatal_flag = max(int(row["flag_local_fatal"]), int(row["flag_global_fatal"]))
    if event_flag == 0 and fatal_flag == 0:
        return "No unusual change"
    if event_flag == 1 and fatal_flag == 0:
        return "Activity surge"
    if event_flag == 0 and fatal_flag == 1:
        return "Fatality surge"
    return "Both surges"


@lru_cache(maxsize=1)
def _build_alert_frames():
    geojson = load_geojson()
    geo_lookup = _geo_lookup(geojson)

    main = load_acled_main().copy()
    main["key_event"] = main["key_event"].astype(str)
    main["admin1"] = main["admin1"].astype(str)
    main["event_date"] = pd.to_datetime(main["event_date"], errors="coerce").dt.normalize()
    main = main.dropna(subset=["event_date", "Tsp_Pcode"]).copy()

    latest_event_date = main["event_date"].max()

    full_mix = main[["Tsp_Pcode", "admin1", "event_date", "key_event", "fatalities"]].copy()

    combat = main[main["key_event"].isin(COMBAT_EVENTS)].copy()
    combat_daily = (
        combat.groupby(["Tsp_Pcode", "admin1", "event_date"], observed=True)
        .agg(events_1d=("event_id_cnty", "size"), fatalities_1d=("fatalities", "sum"))
        .reset_index()
    )

    date_range = pd.date_range(
        start=combat_daily["event_date"].min(),
        end=latest_event_date,
        freq="D",
    )
    grid = pd.MultiIndex.from_product(
        [geo_lookup["Tsp_Pcode"].tolist(), date_range],
        names=["Tsp_Pcode", "event_date"],
    ).to_frame(index=False)

    daily_full = (
        grid.merge(geo_lookup, on="Tsp_Pcode", how="left")
        .merge(combat_daily, on=["Tsp_Pcode", "event_date"], how="left")
    )
    daily_full["admin1"] = daily_full["admin1"].fillna(daily_full["admin1_geo"])
    daily_full["township"] = daily_full["township"].fillna(daily_full["Tsp_Pcode"])
    daily_full["events_1d"] = pd.to_numeric(daily_full["events_1d"], errors="coerce").fillna(0).astype(int)
    daily_full["fatalities_1d"] = pd.to_numeric(daily_full["fatalities_1d"], errors="coerce").fillna(0).astype(int)
    daily_full = daily_full.sort_values(["Tsp_Pcode", "event_date"]).reset_index(drop=True)

    daily_full["event_window"] = (
        daily_full.groupby("Tsp_Pcode", observed=True)["events_1d"]
        .transform(lambda s: s.rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).sum())
    )
    daily_full["fatalities_window"] = (
        daily_full.groupby("Tsp_Pcode", observed=True)["fatalities_1d"]
        .transform(lambda s: s.rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).sum())
    )

    window_ends = _build_window_endpoints(date_range.min(), latest_event_date)
    windows = daily_full[daily_full["event_date"].isin(window_ends)].copy()
    windows["local_med_event"] = windows.groupby("Tsp_Pcode", observed=True)["event_window"].transform(_rolling_local_median)
    windows["local_mad_event"] = windows.groupby("Tsp_Pcode", observed=True)["event_window"].transform(_rolling_local_mad)
    windows["glob_med_event"] = windows.groupby("Tsp_Pcode", observed=True)["event_window"].transform(_expanding_global_median)
    windows["glob_mad_event"] = windows.groupby("Tsp_Pcode", observed=True)["event_window"].transform(_expanding_global_mad)
    windows["local_med_fatal"] = windows.groupby("Tsp_Pcode", observed=True)["fatalities_window"].transform(_rolling_local_median)
    windows["local_mad_fatal"] = windows.groupby("Tsp_Pcode", observed=True)["fatalities_window"].transform(_rolling_local_mad)
    windows["glob_med_fatal"] = windows.groupby("Tsp_Pcode", observed=True)["fatalities_window"].transform(_expanding_global_median)
    windows["glob_mad_fatal"] = windows.groupby("Tsp_Pcode", observed=True)["fatalities_window"].transform(_expanding_global_mad)

    last_idx = windows.groupby("Tsp_Pcode", observed=True)["event_date"].idxmax()
    historical = windows.drop(index=last_idx, errors="ignore")

    thresholds = {
        "thr_local_zero_event": _percentile_int(historical.loc[historical["local_med_event"] == 0, "event_window"], PCT_FOR_EMERGENCE),
        "thr_local_one_event": _percentile_int(historical.loc[historical["local_med_event"] == 1, "event_window"], PCT_FOR_EMERGENCE),
        "thr_local_zero_fatal": _percentile_int(historical.loc[historical["local_med_fatal"] == 0, "fatalities_window"], PCT_FOR_EMERGENCE),
        "thr_local_one_fatal": _percentile_int(historical.loc[historical["local_med_fatal"] == 1, "fatalities_window"], PCT_FOR_EMERGENCE),
        "thr_global_zero_event": _percentile_int(historical.loc[historical["glob_med_event"] == 0, "event_window"], PCT_FOR_EMERGENCE),
        "thr_global_one_event": _percentile_int(historical.loc[historical["glob_med_event"] == 1, "event_window"], PCT_FOR_EMERGENCE),
        "thr_global_zero_fatal": _percentile_int(historical.loc[historical["glob_med_fatal"] == 0, "fatalities_window"], PCT_FOR_EMERGENCE),
        "thr_global_one_fatal": _percentile_int(historical.loc[historical["glob_med_fatal"] == 1, "fatalities_window"], PCT_FOR_EMERGENCE),
    }

    current = windows.loc[last_idx].copy().rename(columns={
        "event_date": "current_window_end",
        "event_window": "current_event_window",
        "fatalities_window": "current_fatalities_window",
    })

    current[["flag_local_event", "reason_local_event"]] = current.apply(
        lambda r: pd.Series(_flag_with_reason(
            r["current_event_window"], r["local_med_event"], r["local_mad_event"],
            scope="local", metric="event", thresholds=thresholds)),
        axis=1,
    )
    current[["flag_global_event", "reason_global_event"]] = current.apply(
        lambda r: pd.Series(_flag_with_reason(
            r["current_event_window"], r["glob_med_event"], r["glob_mad_event"],
            scope="global", metric="event", thresholds=thresholds)),
        axis=1,
    )
    current[["flag_local_fatal", "reason_local_fatal"]] = current.apply(
        lambda r: pd.Series(_flag_with_reason(
            r["current_fatalities_window"], r["local_med_fatal"], r["local_mad_fatal"],
            scope="local", metric="fatal", thresholds=thresholds)),
        axis=1,
    )
    current[["flag_global_fatal", "reason_global_fatal"]] = current.apply(
        lambda r: pd.Series(_flag_with_reason(
            r["current_fatalities_window"], r["glob_med_fatal"], r["glob_mad_fatal"],
            scope="global", metric="fatal", thresholds=thresholds)),
        axis=1,
    )

    current["flag_description"] = current.apply(_build_flag_description, axis=1)
    current["map_category"] = current.apply(_assign_category, axis=1)
    current["map_code"] = current["map_category"].map(ALERT_CODES).astype(int)
    current["flagged"] = current["map_category"] != "No unusual change"
    current["severity_score"] = (
        current["map_category"].map({"No unusual change": 0, "Activity surge": 1000, "Fatality surge": 2000, "Both surges": 3000})
        + current["current_fatalities_window"].fillna(0) * 10
        + current["current_event_window"].fillna(0)
    )
    current = current.sort_values(
        ["severity_score", "current_fatalities_window", "current_event_window", "township"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    latest_end = current["current_window_end"].max()
    meta = {
        "latest_end": latest_end,
        "window_start": _window_start(latest_end),
        "window_label": _window_label(latest_end),
        "combat_label": "Ground-based attack, Air attack, Drone attack, and Massacres",
        "thresholds": thresholds,
        "window_days": WINDOW_DAYS,
        "local_blocks": LOCAL_LOOKBACK,
    }
    return current, windows, full_mix, geojson, meta


def _get_defaults():
    current, _, _, _, meta = _build_alert_frames()
    checked = load_last_checked()
    checked_str = checked.get("display") or checked.get("date_display") or "Not yet recorded"
    checked_note = checked.get("cadence_note") or "ACLED check every 6 hours"
    timezone_label = checked.get("timezone_label") or "Yangon time"
    if timezone_label and timezone_label.lower() not in checked_note.lower():
        checked_note = f"{checked_note} · {timezone_label}"

    default_tsp = current.loc[current["flagged"], "Tsp_Pcode"].iloc[0] if current["flagged"].any() else current["Tsp_Pcode"].iloc[0]
    regions = sorted(current["admin1"].dropna().astype(str).unique().tolist())
    return {
        "checked_str": checked_str,
        "checked_note": checked_note,
        "window_label": meta["window_label"],
        "default_tsp": default_tsp,
        "regions": regions,
    }


def _get_layout_defaults():
    checked = load_last_checked()
    checked_str = checked.get("display") or checked.get("date_display") or "Not yet recorded"
    checked_note = checked.get("cadence_note") or "ACLED check every 6 hours"
    timezone_label = checked.get("timezone_label") or "Yangon time"
    if timezone_label and timezone_label.lower() not in checked_note.lower():
        checked_note = f"{checked_note} · {timezone_label}"

    main = load_acled_main()
    regions = sorted(main["admin1"].dropna().astype(str).unique().tolist())
    return {
        "checked_str": checked_str,
        "checked_note": checked_note,
        "regions": regions,
    }


def _chip_class(category: str) -> str:
    slug = category.lower().replace(" ", "-")
    return f"alert-chip alert-chip--{slug}"


def _filter_snapshot(region: str | None):
    current, history, full_mix, geojson, meta = _build_alert_frames()
    if region:
        current = current[current["admin1"] == region].copy()
        history = history[history["admin1"] == region].copy()
        full_mix = full_mix[full_mix["admin1"] == region].copy()
        geojson = filter_geo_by_property(geojson, "ST", region)
    return current, history, full_mix, geojson, meta


def _build_map(snapshot: pd.DataFrame, geojson: dict, window_label: str, selected_tsp: str | None):
    df = ensure_full_geoindex(
        snapshot[[
            "Tsp_Pcode", "township", "admin1", "map_code", "map_category",
            "current_event_window", "current_fatalities_window", "flag_description",
        ]],
        geojson,
        id_col="Tsp_Pcode",
        fill_columns=["township", "admin1", "map_code", "map_category", "current_event_window", "current_fatalities_window", "flag_description"],
        fill_value=None,
    )
    df["township"] = df["township"].replace("", np.nan).fillna(df["Tsp_Pcode"])
    df["admin1"] = df["admin1"].replace("", np.nan).fillna("")
    df["map_code"] = pd.to_numeric(df["map_code"], errors="coerce").fillna(0).astype(int)
    df["map_category"] = df["map_category"].replace("", np.nan).fillna("No unusual change")
    df["current_event_window"] = pd.to_numeric(df["current_event_window"], errors="coerce").fillna(0).astype(int)
    df["current_fatalities_window"] = pd.to_numeric(df["current_fatalities_window"], errors="coerce").fillna(0).astype(int)
    df["flag_description"] = df["flag_description"].replace("", np.nan).fillna("No unusual change flagged in the latest alert window.")

    customdata = np.stack([
        df["township"],
        df["admin1"],
        df["map_category"],
        df["current_event_window"],
        df["current_fatalities_window"],
    ], axis=-1)

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=df["Tsp_Pcode"],
        z=df["map_code"],
        featureidkey="properties.TS_PCODE",
        colorscale=_discrete_alert_scale(),
        zmin=0,
        zmax=3,
        showscale=False,
        marker_line_color="rgba(255,255,255,0.92)",
        marker_line_width=0.5,
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Alert: %{customdata[2]}<br>"
            + f"{WINDOW_DAYS}-day activity: "
            + "%{customdata[3]}<br>"
            + f"{WINDOW_DAYS}-day fatalities: "
            + "%{customdata[4]}<extra></extra>"
        ),
    ))

    if selected_tsp:
        selected_geo = filter_geo_by_property(geojson, "TS_PCODE", selected_tsp)
        if selected_geo.get("features"):
            fig.add_trace(go.Choropleth(
                geojson=selected_geo,
                locations=[selected_tsp],
                z=[1],
                featureidkey="properties.TS_PCODE",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                hoverinfo="skip",
                marker_line_color="#1f2937",
                marker_line_width=1.9,
            ))

    apply_tight_geos(fig, geojson, height=760, show_colorbar=False)
    fig.update_layout(
        title=dict(
            text=window_label,
            x=0.5,
            xanchor="center",
            font=dict(size=13, color="#4a5d73", family=PLOTLY_FONT),
        ),
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color="#24384d", size=11),
            align="left",
        ),
    )
    return fig


def _build_ranking(snapshot: pd.DataFrame):
    flagged = snapshot[snapshot["flagged"]].copy()
    if flagged.empty:
        return html.Div("No unusual change is currently flagged in this view.", className="table-empty")

    rows = []
    for idx, row in flagged.head(12).iterrows():
        rows.append(html.Div([
            html.Div(f"{idx + 1}", className="alert-list-rank"),
            html.Div([
                html.Div(row["township"], className="alert-list-name"),
                html.Div(row["admin1"], className="alert-list-meta"),
            ], className="alert-list-main"),
            html.Div(row["map_category"], className=_chip_class(row["map_category"])),
            html.Div([
                html.Div(f"{_safe_int(row['current_event_window'])} events", className="alert-list-stat"),
                html.Div(f"{_safe_int(row['current_fatalities_window'])} fatalities", className="alert-list-stat"),
            ], className="alert-list-stats"),
        ], className="alert-list-row"))
    return html.Div(rows, className="alert-list")


def _township_options(snapshot: pd.DataFrame):
    ordered = snapshot.sort_values(
        ["severity_score", "township"],
        ascending=[False, True],
    )
    return [
        {
            "label": f"{row.township} · {row.admin1} · {row.map_category}",
            "value": row.Tsp_Pcode,
        }
        for row in ordered.itertuples()
    ]


def _build_detail_figure(history: pd.DataFrame, township_name: str):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(f"Activity ({WINDOW_DAYS}-day total)", f"Fatalities ({WINDOW_DAYS}-day total)"),
    )

    x = history["event_date"]
    fig.add_trace(go.Scatter(
        x=x,
        y=history["event_window"],
        mode="lines",
        name="Current activity",
        line=dict(color="#2563eb", width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x,
        y=history["local_med_event"],
        mode="lines",
        name="Recent baseline",
        line=dict(color="#60a5fa", width=1.4, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x,
        y=history["glob_med_event"],
        mode="lines",
        name="Long-term baseline",
        line=dict(color="#1d4ed8", width=1.2, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x,
        y=history["fatalities_window"],
        mode="lines",
        name="Current fatalities",
        line=dict(color="#dc2626", width=2.2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x,
        y=history["local_med_fatal"],
        mode="lines",
        name="Recent fatality baseline",
        line=dict(color="#f87171", width=1.4, dash="dash"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x,
        y=history["glob_med_fatal"],
        mode="lines",
        name="Long-term fatality baseline",
        line=dict(color="#b91c1c", width=1.2, dash="dot"),
    ), row=2, col=1)

    fig.update_layout(
        height=450,
        margin=dict(l=44, r=18, t=24, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color="#24384d"),
        ),
    )
    fig.update_annotations(font=dict(size=11, family=PLOTLY_FONT, color="#5b6d80"), yshift=10)
    fig.update_xaxes(
        showgrid=False,
        tickformat="%b\n%Y",
        tickfont=dict(size=10, family=PLOTLY_FONT),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        zeroline=False,
        tickfont=dict(size=10, family=PLOTLY_FONT),
    )
    return fig


def _build_mix_figure(full_mix: pd.DataFrame, township_code: str, window_start: pd.Timestamp, window_end: pd.Timestamp):
    mix = (
        full_mix[
            (full_mix["Tsp_Pcode"] == township_code)
            & (full_mix["event_date"] >= window_start)
            & (full_mix["event_date"] <= window_end)
        ]
        .groupby("key_event", observed=True)
        .size()
        .rename("events")
        .reset_index()
    )
    if mix.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No event mix available for this township in the latest alert window.",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, family=PLOTLY_FONT, color="#6b7c89"),
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=16, b=0), xaxis_visible=False, yaxis_visible=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    mix["color"] = mix["key_event"].map(KEY_EVENT_COLORS).fillna("#9ca3af")
    mix = mix.sort_values("events", ascending=True)

    fig = go.Figure(go.Bar(
        x=mix["events"],
        y=mix["key_event"],
        orientation="h",
        marker=dict(color=mix["color"]),
        text=mix["events"].map(_fmt),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x} events<extra></extra>",
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=170, r=24, t=20, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        xaxis=dict(showgrid=True, gridcolor=PLOTLY_GRID, zeroline=False, tickfont=dict(size=10), title=f"Events in latest {WINDOW_DAYS}-day window"),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color="#24384d"),
        ),
    )
    return fig


def _metric_reason_block(
    row: pd.Series,
    *,
    metric: str,
    thresholds: dict,
):
    if metric == "event":
        title = "Armed conflict activity"
        current = row["current_event_window"]
        recent_base = row["local_med_event"]
        long_base = row["glob_med_event"]
        recent_flag = int(row["flag_local_event"])
        long_flag = int(row["flag_global_event"])
        recent_reason = row["reason_local_event"]
        long_reason = row["reason_global_event"]
        recent_threshold = _threshold_value(recent_base, row["local_mad_event"], scope="local", metric="event", thresholds=thresholds)
        long_threshold = _threshold_value(long_base, row["glob_mad_event"], scope="global", metric="event", thresholds=thresholds)
    else:
        title = "Reported fatalities"
        current = row["current_fatalities_window"]
        recent_base = row["local_med_fatal"]
        long_base = row["glob_med_fatal"]
        recent_flag = int(row["flag_local_fatal"])
        long_flag = int(row["flag_global_fatal"])
        recent_reason = row["reason_local_fatal"]
        long_reason = row["reason_global_fatal"]
        recent_threshold = _threshold_value(recent_base, row["local_mad_fatal"], scope="local", metric="fatal", thresholds=thresholds)
        long_threshold = _threshold_value(long_base, row["glob_mad_fatal"], scope="global", metric="fatal", thresholds=thresholds)

    if recent_flag and long_flag:
        comparison = "This signal is above both the recent and longer-term township thresholds."
    elif recent_flag:
        comparison = "This signal is above the recent township threshold."
    elif long_flag:
        comparison = "This signal is above the longer-term township threshold."
    else:
        comparison = "This signal is not currently driving the alert."

    emergence_bits = []
    if recent_reason in {"EMERGENCE_ZERO", "EMERGENCE_ONE"}:
        emergence_bits.append("recent baseline reflects near-zero prior activity")
    if long_reason in {"EMERGENCE_ZERO", "EMERGENCE_ONE"}:
        emergence_bits.append("longer-term baseline reflects near-zero prior activity")
    emergence_note = ""
    if emergence_bits:
        emergence_note = " This is treated as emergence because the " + " and the ".join(emergence_bits) + "."

    recent_compare = _relative_to_baseline_text(current, recent_base, "recent baseline")
    long_compare = _relative_to_baseline_text(current, long_base, "long-term baseline")

    summary = (
        f"Current {WINDOW_DAYS}-day total is {_fmt_metric(current)}. That is {recent_compare} "
        f"and {long_compare}. {comparison}{emergence_note}"
    )

    return html.Div([
        html.Div(title, className="alert-reason-title"),
        html.Div(summary, className="alert-reason-copy"),
        html.Div([
            html.Div([
                html.Span("Current", className="alert-mini-stat-key"),
                html.Span(_fmt_metric(current), className="alert-mini-stat-value"),
            ], className="alert-mini-stat"),
            html.Div([
                html.Span("Recent baseline", className="alert-mini-stat-key"),
                html.Span(_fmt_metric(recent_base), className="alert-mini-stat-value"),
            ], className="alert-mini-stat"),
            html.Div([
                html.Span("Long-term baseline", className="alert-mini-stat-key"),
                html.Span(_fmt_metric(long_base), className="alert-mini-stat-value"),
            ], className="alert-mini-stat"),
            html.Div([
                html.Span("Recent threshold", className="alert-mini-stat-key"),
                html.Span(_fmt_metric(recent_threshold), className="alert-mini-stat-value"),
            ], className="alert-mini-stat"),
            html.Div([
                html.Span("Long-term threshold", className="alert-mini-stat-key"),
                html.Span(_fmt_metric(long_threshold), className="alert-mini-stat-value"),
            ], className="alert-mini-stat"),
        ], className="alert-reason-stats"),
    ], className="alert-reason-block")


def _detail_children(row: pd.Series, *, thresholds: dict, window_label: str):
    chips = html.Div([
        html.Div([
            html.Span("Selected township", className="hero-status-key"),
            html.Span(row["township"], className="hero-status-value-inline"),
        ], className="hero-status-pill"),
        html.Div([
            html.Span("Region", className="hero-status-key"),
            html.Span(row["admin1"], className="hero-status-value-inline"),
        ], className="hero-status-pill"),
        html.Div([
            html.Span("Alert type", className="hero-status-key"),
            html.Span(row["map_category"], className="hero-status-value-inline"),
        ], className="hero-status-pill"),
        html.Div([
            html.Span("Window", className="hero-status-key"),
            html.Span(window_label, className="hero-status-value-inline"),
        ], className="hero-status-pill"),
    ], className="hero-status-inline hero-status-inline--left")

    return [
        html.Div("Inspect selected township", className="overview-note-kicker"),
        html.Div(row["township"], className="card-title"),
        html.Div(
            f"This panel explains why the township was or was not flagged in {window_label}. "
            f"Recent baseline means the median of the previous {LOCAL_LOOKBACK} {WINDOW_DAYS}-day windows for this township. "
            f"Long-term baseline means the median of all earlier {WINDOW_DAYS}-day windows for the same township.",
            className="card-subtitle",
        ),
        chips,
        html.Div([
            _metric_reason_block(row, metric="event", thresholds=thresholds),
            _metric_reason_block(row, metric="fatal", thresholds=thresholds),
        ], className="alert-reason-grid"),
    ]


def layout():
    defaults = _get_layout_defaults()
    checked_str = defaults["checked_str"]
    checked_note = defaults["checked_note"]
    region_options = [{"label": r, "value": r} for r in defaults["regions"]]
    init_map = _empty_fig("Preparing the latest alert window…", height=760)
    init_trend = _empty_fig("Loading township trend…", height=450)
    init_mix = _empty_fig("Loading current event mix…", height=420)
    init_rank = html.Div("Loading the current flagged townships…", className="table-empty")

    legend = html.Div([
        html.Span("No unusual change", className=_chip_class("No unusual change")),
        html.Span("Activity surge", className=_chip_class("Activity surge")),
        html.Span("Fatality surge", className=_chip_class("Fatality surge")),
        html.Span("Both surges", className=_chip_class("Both surges")),
    ], className="alert-legend")

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Township Alerts", className="page-title"),
                html.Div(
                    "Show, as transparently as possible, where recent armed conflict activity or reported fatalities are unusually high relative to each township's own history.",
                    className="page-subtitle",
                ),
                html.Div([
                    html.Div("Prototype · work in progress", className="hero-pill hero-pill--prototype"),
                    html.Div("Aim: transparent township alerting", className="hero-pill"),
                    html.Div(f"Window: latest available {WINDOW_DAYS} days", className="hero-pill"),
                    html.Div("Combat scope: Ground-based attack, Air attack, Drone attack, and Massacres", className="hero-pill"),
                ], className="hero-pill-row hero-pill-row--tight"),
            ], className="page-header-left"),
            html.Div([
                html.Div([
                    html.Span("Alert Window", className="hero-status-key"),
                    html.Span("Calculating latest 30-day window…", id="an-window-value", className="hero-status-value-inline"),
                ], className="hero-status-pill"),
                html.Div([
                    html.Span("Last checked with ACLED", className="hero-status-key"),
                    html.Span(checked_str, className="hero-status-value-inline"),
                ], className="hero-status-pill"),
                html.Div(checked_note, className="hero-status-note hero-status-note--inline"),
            ], className="hero-status-inline"),
        ], className="page-header overview-hero"),

        html.Div([
            html.Div([
                html.Div([
                    html.Label("Region", className="filter-label"),
                    dcc.Dropdown(
                        id="an-region",
                        options=region_options,
                        value=None,
                        clearable=True,
                        placeholder="All Myanmar",
                    ),
                ], className="filter-group"),
                html.Div([
                    html.Label("Inspect township", className="filter-label"),
                    dcc.Dropdown(
                        id="an-township",
                        options=[],
                        value=None,
                        clearable=False,
                        searchable=True,
                        placeholder="Loading townships…",
                    ),
                ], className="filter-group filter-group--wide"),
                ], className="filter-controls"),
        ], className="filter-card"),

        html.Div([
            html.Div([
                html.Div("Why 30 days?", className="alert-method-title"),
                html.Div(
                    f"This page now works from event-level dates, not monthly buckets. The current window uses the latest available {WINDOW_DAYS} days so the signal is more responsive while still smoothing daily noise.",
                    className="alert-method-copy",
                ),
            ], className="alert-method-block"),
            html.Div([
                html.Div("Current window", className="alert-method-title"),
                html.Div(
                    "The exact current alert window appears above once loaded. It ends with the latest event date currently represented in the processed data, not today's date.",
                    className="alert-method-copy",
                ),
            ], className="alert-method-block"),
            html.Div([
                html.Div("Recent baseline", className="alert-method-title"),
                html.Div(
                    f"The dashed baseline is the median of the previous {LOCAL_LOOKBACK} {WINDOW_DAYS}-day windows for the same township.",
                    className="alert-method-copy",
                ),
            ], className="alert-method-block"),
            html.Div([
                html.Div("Long-term baseline", className="alert-method-title"),
                html.Div(
                    f"The dotted baseline is the median of all earlier {WINDOW_DAYS}-day windows for the same township.",
                    className="alert-method-copy",
                ),
            ], className="alert-method-block"),
            html.Div([
                html.Div("How a flag is triggered", className="alert-method-title"),
                html.Div(
                    f"A township is flagged when the current {WINDOW_DAYS}-day total rises above a township-specific threshold. "
                    "Most thresholds are baseline plus 2.5 times the township's median absolute deviation; near-zero histories use emergence thresholds instead.",
                    className="alert-method-copy",
                ),
            ], className="alert-method-block"),
        ], className="dash-card alert-method-card alert-method-grid"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Township Alert Map", className="card-title"),
                        html.Div("Click a township to inspect why it is flagged in the latest alert window.",
                                 className="card-subtitle"),
                    ], className="map-stage-copy"),
                ], className="dash-card-head map-stage-head"),
                dcc.Loading(
                        dcc.Graph(
                            id="an-map",
                            figure=init_map,
                        className="map-graph",
                        config={**_chart_config("township_alert_map", show_modebar=True), "displayModeBar": True},
                    ),
                    type="dot",
                    color="#2563eb",
                ),
                html.Div([
                    html.Div("How to read the map", className="highest-label"),
                    html.Div("Grey townships look normal in the latest alert window. Colored townships are unusually high relative to their own history. Click a township to see the exact comparison against its recent and long-term baselines.",
                             className="highest-region"),
                ], className="highest-events"),
                html.Div(legend, className="dash-card-body"),
            ], className="dash-card map-stage alerts-map-stage"),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Flagged Townships", className="kpi-label"),
                        html.Div("—", id="an-kpi-flagged", className="kpi-value"),
                        html.Div("non-normal in the current alert window", className="kpi-sub"),
                    ], className="kpi-card kpi-accent-teal"),
                    html.Div([
                        html.Div("Activity Alerts", className="kpi-label"),
                        html.Div("—", id="an-kpi-activity", className="kpi-value"),
                        html.Div("activity unusually high against township history", className="kpi-sub"),
                    ], className="kpi-card kpi-accent-blue"),
                    html.Div([
                        html.Div("Fatality Alerts", className="kpi-label"),
                        html.Div("—", id="an-kpi-fatal", className="kpi-value"),
                        html.Div("fatality estimates unusually high against history", className="kpi-sub"),
                    ], className="kpi-card kpi-accent-red"),
                    html.Div([
                        html.Div("Both Surges", className="kpi-label"),
                        html.Div("—", id="an-kpi-both", className="kpi-value"),
                        html.Div("activity and fatalities high together", className="kpi-sub"),
                    ], className="kpi-card kpi-accent-purple"),
                ], className="kpis-row kpis-row--4 kpis-compact"),

                html.Div([
                    html.Div([
                        html.Div("Top Flagged Townships", className="card-title"),
                        html.Div("Ranked by the strength of the current signal after comparing each township to its own history.",
                                 className="card-subtitle"),
                    ], className="dash-card-head"),
                    html.Div(init_rank, id="an-ranking", className="dash-card-body"),
                ], className="dash-card"),

            ], className="col-charts"),
        ], className="page-body alerts-body"),

        html.Div([
            html.Div([
                html.Div(
                    id="an-detail-copy",
                    children=[
                        html.Div("Inspect selected township", className="overview-note-kicker"),
                        html.Div("Preparing township explanation…", className="card-title"),
                        html.Div(
                            "The selected township summary, thresholds, and comparison against recent and long-term baselines will appear here once the alert window is ready.",
                            className="card-subtitle",
                        ),
                    ],
                    className="dash-card-body",
                ),
            ], className="dash-card alert-detail-card"),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Township Trend", className="card-title"),
                        html.Div(
                            f"Solid lines show rolling {WINDOW_DAYS}-day totals. Dashed lines show the recent baseline. Dotted lines show the long-term baseline.",
                            className="card-subtitle",
                        ),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(id="an-trend", figure=init_trend, config=_chart_config("township_alert_trend")),
                        type="dot", color="#2563eb"
                    ),
                ], className="dash-card"),

                html.Div([
                    html.Div([
                        html.Div("Current Event Mix", className="card-title"),
                        html.Div(
                            "Composition of all dashboard event types recorded in this township during the latest alert window.",
                            className="card-subtitle",
                        ),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(id="an-mix", figure=init_mix, config=_chart_config("township_alert_mix")),
                        type="dot", color="#2563eb"
                    ),
                ], className="dash-card"),
            ], className="alerts-detail-grid"),
        ], className="alerts-inspector-section"),
    ], className="page-wrap")


@callback(
    Output("an-window-value", "children"),
    Output("an-map", "figure"),
    Output("an-kpi-flagged", "children"),
    Output("an-kpi-activity", "children"),
    Output("an-kpi-fatal", "children"),
    Output("an-kpi-both", "children"),
    Output("an-ranking", "children"),
    Output("an-township", "options"),
    Output("an-township", "value"),
    Input("an-region", "value"),
    Input("an-map", "clickData"),
    State("an-township", "value"),
)
def update_alert_snapshot(region, click_data, current_tsp):
    snapshot, _, _, geojson, meta = _filter_snapshot(region)
    options = _township_options(snapshot)
    option_values = [o["value"] for o in options]
    clicked_tsp = None
    if click_data and click_data.get("points"):
        clicked_tsp = click_data["points"][0].get("location")

    if clicked_tsp in option_values:
        selected_tsp = clicked_tsp
    elif current_tsp in option_values:
        selected_tsp = current_tsp
    else:
        selected_tsp = option_values[0] if option_values else None

    fig = _build_map(snapshot, geojson, meta["window_label"], selected_tsp)
    flagged = snapshot["flagged"].sum()
    event_alerts = ((snapshot["flag_local_event"] == 1) | (snapshot["flag_global_event"] == 1)).sum()
    fatal_alerts = ((snapshot["flag_local_fatal"] == 1) | (snapshot["flag_global_fatal"] == 1)).sum()
    both_alerts = (snapshot["map_category"] == "Both surges").sum()

    return (
        meta["window_label"],
        fig,
        _fmt(flagged),
        _fmt(event_alerts),
        _fmt(fatal_alerts),
        _fmt(both_alerts),
        _build_ranking(snapshot),
        options,
        selected_tsp,
    )


@callback(
    Output("an-detail-copy", "children"),
    Output("an-trend", "figure"),
    Output("an-mix", "figure"),
    Input("an-township", "value"),
    Input("an-region", "value"),
    prevent_initial_call=True,
)
def update_alert_detail(township_code, region):
    snapshot, history, full_mix, _, meta = _filter_snapshot(region)
    if township_code not in snapshot["Tsp_Pcode"].astype(str).tolist():
        township_code = snapshot.iloc[0]["Tsp_Pcode"]

    row = snapshot.loc[snapshot["Tsp_Pcode"] == township_code].iloc[0]
    hist = history[history["Tsp_Pcode"] == township_code].copy()
    return (
        _detail_children(row, thresholds=meta["thresholds"], window_label=meta["window_label"]),
        _build_detail_figure(hist, row["township"]),
        _build_mix_figure(full_mix, township_code, meta["window_start"], meta["latest_end"]),
    )
