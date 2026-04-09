"""
Page 1 — Overview  (design v4)

Filter bar:  DatePickerRange + quick-date buttons + Region + Type + Apply / Reset
Map:         Choropleth with neighbouring-country labels.
             Mode cards: Period Total  |  Quarterly Animation (Plotly native — no lag).
Charts:      Monthly Trend (milestone lines) + Key Activity Types bar.
KPI cards:   No sparklines — just big numbers.

Animation strategy
------------------
Plotly NATIVE animation frames are used for the quarterly choropleth.
All ~20 frames (330 z-values each) are bundled in the initial figure JSON (~50 KB).
Frame advancement is pure client-side JS — zero Python round-trips per frame.
This eliminates the per-frame callback lag of the previous Patch() approach.
"""
import re
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate

from components.loaders   import (load_acled_main, load_geojson, load_last_checked)
from components.colors    import (KEY_EVENT_COLORS, KEY_EVENT_ORDER,
                                   SEQUENTIAL_BLUES_ZERO_GREY)
from components.map_utils import apply_tight_geos, add_neighbor_labels, filter_geo_by_property
from components.page_bits import data_disclaimer


# ── Constants ──────────────────────────────────────────────────────────────────

# (date, hover title, short pill, line color, description)
MILESTONES = [
    ("2021-02-01", "Feb 1, 2021 – Military Coup",           "Military Coup",       "#dc2626",
     "The Myanmar military seized power and arrested elected leaders, sparking nationwide protests."),
    ("2021-09-07", "Sept 7, 2021 – NUG Declaration of People's Defensive War", "People's Defensive War",    "#d97706",
     "The NUG called for armed resistance against the military, escalating the conflict."),
    ("2023-10-27", "Oct 27, 2023 – Operation 1027",         "Op. 1027",   "#7c3aed",
     "Rebel alliance launched a major offensive, capturing key territory in northern Shan State."),
    ("2025-03-28", "Mar 28, 2025 – 7.7 Earthquake",         "Earthquake", "#0891b2",
     "A major earthquake near Mandalay caused heavy destruction and worsened the crisis."),
    ("2025-12-28", "Dec 28, 2025 – SAC-organized Elections", "SAC-run Elections",  "#6b7280",
     "The SAC held phased elections in controlled areas; the process was widely rejected as illegitimate."),
]

EVENT_DEFINITIONS = {
    "Ground-based attack":          "ground clashes, explosions and shelling between armed groups",
    "Air attack":                   "airstrikes by military aircraft",
    "Drone attack":                 "aerial strikes carried out by unmanned drone aircraft",
    "Massacre":                     "attack on civilians resulting in 5 or more fatalities",
    "Massacres":                    "attack on civilians resulting in 5 or more fatalities",
    "Violence against civilians":   "targeted killings, abductions or sexual violence against civilians",
    "Arrests":                      "detention or arrest of civilians or combatants by any actor",
    "Looting/property destruction": "theft, arson or deliberate destruction of civilian property",
    "Protests":                     "peaceful or disruptive demonstrations, strikes and gatherings",
    "Displacement":                 "forced displacement of civilians from their homes or villages",
    "Others":                       "remaining activity outside the main dashboard groups, especially strategic developments such as changes to group activity, disrupted weapons use, base establishment, agreements, and non-violent transfers of territory",
}

PLOTLY_FONT = "Avenir Next, Segoe UI, Arial, sans-serif"
PLOTLY_DISPLAY = "Iowan Old Style, Palatino Linotype, Book Antiqua, Georgia, serif"
PLOTLY_TEXT = "#415669"
PLOTLY_GRID = "#e6ddd0"
PLOTLY_HOVER_BG = "rgba(255,251,246,0.98)"
PLOTLY_HOVER_BORDER = "#d9cfbf"

# ── Module-level helpers ───────────────────────────────────────────────────────

def _get_defaults() -> dict:
    df = load_acled_main()
    checked = load_last_checked()
    checked_str = checked.get("display") or checked.get("date_display") or "Not yet recorded"
    checked_note = checked.get("cadence_note") or "Checked against ACLED weekly on Tuesday evening"
    timezone_label = checked.get("timezone_label") or "Yangon time"
    if timezone_label and timezone_label.lower() not in checked_note.lower():
        checked_note = f"{checked_note} · {timezone_label}"
    return {
        "start_val":  df["event_date"].min().strftime("%Y-%m-%d"),
        "end_val":    df["event_date"].max().strftime("%Y-%m-%d"),
        "start_month": df["event_date"].min().strftime("%Y-%m"),
        "end_month":   df["event_date"].max().strftime("%Y-%m"),
        "latest_str":  df["event_date"].max().strftime("%d %b %Y"),
        "checked_str": checked_str,
        "checked_note": checked_note,
    }


# ── Generic helpers ────────────────────────────────────────────────────────────

def _fmt(n) -> str:
    return f"{int(n):,}"


def _empty_fig(msg="No data for this selection", height=500):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#7a8895", family=PLOTLY_FONT), xref="paper", yref="paper")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis_visible=False, yaxis_visible=False,
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
    )
    return fig


def _month_opts(min_date, max_date) -> list[dict]:
    months = pd.date_range(
        pd.Timestamp(min_date).to_period("M").to_timestamp(),
        pd.Timestamp(max_date).to_period("M").to_timestamp(),
        freq="MS",
    )
    return [{"value": m.strftime("%Y-%m"), "label": m.strftime("%b %Y")} for m in months]


def _geo_pcodes(geo: dict) -> list[str]:
    return [f["properties"]["TS_PCODE"] for f in geo["features"]]


def _tsp_text(geo: dict) -> list[str]:
    return [f["properties"].get("TS", f["properties"]["TS_PCODE"])
            for f in geo["features"]]


def _month_to_quarter(month_str: str) -> str:
    year, m = month_str.split("-")
    return f"{year}-Q{(int(m) - 1) // 3 + 1}"


def _quarter_label(q_str: str) -> str:
    q_str = str(q_str)
    if "-" in q_str:
        year, q = q_str.split("-")
    else:
        year, q = q_str[:4], q_str[4:]
    return f"{q} {year}"


def _format_selected_range(
    start_date: str | None,
    end_date: str | None,
    start_month: str | None = None,
    end_month: str | None = None,
) -> str:
    def _fmt_date(value: str | None, *, month_fallback: bool = False) -> str:
        if value:
            try:
                return pd.to_datetime(value).strftime("%d %b %Y")
            except Exception:
                pass
        if month_fallback:
            return value or "—"
        return "—"

    start_txt = _fmt_date(start_date)
    end_txt = _fmt_date(end_date)
    if start_txt == "—" and start_month:
        start_txt = pd.Timestamp(start_month + "-01").strftime("%d %b %Y")
    if end_txt == "—" and end_month:
        end_ts = pd.Period(end_month, freq="M").end_time
        end_txt = pd.Timestamp(end_ts).strftime("%d %b %Y")
    if start_txt == end_txt:
        return start_txt
    return f"{start_txt} to {end_txt}"


def _window_days(start_date: str | None, end_date: str | None) -> int | None:
    if not start_date or not end_date:
        return None
    try:
        return max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1, 1)
    except Exception:
        return None


def _filter_overview_events(acled_df, start_date, end_date, region, key_events):
    df = acled_df.copy()
    if start_date:
        df = df[df["event_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["event_date"] <= pd.to_datetime(end_date)]
    if region:
        df = df[df["admin1"] == region]
    if key_events:
        df = df[df["key_event"].isin(key_events)]
    return df


def _is_generic_civilian(name: str) -> bool:
    return bool(re.match(r"^civilians?\s*\(", name.strip(), re.IGNORECASE))


def _hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_filter_summary(start_month, end_month, region, key_events,
                          mode: str = "time_range",
                          preset_label: str | None = None) -> str:
    def fmt_m(m):
        try:
            return pd.Timestamp(m + "-01").strftime("%b %Y")
        except Exception:
            return m or ""

    if key_events and len(key_events) == 1:
        evt  = key_events[0]
        defn = EVENT_DEFINITIONS.get(evt, "")
        type_str = f"{evt} ({defn})" if defn else evt
    elif key_events:
        type_str = f"{len(key_events)} event types"
    else:
        type_str = "All event types"

    geo = f"in {region}" if region else "across Myanmar"

    if mode == "animated":
        return (f"Animating: {type_str} {geo} · "
                f"Quarter by quarter from {fmt_m(start_month)}")
    if preset_label:
        return f"Showing: {type_str} {geo} · {preset_label}"
    return f"Showing: {type_str} {geo} · {fmt_m(start_month)} – {fmt_m(end_month)}"


def _build_filter_chips(start_month, end_month, region, key_events,
                        mode: str = "time_range",
                        preset_label: str | None = None,
                        start_date: str | None = None,
                        end_date: str | None = None):
    def fmt_m(m):
        try:
            return pd.Timestamp(m + "-01").strftime("%b %Y")
        except Exception:
            return m or ""

    if key_events and len(key_events) == 1:
        type_str = key_events[0]
    elif key_events:
        type_str = f"{len(key_events)} event types"
    else:
        type_str = "All event types"

    time_str = preset_label or _format_selected_range(start_date, end_date, start_month, end_month)
    chips = [
        ("Date", time_str),
        ("Region", region or "All Myanmar"),
        ("Type", type_str),
    ]
    if mode == "animated":
        chips.append(("View", "Animated by quarter"))
    else:
        chips.append(("View", "Cumulative"))

    return html.Div(
        [html.Span("Now showing", className="filter-chip filter-chip--label")] +
        [
            html.Span([
                html.Span(f"{label}: ", className="filter-chip-key"),
                html.Span(value, className="filter-chip-value"),
            ], className="filter-chip")
            for label, value in chips
        ],
        className="filter-chip-row",
    )


def _chart_config(filename: str, *, show_modebar: bool = False) -> dict:
    """Modebar config that shows only the download (camera) button on hover."""
    return {
        "displayModeBar": "hover" if show_modebar else False,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        ],
        "toImageButtonOptions": {
            "format": "png", "filename": filename,
            "height": 600, "width": 1200, "scale": 2,
        },
    }


# ── 95th-percentile colour cap ─────────────────────────────────────────────────

def _q95(series: pd.Series) -> int:
    s = series[series > 0]
    if s.empty:
        return 1
    return max(int(s.quantile(0.95)) if len(s) >= 5 else int(s.max()), 1)


# ── Store builder ──────────────────────────────────────────────────────────────

def _build_store(acled_df, geo, start_date, end_date, region, key_events,
                 mode: str) -> dict | None:
    df = _filter_overview_events(acled_df, start_date, end_date, region, key_events)
    if df.empty:
        return None

    active_geo = filter_geo_by_property(geo, "ST", region) if region else geo
    pcodes     = _geo_pcodes(active_geo)
    text       = _tsp_text(active_geo)
    n          = len(pcodes)
    pcode_idx  = {p: i for i, p in enumerate(pcodes)}
    start_month = pd.to_datetime(start_date).strftime("%Y-%m") if start_date else None
    end_month = pd.to_datetime(end_date).strftime("%Y-%m") if end_date else None

    agg = df.groupby(["Tsp_Pcode", "event_date"]).agg(
        events=("event_id_cnty", "nunique"),
        fatalities=("fatalities", "sum"),
    ).reset_index()

    if mode == "animated":
        agg["quarter"] = agg["event_date"].dt.to_period("Q").astype(str)
        qagg = agg.groupby(["Tsp_Pcode", "quarter"]).agg(
            events=("events", "sum"), fatalities=("fatalities", "sum"),
        ).reset_index()
        all_quarters = sorted(qagg["quarter"].unique())
        matrix_ev, matrix_fat = [], []
        for q in all_quarters:
            z_ev  = [0] * n
            z_fat = [0] * n
            sub = qagg[qagg["quarter"] == q]
            for _, row in sub.iterrows():
                i = pcode_idx.get(row["Tsp_Pcode"])
                if i is not None:
                    z_ev[i]  = int(row["events"])
                    z_fat[i] = int(row["fatalities"])
            matrix_ev.append(z_ev)
            matrix_fat.append(z_fat)
        # Use per-quarter distribution (not cumulative totals) so the colour
        # scale is calibrated to a single quarter's intensity, making changes
        # visible across the animation.
        return {
            "mode": "animated", "frames": all_quarters,
            "frame_labels": [_quarter_label(q) for q in all_quarters],
            "matrix": matrix_ev, "matrix_fat": matrix_fat,
            "max_val":     _q95(qagg["events"]),
            "max_val_fat": _q95(qagg["fatalities"]),
            "start_month": start_month,
            "end_month": end_month,
            "pcodes": pcodes, "text": text, "region": region,
        }
    else:
        total = agg.groupby("Tsp_Pcode").agg(
            events=("events", "sum"), fatalities=("fatalities", "sum"),
        ).reset_index()
        z_ev  = [0] * n
        z_fat = [0] * n
        for _, row in total.iterrows():
            i = pcode_idx.get(row["Tsp_Pcode"])
            if i is not None:
                z_ev[i]  = int(row["events"])
                z_fat[i] = int(row["fatalities"])
        return {
            "mode": "time_range",
            "z": z_ev, "z_fat": z_fat,
            "max_val":     _q95(pd.Series(z_ev)),
            "max_val_fat": _q95(pd.Series(z_fat)),
            "start_month": start_month,
            "end_month": end_month,
            "pcodes": pcodes, "text": text, "region": region,
        }


# ── Choropleth builders ────────────────────────────────────────────────────────

def _build_animated_choropleth(store: dict, geo: dict,
                               metric: str = "events") -> go.Figure:
    """Plotly native animated choropleth — all frames bundled client-side."""
    active_geo = filter_geo_by_property(geo, "ST", store["region"]) if store.get("region") else geo
    matrix     = store["matrix_fat"] if metric == "fatalities" else store["matrix"]
    mv         = store["max_val_fat"] if metric == "fatalities" else store["max_val"]
    cb_title   = "Fatalities" if metric == "fatalities" else "Events"
    hover_lbl  = cb_title
    labels     = store["frame_labels"]
    n_frames   = len(labels)

    initial_z  = matrix[0] if n_frames > 0 else [0] * len(store["pcodes"])

    # Base trace
    base_trace = go.Choropleth(
        geojson=active_geo,
        locations=store["pcodes"],
        z=initial_z,
        text=store["text"],
        featureidkey="properties.TS_PCODE",
        colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
        zmin=0, zmax=mv,
        marker_line_width=0.3, marker_line_color="#ffffff",
        hovertemplate=f"<b>%{{text}}</b><br>{hover_lbl}: %{{z:,}}<extra></extra>",
        colorbar=dict(
            title=dict(
                text=cb_title,
                side="top",
                font=dict(size=11, color=PLOTLY_TEXT, family=PLOTLY_FONT),
            ),
            tickfont=dict(size=10, color=PLOTLY_TEXT, family=PLOTLY_FONT),
            orientation="h",
            thickness=10,
            len=0.34,
            x=0.5,
            xanchor="center",
            y=0.035,
            yanchor="bottom",
            outlinewidth=0,
        ),
    )

    # Frames — only update z per frame (efficient, tiny payload per frame)
    frames = [
        go.Frame(
            data=[go.Choropleth(z=z)],
            traces=[0],
            name=str(i),
            layout=go.Layout(title_text=f"<b>{lbl}</b>"),
        )
        for i, (z, lbl) in enumerate(zip(matrix, labels))
    ]

    # Slider steps — label only Q1 and endpoints
    slider_steps = [
        {
            "args": [[str(i)],
                     {"frame": {"duration": 400, "redraw": True}, "mode": "immediate"}],
            "label": lbl if (lbl.startswith("Q1") or i == 0 or i == n_frames - 1) else "",
            "method": "animate",
        }
        for i, lbl in enumerate(labels)
    ]

    sliders = [{
        "active": 0,
        "steps": slider_steps,
            "x": 0.05, "len": 0.9,
            "y": 0, "yanchor": "top",
            "pad": {"t": 50, "b": 10},
            "currentvalue": {
                "visible": True, "prefix": "Quarter: ",
                "xanchor": "center",
                "font": {"size": 11, "color": PLOTLY_TEXT, "family": PLOTLY_FONT},
            },
            "transition": {"duration": 200},
            "bgcolor": "rgba(250,246,240,0.96)",
            "bordercolor": PLOTLY_HOVER_BORDER,
            "borderwidth": 1,
        }]

    updatemenus = [{
        "type": "buttons",
        "showactive": False,
        "y": 1.06, "x": 0.01, "xanchor": "left", "yanchor": "top",
        "bgcolor": "rgba(255,251,246,0.96)",
        "bordercolor": PLOTLY_HOVER_BORDER,
        "borderwidth": 1,
        "font": {"size": 11, "color": PLOTLY_TEXT, "family": PLOTLY_FONT},
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 600, "redraw": True},
                                "fromcurrent": True}],
                "label": "▶  Play",
                "method": "animate",
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate"}],
                "label": "⏸  Pause",
                "method": "animate",
            },
        ],
    }]

    fig = go.Figure(data=[base_trace], frames=frames)

    apply_tight_geos(fig, active_geo, height=860)
    if not store.get("region"):
        add_neighbor_labels(fig)

    fig.update_layout(
        margin=dict(l=0, r=0, t=52, b=112),
        title=dict(
            text=f"<b>{labels[0] if labels else ''}</b>",
            x=0.5, y=0.97, font=dict(size=14, color="#32475b", family=PLOTLY_DISPLAY),
        ),
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=11),
        ),
        sliders=sliders,
        updatemenus=updatemenus,
    )
    return fig


def _build_choropleth(
    store: dict,
    geo: dict,
    metric: str = "events",
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> go.Figure:
    """Static (period total) choropleth."""
    active_geo = filter_geo_by_property(geo, "ST", store["region"]) if store.get("region") else geo
    z          = store["z_fat"] if metric == "fatalities" else store["z"]
    mv         = store["max_val_fat"] if metric == "fatalities" else store["max_val"]
    cb_title   = "Fatalities" if metric == "fatalities" else "Events"
    hover_lbl  = cb_title

    fig = go.Figure(go.Choropleth(
        geojson=active_geo,
        locations=store["pcodes"],
        z=z, text=store["text"],
        featureidkey="properties.TS_PCODE",
        colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
        zmin=0, zmax=mv,
        marker_line_width=0.3, marker_line_color="#ffffff",
        hovertemplate=f"<b>%{{text}}</b><br>{hover_lbl}: %{{z:,}}<extra></extra>",
        colorbar=dict(
            title=dict(
                text=cb_title,
                side="top",
                font=dict(size=11, color=PLOTLY_TEXT, family=PLOTLY_FONT),
            ),
            tickfont=dict(size=10, color=PLOTLY_TEXT, family=PLOTLY_FONT),
            orientation="h",
            thickness=10,
            len=0.36,
            x=0.5,
            xanchor="center",
            y=0.04,
            yanchor="bottom",
            outlinewidth=0,
        ),
    ))

    apply_tight_geos(fig, active_geo, height=820)
    if not store.get("region"):
        add_neighbor_labels(fig)

    title_text = _format_selected_range(
        start_date,
        end_date,
        store.get("start_month"),
        store.get("end_month"),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=36, b=28),
        title=dict(text=f"<b>{title_text}</b>", x=0.5, y=0.97,
                   font=dict(size=14, color="#32475b", family=PLOTLY_DISPLAY)),
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=11),
        ),
    )
    return fig


# ── Trend chart ────────────────────────────────────────────────────────────────

def _build_trend(acled_df, start_date, end_date, region, key_events) -> go.Figure:
    df = _filter_overview_events(acled_df, start_date, end_date, region, key_events)
    if df.empty:
        return _empty_fig(height=320)

    if key_events and len(key_events) == 1:
        label = key_events[0]
        color = KEY_EVENT_COLORS.get(label, "#94a3b8")
    else:
        label = "Total Events"
        color = "#355c84"

    start_ts = pd.to_datetime(start_date) if start_date else df["event_date"].min()
    end_ts = pd.to_datetime(end_date) if end_date else df["event_date"].max()
    use_daily = (_window_days(start_date, end_date) or 999) <= 45

    if use_daily:
        idx = pd.date_range(start_ts, end_ts, freq="D")
        series = (
            df.groupby(df["event_date"].dt.normalize())["event_id_cnty"]
            .nunique()
            .reindex(idx, fill_value=0)
        )
        trend = pd.DataFrame({"dt": idx, "events": series.values})
        hover_fmt = "%d %b %Y"
        tick_fmt = "%d %b"
    else:
        monthly = (
            df.assign(period_dt=df["event_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("period_dt")["event_id_cnty"]
            .nunique()
            .reset_index(name="events")
            .sort_values("period_dt")
        )
        trend = monthly.rename(columns={"period_dt": "dt"})
        hover_fmt = "%b %Y"
        tick_fmt = "%b %Y"

    x_min = trend["dt"].min()
    x_max = trend["dt"].max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["dt"], y=trend["events"],
        mode="lines+markers" if use_daily else "lines",
        name=label,
        showlegend=False,
        fill="tozeroy",
        fillcolor=_hex_rgba(color, 0.10),
        line=dict(color=color, width=2.5),
        marker=dict(color=color, size=6),
        hovertemplate=f"<b>{label}</b><br>%{{x|{hover_fmt}}}: %{{y:,}}<extra></extra>",
    ))

    if x_min is not None:
        for i, (date_str, title, short_name, milestone_color, desc) in enumerate(MILESTONES):
            mdt = pd.Timestamp(date_str)
            if x_min <= mdt <= (x_max + pd.Timedelta(days=90)):
                fig.add_shape(
                    type="line",
                    x0=mdt, x1=mdt, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color=milestone_color, width=1.4, dash="dot"),
                    opacity=0.5,
                )
                fig.add_annotation(
                    x=mdt, y=1.075 if i % 2 == 0 else 1.035, xref="x", yref="paper",
                    text=f"<b>{short_name}</b>",
                    showarrow=False,
                    font=dict(size=7.4, color=milestone_color, family=PLOTLY_FONT),
                    xanchor="center", yanchor="bottom",
                    bgcolor=PLOTLY_HOVER_BG,
                    bordercolor=milestone_color, borderwidth=1, borderpad=1,
                )
                fig.add_trace(go.Scatter(
                    x=[mdt], y=[0.5], yaxis="y2",
                    mode="markers",
                    marker=dict(color=milestone_color, size=7, symbol="diamond",
                                line=dict(color="white", width=1), opacity=0.75),
                    showlegend=False, name="",
                    hovertemplate=(
                        f"<b>{title}</b><br>"
                        f"<span style='color:#6b7280'>{desc}</span>"
                        f"<extra></extra>"
                    ),
                ))

    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=40, b=4),
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
        showlegend=False,
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        xaxis=dict(
            showgrid=False,
            tickformat=tick_fmt,
            tickangle=-30,
            tickfont=dict(size=10, color=PLOTLY_TEXT, family=PLOTLY_FONT),
            linecolor=PLOTLY_GRID,
        ),
        yaxis=dict(showgrid=True, gridcolor=PLOTLY_GRID, zeroline=False,
                   tickfont=dict(size=10, color=PLOTLY_TEXT, family=PLOTLY_FONT),
                   title=None, rangemode="tozero"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=11),
        ),
    )
    return fig


# ── Key Activity Types bar chart ──────────────────────────────────────────────

def _build_activity_bar(acled_df, start_date, end_date, region, key_events) -> go.Figure:
    df = _filter_overview_events(acled_df, start_date, end_date, region, None)
    if df.empty:
        return _empty_fig("No data", height=420)

    counts = (
        df.groupby("key_event", observed=True)["event_id_cnty"]
        .nunique()
        .reindex(KEY_EVENT_ORDER).fillna(0)
    )
    counts = counts[counts > 0].sort_values(ascending=False)
    if counts.empty:
        return _empty_fig("No data", height=420)

    selected = set(key_events or [])
    defs = [[EVENT_DEFINITIONS.get(k, "")] for k in counts.index]
    accent = "#355c84"
    context = "#ced9e7"
    colors = [
        accent if (selected and evt in selected) else (context if selected else accent)
        for evt in counts.index
    ]
    fig = go.Figure(go.Bar(
        y=counts.index,
        x=counts.values,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(56,78,99,0.08)", width=0.5)),
        text=[f"{int(v):,}" for v in counts.values],
        textposition="outside",
        textfont=dict(size=10, color=PLOTLY_TEXT, family=PLOTLY_FONT),
        cliponaxis=False,
        customdata=defs,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x:,} events<br>"
            "<i>%{customdata[0]}</i>"
            "<extra></extra>"
        ),
    ))
    if selected:
        note = "Selected event type highlighted for context" if len(selected) == 1 else "Selected event types highlighted for context"
        fig.add_annotation(
            x=1, y=1.07, xref="paper", yref="paper",
            text=note,
            showarrow=False,
            xanchor="right",
            font=dict(size=9, color="#64748B", family=PLOTLY_FONT),
        )
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=80, t=12, b=4),
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        yaxis=dict(
            tickfont=dict(size=9, color=PLOTLY_TEXT, family=PLOTLY_FONT), title=None,
            autorange="reversed", automargin=True,
        ),
        xaxis=dict(
            showgrid=True, gridcolor=PLOTLY_GRID,
            tickfont=dict(size=9, color=PLOTLY_TEXT, family=PLOTLY_FONT), title=None,
            rangemode="tozero",
        ),
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=11),
        ),
    )
    return fig


# ── Highest Events helper ─────────────────────────────────────────────────────

def _highest_events(acled_df, start_date, end_date, region, key_events):
    df = _filter_overview_events(acled_df, start_date, end_date, region, key_events)
    if df.empty:
        return "—", "—"
    by_tsp = (
        df.groupby("Tsp_Pcode", observed=True)["event_id_cnty"]
        .nunique()
        .sort_values(ascending=False)
    )
    if by_tsp.empty:
        return "—", "—"
    top_pcode = by_tsp.index[0]
    match = acled_df[acled_df["Tsp_Pcode"] == top_pcode]["admin3"]
    top_name = str(match.iloc[0]) if not match.empty else str(top_pcode)
    return top_name, f"{int(by_tsp.iloc[0]):,}"


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    meta        = _get_defaults()
    df          = load_acled_main()
    latest_str  = meta["latest_str"]
    checked_str = meta["checked_str"]
    checked_note = meta["checked_note"]
    start_month = meta["start_month"]
    end_month   = meta["end_month"]
    start_date  = meta["start_val"]   # YYYY-MM-DD for DatePickerSingle
    end_date    = meta["end_val"]

    admin1_opts  = sorted(df["admin1"].dropna().unique())
    key_evt_opts = [k for k in KEY_EVENT_ORDER if k in df["key_event"].unique()]

    default_store = {
        "start_month":  start_month,
        "end_month":    end_month,
        "start_date":   start_date,
        "end_date":     end_date,
        "region":       None,
        "key_events":   None,
        "preset_label": None,
    }

    # Keep first layout light; charts are hydrated by callback after mount.
    init_map        = _empty_fig(height=820)
    init_trend      = _empty_fig(height=320)
    init_bar        = _empty_fig("Loading conflict types...", height=420)
    init_conflicts  = "—"
    init_fatalities = "—"
    h_region, h_count = "—", "—"
    init_summary    = _build_filter_chips(
        start_month,
        end_month,
        None,
        None,
        "time_range",
        start_date=start_date,
        end_date=end_date,
    )

    return html.Div([

        # ── hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="ov-applied-filters", data=default_store),
        dcc.Store(id="ov-mode", data="time_range"),

        # ── hero ───────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("Overview", className="page-title"),
                html.Div(
                    "Township-level Myanmar conflict dashboard for exploring reported events, fatality estimates, spatial concentration, and change over time.",
                    className="page-subtitle",
                ),
                html.Div([
                    html.Span("Also see", className="page-link-label"),
                    html.A("Actor Analysis", href="/actor"),
                    html.A("Township Alerts", href="/alerts"),
                    html.A("About", href="/about"),
                ], className="page-link-row"),
                html.Div([
                    html.Div("Prototype · work in progress", className="hero-pill hero-pill--prototype"),
                ], className="hero-pill-row hero-pill-row--tight"),
            ], className="page-header-left"),
            html.Div([
                html.Div([
                    html.Span("Data Through", className="hero-status-key"),
                    html.Span(latest_str, className="hero-status-value-inline"),
                ], className="hero-status-pill"),
                html.Div([
                    html.Span("Last checked with ACLED", className="hero-status-key"),
                    html.Span(checked_str, className="hero-status-value-inline"),
                ], className="hero-status-pill"),
                html.Div(checked_note, className="hero-status-note hero-status-note--inline"),
            ], className="hero-status-inline"),
        ], className="page-header overview-hero"),

        # ── filter card ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Button("Last 7 days",    id="ov-btn-7d",  n_clicks=0, className="quick-btn"),
                html.Button("Last 30 days",   id="ov-btn-30d", n_clicks=0, className="quick-btn"),
                html.Button("Since Feb 2021", id="ov-btn-all", n_clicks=0, className="quick-btn"),

                html.Div([
                    html.Div([
                        html.Div("From", className="date-card-header"),
                        dcc.DatePickerSingle(
                            id="ov-from-date",
                            date=start_date,
                            display_format="DD/MM/YYYY",
                            first_day_of_week=1,
                            className="date-picker-single",
                        ),
                    ], className="date-card date-card--compact"),
                ], className="filter-strip-item filter-strip-item--date"),

                html.Div([
                    html.Div([
                        html.Div("To", className="date-card-header"),
                        dcc.DatePickerSingle(
                            id="ov-to-date",
                            date=end_date,
                            display_format="DD/MM/YYYY",
                            first_day_of_week=1,
                            className="date-picker-single",
                        ),
                    ], className="date-card date-card--compact"),
                ], className="filter-strip-item filter-strip-item--date"),

                html.Div([
                    dcc.Dropdown(id="ov-region",
                                 options=[{"label": r, "value": r} for r in admin1_opts],
                                 multi=False, placeholder="Region · All Myanmar", clearable=True),
                ], className="filter-strip-item filter-strip-item--select"),

                html.Div([
                    dcc.Dropdown(id="ov-key-event",
                                 options=[{"label": k, "value": k} for k in key_evt_opts],
                                 multi=False, placeholder="Event Type · All", clearable=True),
                ], className="filter-strip-item filter-strip-item--type"),

                html.Button("Reset", id="ov-reset-btn",
                            n_clicks=0, className="btn-reset btn-reset--solo"),
            ], className="filter-strip-row"),
            html.Div(id="ov-key-event-note", className="filter-strip-help"),
        ], id="ov-filter-card", className="filter-card filter-card--inline"),

        # ── filter summary ─────────────────────────────────────────────────────
        html.Div(init_summary, id="ov-filter-summary", className="filter-summary"),

        # ── national picture feature grid ─────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H2("Conflict Geography", className="card-title"),
                        html.Div(
                            "Distribution of reported events or reported fatality estimates across Myanmar townships",
                            className="card-subtitle",
                        ),
                    ], className="map-stage-copy"),
                    html.Div([
                        dcc.RadioItems(
                            id="ov-metric",
                            options=[
                                {"label": "Events",     "value": "events"},
                                {"label": "Fatalities", "value": "fatalities"},
                            ],
                            value="events",
                            inline=True,
                            className="metric-toggle metric-toggle--quiet",
                        ),
                        html.Div([
                            html.Div("Cumulative", className="view-toggle-label"),
                        ], id="ov-mode-total-btn", n_clicks=0,
                           className="view-toggle-btn view-toggle-btn--active"),
                        html.Div([
                            html.Div("Animated", className="view-toggle-label"),
                        ], id="ov-mode-anim-btn", n_clicks=0,
                           className="view-toggle-btn"),
                    ], className="map-stage-controls"),
                ], className="dash-card-head map-stage-head"),

                dcc.Loading(
                        dcc.Graph(
                            id="ov-map",
                            figure=init_map,
                            className="map-graph",
                            style={"height": "100%"},
                            config={**_chart_config("myanmar_conflict_map", show_modebar=True),
                                "displayModeBar": True},
                        ),
                    type="dot", color="#2563eb",
                ),

                html.Div([
                    html.Div("Most Active Township", className="highest-label"),
                    html.Div([
                        html.Div(h_region, id="ov-highest-region", className="highest-region"),
                        html.Div(h_count, id="ov-highest-count", className="highest-count"),
                    ], className="highest-values"),
                ], className="highest-events"),
            ], className="dash-card map-stage overview-map-stage"),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Reported Events", className="kpi-label"),
                        html.Div(init_conflicts, id="ov-kpi-conflicts", className="kpi-value"),
                        html.Div("All reported dashboard event records in the selected scope", className="kpi-sub"),
                    ], className="kpi-card kpi-card--hero kpi-accent-blue"),
                    html.Div([
                        html.Div("Reported Fatality Estimate", className="kpi-label"),
                        html.Div(init_fatalities, id="ov-kpi-fatalities", className="kpi-value"),
                        html.Div("Summed ACLED reported fatalities in the selected scope", className="kpi-sub"),
                    ], className="kpi-card kpi-card--support kpi-accent-red"),
                ], className="kpis-row kpis-row--2 overview-kpi-strip"),

                html.Div([
                    html.Div([
                        html.H2("Event Trend", className="card-title"),
                        html.Div(
                            "Use the timeline to see when reported events rose, fell, or shifted around major turning points",
                            className="card-subtitle",
                        ),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(
                            id="ov-trend",
                            figure=init_trend,
                            config=_chart_config("myanmar_monthly_trend"),
                        ),
                        type="dot", color="#2563eb",
                    ),
                ], className="dash-card"),

                html.Div([
                    html.Div([
                        html.H2("Event Types", className="card-title"),
                        html.Div(
                            "Compare the composition of the selected period, not just the total volume",
                            className="card-subtitle",
                        ),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(
                            id="ov-activity-bar",
                            figure=init_bar,
                            config=_chart_config("myanmar_activity_types"),
                        ),
                        type="dot", color="#2563eb",
                    ),
                ], className="dash-card"),
            ], className="overview-feature-side"),
        ], className="overview-feature-grid"),

        # ── data disclaimer ────────────────────────────────────────────────────
        data_disclaimer(),

    ], className="page-wrap")


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# 1. View toggle buttons → update mode store + button styles
@callback(
    Output("ov-mode",           "data"),
    Output("ov-mode-total-btn", "className"),
    Output("ov-mode-anim-btn",  "className"),
    Input("ov-mode-total-btn",  "n_clicks"),
    Input("ov-mode-anim-btn",   "n_clicks"),
    prevent_initial_call=True,
)
def switch_mode(n_total, n_anim):
    if ctx.triggered_id == "ov-mode-anim-btn":
        return "animated", "view-toggle-btn", "view-toggle-btn view-toggle-btn--active"
    return "time_range", "view-toggle-btn view-toggle-btn--active", "view-toggle-btn"


# 2. Quick date preset buttons → update date pickers
@callback(
    Output("ov-from-date",       "date"),
    Output("ov-to-date",         "date"),
    Input("ov-btn-7d",           "n_clicks"),
    Input("ov-btn-30d",          "n_clicks"),
    Input("ov-btn-all",          "n_clicks"),
    prevent_initial_call=True,
)
def set_quick_dates(n7, n30, nall):
    max_dt = load_acled_main()["event_date"].max()
    if ctx.triggered_id == "ov-btn-7d":
        start_dt = max_dt - pd.Timedelta(days=7)
    elif ctx.triggered_id == "ov-btn-30d":
        start_dt = max_dt - pd.Timedelta(days=30)
    else:
        start_dt = pd.Timestamp("2021-02-01")
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = max_dt.strftime("%Y-%m-%d")
    return start_date, end_date


# 4. Filter controls → save filter state automatically
@callback(
    Output("ov-applied-filters", "data", allow_duplicate=True),
    Input("ov-from-date",        "date"),
    Input("ov-to-date",          "date"),
    Input("ov-region",           "value"),
    Input("ov-key-event",        "value"),
    prevent_initial_call=True,
)
def apply_filters(from_date, to_date, region, key_events):
    d = _get_defaults()
    start_m = (from_date or d["start_val"])[:7]
    end_m   = (to_date   or d["end_val"])[:7]
    if isinstance(key_events, str):
        key_events = [key_events]
    return {"start_month": start_m, "end_month": end_m,
            "start_date": from_date or d["start_val"],
            "end_date": to_date or d["end_val"],
            "region": region, "key_events": key_events,
            "preset_label": None}


@callback(
    Output("ov-key-event-note", "children"),
    Input("ov-key-event", "value"),
    prevent_initial_call=False,
)
def update_incident_type_note(selected_event):
    if not selected_event:
        return ""
    definition = EVENT_DEFINITIONS.get(selected_event, "")
    if not definition:
        return ""
    if selected_event == "Others":
        return "Others mainly covers records not shown separately here, especially strategic developments such as changes to group activity, disrupted weapons use, base establishment, agreements, non-violent transfers of territory, and a small number of riot records."
    return f"{selected_event}: {definition}."


# 5. Reset button → clear all filters and restore date pickers to full range
@callback(
    Output("ov-region",          "value"),
    Output("ov-key-event",       "value"),
    Output("ov-from-date",       "date",               allow_duplicate=True),
    Output("ov-to-date",         "date",               allow_duplicate=True),
    Output("ov-applied-filters", "data", allow_duplicate=True),
    Input("ov-reset-btn",        "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n):
    d = _get_defaults()
    defaults = {"start_month": d["start_month"], "end_month": d["end_month"],
                "start_date": d["start_val"], "end_date": d["end_val"],
                "region": None, "key_events": None, "preset_label": None}
    return None, None, d["start_val"], d["end_val"], defaults


# 6. Filters or mode or metric → rebuild all charts
@callback(
    Output("ov-map",            "figure"),
    Output("ov-kpi-conflicts",  "children"),
    Output("ov-kpi-fatalities", "children"),
    Output("ov-filter-summary", "children"),
    Output("ov-trend",          "figure"),
    Output("ov-activity-bar",   "figure"),
    Output("ov-highest-region", "children"),
    Output("ov-highest-count",  "children"),
    Input("ov-applied-filters", "data"),
    Input("ov-mode",            "data"),
    Input("ov-metric",          "value"),
    prevent_initial_call=False,
)
def update_charts(applied, mode, metric):
    if not applied:
        raise PreventUpdate

    start_month  = applied.get("start_month")
    end_month    = applied.get("end_month")
    start_date   = applied.get("start_date")
    end_date     = applied.get("end_date")
    region       = applied.get("region")
    key_events   = applied.get("key_events")
    preset_label = applied.get("preset_label")
    if isinstance(key_events, str):
        key_events = [key_events]

    mode   = mode   or "time_range"
    metric = metric or "events"

    acled = load_acled_main()
    geo   = load_geojson()

    ac = _filter_overview_events(acled, start_date, end_date, region, key_events)
    n_conflicts  = _fmt(len(ac))
    n_fatalities = _fmt(int(ac["fatalities"].sum()))

    filter_summary = _build_filter_chips(start_month, end_month, region, key_events,
                                         mode, preset_label,
                                         start_date=start_date, end_date=end_date)
    store = _build_store(acled, geo, start_date, end_date, region, key_events, mode)

    if store is None:
        empty = _empty_fig("No data for this selection", height=820)
        return (empty, n_conflicts, n_fatalities, filter_summary,
                _empty_fig(height=320), _empty_fig("No data", height=350), "—", "—")

    if mode == "animated":
        fig_map = _build_animated_choropleth(store, geo, metric)
    else:
        fig_map = _build_choropleth(
            store,
            geo,
            metric,
            start_date=start_date,
            end_date=end_date,
        )

    fig_trend = _build_trend(acled, start_date, end_date, region, key_events)
    fig_bar   = _build_activity_bar(acled, start_date, end_date, region, key_events)
    h_region, h_count = _highest_events(acled, start_date, end_date, region, key_events)

    return fig_map, n_conflicts, n_fatalities, filter_summary, fig_trend, fig_bar, h_region, h_count
