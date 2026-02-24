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
import datetime
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate

from components.loaders   import (load_acled_main, load_monthly_township,
                                   load_geojson)
from components.colors    import (KEY_EVENT_COLORS, KEY_EVENT_ORDER,
                                   SEQUENTIAL_BLUES_ZERO_GREY)
from components.map_utils import apply_tight_geos, filter_geo_by_property


# ── Constants ──────────────────────────────────────────────────────────────────

MILESTONES = [
    ("2021-02-01", "Military Coup"),
    ("2021-09-07", "NUG Defensive War"),
    ("2023-10-27", "Operation 1027"),
    ("2025-03-28", "7.7 Earthquake"),
    ("2025-12-28", "General Elections"),
]

EVENT_DEFINITIONS = {
    "Ground-based attack":          "armed clashes and explosions on the ground",
    "Air attack":                   "confirmed military airstrikes by state forces",
    "Drone attack":                 "aerial attacks using unmanned drone aircraft",
    "Massacre":                     "violence against civilians with 5+ fatalities",
    "Massacres":                    "violence against civilians with 5+ fatalities",
    "Violence against civilians":   "targeted attacks, abductions, or sexual violence",
    "Arrests":                      "detention or arrest of individuals by any actor",
    "Looting/property destruction": "theft or destruction of civilian property",
    "Protests":                     "demonstrations and gatherings",
    "Displacement":                 "forced movement of civilians",
    "Others":                       "conflict events not classified in the above categories",
}

# Neighbouring country labels — lon/lat placed just outside Myanmar's borders.
# Rendered as Scattergeo text (italic, grey) on the choropleth figure.
NEIGHBOR_LABELS = [
    ("Bangladesh", 22.0,  90.8),   # west of Rakhine
    ("India",       27.8,  94.5),  # north of Kachin/Sagaing
    ("China",       25.5, 102.0),  # northeast of Shan/Kachin
    ("Laos",        22.0, 102.8),  # east of Shan
    ("Thailand",    14.5, 101.5),  # southeast of Tanintharyi
]


# ── Module-level helpers ───────────────────────────────────────────────────────

def _get_defaults() -> dict:
    df = load_acled_main()
    return {
        "start_val":  df["event_date"].min().strftime("%Y-%m-%d"),
        "end_val":    df["event_date"].max().strftime("%Y-%m-%d"),
        "start_month": df["event_date"].min().strftime("%Y-%m"),
        "end_month":   df["event_date"].max().strftime("%Y-%m"),
        "latest_str":  df["event_date"].max().strftime("%d %b %Y"),
    }


# ── Generic helpers ────────────────────────────────────────────────────────────

def _fmt(n) -> str:
    return f"{int(n):,}"


def _empty_fig(msg="No data for this selection", height=500):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#9ca3af"), xref="paper", yref="paper")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis_visible=False, yaxis_visible=False,
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
    year, q = q_str.split("-")
    return f"{q} {year}"


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
        type_str = f"{len(key_events)} conflict types"
    else:
        type_str = "All conflict types"

    geo = f"in {region}" if region else "across Myanmar"

    if mode == "animated":
        return (f"Animating: {type_str} {geo} · "
                f"Quarter by quarter from {fmt_m(start_month)}")
    if preset_label:
        return f"Showing: {type_str} {geo} · {preset_label}"
    return f"Showing: {type_str} {geo} · {fmt_m(start_month)} – {fmt_m(end_month)}"


def _chart_config(filename: str) -> dict:
    """Modebar config that shows only the download (camera) button on hover."""
    return {
        "displayModeBar": "hover",
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


# ── Neighbouring country labels on the choropleth map ─────────────────────────

def _add_country_labels(fig: go.Figure) -> None:
    """Add italic grey text labels for Myanmar's neighbours on the geo map."""
    fig.add_trace(go.Scattergeo(
        lon=[ll[2] for ll in NEIGHBOR_LABELS],
        lat=[ll[1] for ll in NEIGHBOR_LABELS],
        text=[ll[0] for ll in NEIGHBOR_LABELS],
        mode="text",
        textfont=dict(size=9, color="#94A3B8",
                      family="Inter, Segoe UI, sans-serif"),
        showlegend=False,
        hoverinfo="skip",
        name="",
    ))
    # Expand the geo viewport beyond Myanmar's tight bounds to show labels
    fig.update_geos(
        lonaxis_range=[89.5, 104.0],
        lataxis_range=[8.0,  30.5],
    )


# ── Store builder ──────────────────────────────────────────────────────────────

def _build_store(monthly_df, geo, start_month, end_month, region, key_events,
                 mode: str) -> dict | None:
    df = monthly_df.copy()
    if start_month: df = df[df["month"] >= start_month]
    if end_month:   df = df[df["month"] <= end_month]
    if region:      df = df[df["admin1"] == region]
    if key_events:  df = df[df["key_event"].isin(key_events)]
    if df.empty:    return None

    active_geo = filter_geo_by_property(geo, "ST", region) if region else geo
    pcodes     = _geo_pcodes(active_geo)
    text       = _tsp_text(active_geo)
    n          = len(pcodes)
    pcode_idx  = {p: i for i, p in enumerate(pcodes)}

    agg = df.groupby(["Tsp_Pcode", "month"]).agg(
        events=("events", "sum"), fatalities=("fatalities", "sum"),
    ).reset_index()

    if mode == "animated":
        agg["quarter"] = agg["month"].apply(_month_to_quarter)
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
        colorbar=dict(title=cb_title, thickness=12, len=0.4, x=1.01),
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
            "label": lbl if (lbl.endswith("Q1") or i == 0 or i == n_frames - 1) else "",
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
            "font": {"size": 11, "color": "#475569"},
        },
        "transition": {"duration": 200},
        "bgcolor": "#F1F5F9",
        "bordercolor": "#E5E7EB",
        "borderwidth": 1,
    }]

    updatemenus = [{
        "type": "buttons",
        "showactive": False,
        "y": 1.06, "x": 0.01, "xanchor": "left", "yanchor": "top",
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

    # Country labels (full Myanmar view only)
    apply_tight_geos(fig, active_geo, height=580)

    if not store.get("region"):
        _add_country_labels(fig)

    fig.update_layout(
        margin=dict(l=0, r=60, t=52, b=100),
        title=dict(
            text=f"<b>{labels[0] if labels else ''}</b>",
            x=0.5, y=0.97, font=dict(size=13, color="#475569"),
        ),
        sliders=sliders,
        updatemenus=updatemenus,
    )
    return fig


def _build_choropleth(store: dict, geo: dict, metric: str = "events") -> go.Figure:
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
        colorbar=dict(title=cb_title, thickness=12, len=0.5, x=1.01),
    ))

    apply_tight_geos(fig, active_geo, height=530)

    if not store.get("region"):
        _add_country_labels(fig)

    fig.update_layout(
        margin=dict(l=0, r=60, t=36, b=4),
        title=dict(text="<b>Selected Period</b>", x=0.5, y=0.97,
                   font=dict(size=13, color="#475569")),
    )
    return fig


# ── Monthly trend chart ────────────────────────────────────────────────────────

def _build_trend(monthly_df, start_month, end_month, region, key_events) -> go.Figure:
    current_month = datetime.date.today().strftime("%Y-%m")
    df = monthly_df.copy()
    if start_month: df = df[df["month"] >= start_month]
    if end_month:   df = df[df["month"] <= end_month]
    if region:      df = df[df["admin1"] == region]
    if key_events:  df = df[df["key_event"].isin(key_events)]
    if df.empty:    return _empty_fig(height=240)

    if key_events and len(key_events) == 1:
        label = key_events[0]
        color = KEY_EVENT_COLORS.get(label, "#94a3b8")
    else:
        label = "Total Events"
        color = "#2563eb"

    monthly = (
        df.groupby("month")["events"].sum()
        .reset_index().sort_values("month")
    )
    monthly["month_dt"] = pd.to_datetime(monthly["month"] + "-01")

    complete = monthly[monthly["month"] < current_month]
    partial  = monthly[monthly["month"] == current_month]

    x_min = monthly["month_dt"].min() if not monthly.empty else None
    x_max = monthly["month_dt"].max() if not monthly.empty else None

    fig = go.Figure()
    if not complete.empty:
        fig.add_trace(go.Scatter(
            x=complete["month_dt"], y=complete["events"],
            mode="lines", name=label,
            fill="tozeroy", fillcolor=_hex_rgba(color, 0.10),
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:,}}<extra></extra>",
        ))
    if not partial.empty:
        fig.add_trace(go.Scatter(
            x=partial["month_dt"], y=partial["events"],
            mode="markers", name=label,
            showlegend=complete.empty,
            marker=dict(color=color, size=9, symbol="circle-open",
                        line=dict(width=2, color=color)),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:,}} (partial)<extra></extra>",
        ))

    # Milestone vertical lines
    if x_min is not None:
        for date_str, name in MILESTONES:
            mdt = pd.Timestamp(date_str)
            if x_min <= mdt <= (x_max + pd.Timedelta(days=90)):
                fig.add_shape(
                    type="line",
                    x0=mdt, x1=mdt, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(dash="dash", color="#888888", width=1),
                )
                fig.add_annotation(
                    x=mdt, y=0.97, xref="x", yref="paper",
                    text=name, textangle=-90, showarrow=False,
                    font=dict(size=9, color="#6B7280"),
                    xanchor="left", yanchor="top",
                )

    fig.update_layout(
        height=240,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=8, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickformat="%b %Y", tickangle=-30,
                   tickfont=dict(size=10), linecolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=False,
                   tickfont=dict(size=10), title=None),
        hovermode="x unified",
    )
    return fig


# ── Key Activity Types bar chart ──────────────────────────────────────────────

def _build_activity_bar(monthly_df, start_month, end_month, region, key_events) -> go.Figure:
    df = monthly_df.copy()
    if start_month: df = df[df["month"] >= start_month]
    if end_month:   df = df[df["month"] <= end_month]
    if region:      df = df[df["admin1"] == region]
    if key_events:  df = df[df["key_event"].isin(key_events)]
    if df.empty:    return _empty_fig("No data", height=220)

    counts = (
        df.groupby("key_event")["events"].sum()
        .reindex(KEY_EVENT_ORDER).fillna(0)
    )
    counts = counts[counts > 0].sort_values()
    if counts.empty: return _empty_fig("No data", height=220)

    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color="#2563eb"),
        hovertemplate="<b>%{y}</b><br>%{x:,} events<extra></extra>",
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=16, t=4, b=4),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=10), title=None),
        yaxis=dict(tickfont=dict(size=11), title=None, automargin=True),
    )
    return fig


# ── Highest Events helper ─────────────────────────────────────────────────────

def _highest_events(monthly_df, start_month, end_month, region, key_events):
    df = monthly_df.copy()
    if start_month: df = df[df["month"] >= start_month]
    if end_month:   df = df[df["month"] <= end_month]
    if key_events:  df = df[df["key_event"].isin(key_events)]
    if df.empty:    return "—", "—"
    if region:
        return region, f"{int(df[df['admin1'] == region]['events'].sum()):,}"
    by_r = df.groupby("admin1")["events"].sum().sort_values(ascending=False)
    return by_r.index[0], f"{int(by_r.iloc[0]):,}"


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    meta        = _get_defaults()
    df          = load_acled_main()
    monthly_df  = load_monthly_township()
    geo         = load_geojson()
    latest_str  = meta["latest_str"]
    start_month = meta["start_month"]
    end_month   = meta["end_month"]

    admin1_opts  = sorted(df["admin1"].dropna().unique())
    key_evt_opts = [k for k in KEY_EVENT_ORDER if k in df["key_event"].unique()]
    month_opts   = _month_opts(meta["start_val"], meta["end_val"])

    default_store = {
        "start_month":  start_month,
        "end_month":    end_month,
        "region":       None,
        "key_events":   None,
        "preset_label": None,
    }

    # ── Pre-build initial charts (data is lru_cached — fast) ──────────────────
    store      = _build_store(monthly_df, geo, start_month, end_month, None, None, "time_range")
    init_map   = _build_choropleth(store, geo, "events") if store else _empty_fig()
    init_trend = _build_trend(monthly_df, start_month, end_month, None, None)
    init_bar   = _build_activity_bar(monthly_df, start_month, end_month, None, None)

    # KPIs
    ac = df.copy()
    ac["month"] = ac["event_date"].dt.to_period("M").astype(str)
    ac = ac[(ac["month"] >= start_month) & (ac["month"] <= end_month)]
    init_conflicts   = _fmt(len(ac))
    init_fatalities  = _fmt(int(ac["fatalities"].sum()))
    h_region, h_count = _highest_events(monthly_df, start_month, end_month, None, None)
    init_summary     = _build_filter_summary(start_month, end_month, None, None, "time_range")

    return html.Div([

        # ── hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="ov-applied-filters", data=default_store),
        dcc.Store(id="ov-mode", data="time_range"),

        # ── page header ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H4("Overview", className="page-title"),
                html.Div("Explore conflict incidents across Myanmar",
                         className="page-subtitle"),
            ], className="page-header-left"),
            html.Div(f"Last Updated: {latest_str}", className="page-last-updated"),
        ], className="page-header"),

        # ── filter card ────────────────────────────────────────────────────────
        html.Div([
            html.Div([

                # Time Range: presets + inline From → To (one horizontal line)
                html.Div([
                    html.Label("Time Range", className="filter-label"),
                    html.Div([
                        html.Button("Last 7 days",    id="ov-btn-7d",  n_clicks=0, className="quick-btn"),
                        html.Button("Last 30 days",   id="ov-btn-30d", n_clicks=0, className="quick-btn"),
                        html.Button("Since Feb 2021", id="ov-btn-all", n_clicks=0, className="quick-btn"),
                        html.Div(className="time-sep"),
                        dcc.Dropdown(
                            id="ov-from-month",
                            options=month_opts,
                            value=start_month,
                            clearable=False,
                            className="month-dropdown",
                        ),
                        html.Span("→", className="range-arrow"),
                        dcc.Dropdown(
                            id="ov-to-month",
                            options=month_opts,
                            value=end_month,
                            clearable=False,
                            className="month-dropdown",
                        ),
                    ], className="time-range-controls"),
                ], className="filter-group filter-group--time"),

                html.Div([
                    html.Label("Region", className="filter-label"),
                    dcc.Dropdown(id="ov-region",
                                 options=[{"label": r, "value": r} for r in admin1_opts],
                                 multi=False, placeholder="All Regions", clearable=True),
                ], className="filter-group"),

                html.Div([
                    html.Label("Conflict Type", className="filter-label"),
                    dcc.Dropdown(id="ov-key-event",
                                 options=[{"label": k, "value": k} for k in key_evt_opts],
                                 multi=False, placeholder="All Types", clearable=True),
                ], className="filter-group filter-group--wide"),

                html.Div([
                    html.Label("\u00a0", className="filter-label"),
                    html.Div([
                        html.Button("Apply", id="ov-apply-btn",
                                    n_clicks=0, className="btn-apply"),
                        html.Button("Reset", id="ov-reset-btn",
                                    n_clicks=0, className="btn-reset"),
                    ], className="btn-group"),
                ], className="filter-group filter-group--btns"),

            ], className="filter-controls"),
        ], className="filter-card"),

        # ── filter summary ─────────────────────────────────────────────────────
        html.Div(init_summary, id="ov-filter-summary", className="filter-summary"),

        # ── two-column body ────────────────────────────────────────────────────
        html.Div([

            # ── LEFT: hero map ─────────────────────────────────────────────────
            html.Div([
                html.Div([

                    html.Div([
                        html.Div("Conflict Incidents in Myanmar", className="card-title"),
                        dcc.RadioItems(
                            id="ov-metric",
                            options=[
                                {"label": "Events",     "value": "events"},
                                {"label": "Fatalities", "value": "fatalities"},
                            ],
                            value="events",
                            inline=True,
                            className="metric-toggle",
                        ),
                    ], className="dash-card-head",
                       style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "flexWrap": "wrap", "gap": "8px"}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div("Period Total", className="mode-card-title"),
                                html.Div("Cumulative events for selected time range",
                                         className="mode-card-desc"),
                            ]),
                        ], id="ov-mode-total-btn", n_clicks=0,
                           className="mode-card mode-card--active"),
                        html.Div([
                            html.Div([
                                html.Div("Quarterly Animation", className="mode-card-title"),
                                html.Div("Watch conflict spread quarter by quarter",
                                         className="mode-card-desc"),
                            ]),
                        ], id="ov-mode-anim-btn", n_clicks=0,
                           className="mode-card"),
                    ], className="mode-toggle-cards"),

                    dcc.Loading(
                        dcc.Graph(id="ov-map", figure=init_map,
                                  config={**_chart_config("myanmar_conflict_map"),
                                          "displayModeBar": True}),
                        type="dot", color="#2563eb",
                    ),

                    html.Div([
                        html.Div("Highest Events", className="highest-label"),
                        html.Div([
                            html.Div(h_region, id="ov-highest-region", className="highest-region"),
                            html.Div(h_count,  id="ov-highest-count",  className="highest-count"),
                        ], className="highest-values"),
                    ], className="highest-events"),

                ], className="dash-card"),
            ], className="col-map"),

            # ── RIGHT: charts stack ────────────────────────────────────────────
            html.Div([

                html.Div([
                    html.Div([
                        html.Div("Total Conflicts",             className="kpi-label"),
                        html.Div(init_conflicts, id="ov-kpi-conflicts", className="kpi-value"),
                        html.Div("events recorded",             className="kpi-sub"),
                    ], className="kpi-card kpi-accent-blue"),
                    html.Div([
                        html.Div("Total Fatalities",            className="kpi-label"),
                        html.Div(init_fatalities, id="ov-kpi-fatalities", className="kpi-value"),
                        html.Div("deaths recorded",             className="kpi-sub"),
                    ], className="kpi-card kpi-accent-red"),
                ], className="kpis-row kpis-row--2"),

                html.Div([
                    html.Div([
                        html.Div("Monthly Trend of Events", className="card-title"),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(id="ov-trend", figure=init_trend,
                                  config=_chart_config("myanmar_monthly_trend")),
                        type="dot", color="#2563eb",
                    ),
                ], className="dash-card"),

                html.Div([
                    html.Div([
                        html.Div("Key Activity Types", className="card-title"),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(id="ov-activity-bar", figure=init_bar,
                                  config=_chart_config("myanmar_activity_types")),
                        type="dot", color="#2563eb",
                    ),
                ], className="dash-card"),

            ], className="col-charts"),

        ], className="page-body"),

        # ── data disclaimer ────────────────────────────────────────────────────
        html.Div([
            "Source: ",
            html.A("ACLED", href="https://acleddata.com", target="_blank",
                   style={"color": "inherit", "textDecoration": "underline"}),
            " (Armed Conflict Location & Event Data Project). Our team has reviewed "
            "and recoded events using local field knowledge. Displayed figures may "
            "differ from official sources or on-the-ground reports.",
        ], className="data-disclaimer"),

    ], className="page-wrap")


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# 1. Mode card buttons → update mode store + button styles
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
        return "animated", "mode-card", "mode-card mode-card--active"
    return "time_range", "mode-card mode-card--active", "mode-card"


# 2. Quick date preset buttons → apply immediately + sync dropdowns
@callback(
    Output("ov-applied-filters", "data", allow_duplicate=True),
    Output("ov-from-month",      "value",              allow_duplicate=True),
    Output("ov-to-month",        "value",              allow_duplicate=True),
    Input("ov-btn-7d",           "n_clicks"),
    Input("ov-btn-30d",          "n_clicks"),
    Input("ov-btn-all",          "n_clicks"),
    State("ov-region",           "value"),
    State("ov-key-event",        "value"),
    prevent_initial_call=True,
)
def set_quick_dates(n7, n30, nall, region, key_events):
    max_dt = load_acled_main()["event_date"].max()
    if ctx.triggered_id == "ov-btn-7d":
        start_dt = max_dt - pd.Timedelta(days=7)
        label = "Last 7 days"
    elif ctx.triggered_id == "ov-btn-30d":
        start_dt = max_dt - pd.Timedelta(days=30)
        label = "Last 30 days"
    else:
        start_dt = pd.Timestamp("2021-02-01")
        label = "Since Feb 2021"
    if isinstance(key_events, str):
        key_events = [key_events]
    start_m = start_dt.strftime("%Y-%m")
    end_m   = max_dt.strftime("%Y-%m")
    return (
        {"start_month": start_m, "end_month": end_m,
         "region": region, "key_events": key_events,
         "preset_label": label},
        start_m,
        end_m,
    )


# 4. Apply button → save filter state to store (reads From/To dropdowns)
@callback(
    Output("ov-applied-filters", "data"),
    Input("ov-apply-btn",        "n_clicks"),
    State("ov-from-month",       "value"),
    State("ov-to-month",         "value"),
    State("ov-region",           "value"),
    State("ov-key-event",        "value"),
    prevent_initial_call=True,
)
def apply_filters(n, from_month, to_month, region, key_events):
    d = _get_defaults()
    start_m = from_month or d["start_month"]
    end_m   = to_month   or d["end_month"]
    if isinstance(key_events, str):
        key_events = [key_events]
    return {"start_month": start_m, "end_month": end_m,
            "region": region, "key_events": key_events,
            "preset_label": None}


# 5. Reset button → clear all filters and restore dropdowns to full range
@callback(
    Output("ov-region",          "value"),
    Output("ov-key-event",       "value"),
    Output("ov-from-month",      "value",              allow_duplicate=True),
    Output("ov-to-month",        "value",              allow_duplicate=True),
    Output("ov-applied-filters", "data", allow_duplicate=True),
    Input("ov-reset-btn",        "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n):
    d = _get_defaults()
    defaults = {"start_month": d["start_month"], "end_month": d["end_month"],
                "region": None, "key_events": None, "preset_label": None}
    return None, None, d["start_month"], d["end_month"], defaults


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
    prevent_initial_call=True,
)
def update_charts(applied, mode, metric):
    if not applied:
        raise PreventUpdate

    start_month  = applied.get("start_month")
    end_month    = applied.get("end_month")
    region       = applied.get("region")
    key_events   = applied.get("key_events")
    preset_label = applied.get("preset_label")
    if isinstance(key_events, str):
        key_events = [key_events]

    mode   = mode   or "time_range"
    metric = metric or "events"

    monthly_df = load_monthly_township()
    acled      = load_acled_main()
    geo        = load_geojson()

    # KPIs
    ac = acled.copy()
    ac["month"] = ac["event_date"].dt.to_period("M").astype(str)
    if start_month: ac = ac[ac["month"] >= start_month]
    if end_month:   ac = ac[ac["month"] <= end_month]
    if region:      ac = ac[ac["admin1"] == region]
    if key_events:  ac = ac[ac["key_event"].isin(key_events)]
    n_conflicts  = _fmt(len(ac))
    n_fatalities = _fmt(int(ac["fatalities"].sum()))

    filter_summary = _build_filter_summary(start_month, end_month, region, key_events,
                                           mode, preset_label)
    store = _build_store(monthly_df, geo, start_month, end_month, region, key_events, mode)

    if store is None:
        empty = _empty_fig("No data for this selection")
        return (empty, n_conflicts, n_fatalities, filter_summary,
                _empty_fig(height=240), _empty_fig("No data", height=220), "—", "—")

    if mode == "animated":
        fig_map = _build_animated_choropleth(store, geo, metric)
    else:
        fig_map = _build_choropleth(store, geo, metric)

    fig_trend = _build_trend(monthly_df, start_month, end_month, region, key_events)
    fig_bar   = _build_activity_bar(monthly_df, start_month, end_month, region, key_events)
    h_region, h_count = _highest_events(monthly_df, start_month, end_month, region, key_events)

    return fig_map, n_conflicts, n_fatalities, filter_summary, fig_trend, fig_bar, h_region, h_count
