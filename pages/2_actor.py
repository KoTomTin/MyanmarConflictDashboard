"""
Page 2 — Actor Analysis  (design v4)

Same card / grid design as Overview.
Time range: quick presets (Last 7 / 30 / Since Feb 2021) + inline From→To dropdowns.
Map: Period Total | Quarterly Animation — Plotly native frames (no lag, same as Overview).
Charts: Alliance network table + monthly engagement trend (with export button).
"""
import re
import datetime
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate

from components.loaders   import (load_acled_main, load_actor_level,
                                   load_ally_pairs, load_geojson)
from components.colors    import SEQUENTIAL_BLUES_ZERO_GREY
from components.map_utils import apply_tight_geos, filter_geo_by_property


DEFAULT_ACTOR      = "Myanmar Military Regime"
ARMED_EVENT_TYPES  = frozenset(["Ground-based attack", "Air attack", "Drone attack"])

# (date, hover title, short pill, line color, description)
MILESTONES = [
    ("2021-02-01", "Feb 1, 2021 – Military Coup",           "Coup",       "#dc2626",
     "The Myanmar military seized power and arrested elected leaders, sparking nationwide protests."),
    ("2021-09-07", "Sept 7, 2021 – People's Defensive War", "NUG War",    "#d97706",
     "The NUG called for armed resistance against the military, escalating the conflict."),
    ("2023-10-27", "Oct 27, 2023 – Operation 1027",         "Op. 1027",   "#7c3aed",
     "Rebel alliance launched a major offensive, capturing key territory in northern Shan State."),
    ("2025-03-28", "Mar 28, 2025 – 7.7 Earthquake",         "Earthquake", "#0891b2",
     "A major earthquake near Mandalay caused heavy destruction and worsened the crisis."),
    ("2025-12-28", "Dec 28, 2025 – Junta Elections",        "Elections",  "#6b7280",
     "Military held phased elections in controlled areas; widely rejected as illegitimate."),
]

# Neighbouring country labels — same positions as Overview
NEIGHBOR_LABELS = [
    ("Bangladesh", 22.0,  90.8),
    ("India",       27.8,  94.5),
    ("China",       25.5, 102.0),
    ("Laos",        22.0, 102.8),
    ("Thailand",    14.5, 101.5),
]


# ── Module-level defaults ──────────────────────────────────────────────────────

def _get_defaults() -> dict:
    df = load_acled_main()
    return {
        "start_val":  df["event_date"].min().strftime("%Y-%m"),
        "end_val":    df["event_date"].max().strftime("%Y-%m"),
        "latest_str": df["event_date"].max().strftime("%d %b %Y"),
    }


# ── Generic helpers ────────────────────────────────────────────────────────────

def _fmt(n) -> str:
    return f"{int(n):,}"


def _empty_fig(msg="No data for this selection", height=460):
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


def _is_generic_civilian(name: str) -> bool:
    return bool(re.match(r"^civilians?\s*\(", name.strip(), re.IGNORECASE))


def _is_excluded_actor(name: str) -> bool:
    return _is_generic_civilian(name) or "unidentified" in name.lower()


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


def _q95(series: pd.Series) -> int:
    s = series[series > 0]
    if s.empty:
        return 1
    return max(int(s.quantile(0.95)) if len(s) >= 5 else int(s.max()), 1)


def _chart_config(filename: str) -> dict:
    """Modebar config: camera download button on hover."""
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


# ── Filter summary ─────────────────────────────────────────────────────────────

def _build_filter_summary(start_month, end_month, region, actor_name,
                          mode: str = "time_range",
                          preset_label: str | None = None) -> str:
    def fmt_m(m):
        try:
            return pd.Timestamp(m + "-01").strftime("%b %Y")
        except Exception:
            return m or ""
    geo = f"in {region}" if region else "across Myanmar"
    if mode == "animated":
        return (f"Animating: {actor_name} {geo} · "
                f"Quarter by quarter from {fmt_m(start_month)}")
    if preset_label:
        return f"Showing: {actor_name} {geo} · {preset_label}"
    return f"Showing: {actor_name} {geo} · {fmt_m(start_month)} – {fmt_m(end_month)}"


# ── Actor dropdown ─────────────────────────────────────────────────────────────

def _actor_options(actor_level: pd.DataFrame) -> list[dict]:
    al = actor_level[~actor_level["actor_name"].apply(_is_excluded_actor)]
    counts = (
        al.groupby("actor_name")["event_id_cnty"]
        .nunique().sort_values(ascending=False).reset_index()
    )
    return [{"label": f"{r.actor_name}  ({r.event_id_cnty:,})", "value": r.actor_name}
            for _, r in counts.iterrows()]


# ── Store builder ──────────────────────────────────────────────────────────────

def _armed_ids(acled: pd.DataFrame) -> frozenset:
    """Return event IDs for ground-based, air, and drone attacks only."""
    return frozenset(acled[acled["key_event"].isin(ARMED_EVENT_TYPES)]["event_id_cnty"])


def _build_store(actor_level, geo, actor_name, start_month, end_month,
                 region, acled, mode) -> dict | None:
    ids = _armed_ids(acled)
    al = actor_level[
        (actor_level["actor_name"] == actor_name) &
        (actor_level["event_id_cnty"].isin(ids))
    ].copy()
    al["event_date"] = pd.to_datetime(al["event_date"], errors="coerce")
    al["month"] = al["event_date"].dt.to_period("M").astype(str)

    if start_month: al = al[al["month"] >= start_month]
    if end_month:   al = al[al["month"] <= end_month]
    if region:
        valid_pcodes = set(acled[acled["admin1"] == region]["Tsp_Pcode"].dropna().unique())
        al = al[al["Tsp_Pcode"].isin(valid_pcodes)]
    if al.empty:    return None

    active_geo = filter_geo_by_property(geo, "ST", region) if region else geo
    pcodes     = _geo_pcodes(active_geo)
    text       = _tsp_text(active_geo)
    n          = len(pcodes)
    pcode_idx  = {p: i for i, p in enumerate(pcodes)}

    agg = (
        al.groupby(["Tsp_Pcode", "month"])["event_id_cnty"]
        .nunique().reset_index(name="events")
    )

    if mode == "animated":
        agg["quarter"] = agg["month"].apply(_month_to_quarter)
        qagg = agg.groupby(["Tsp_Pcode", "quarter"])["events"].sum().reset_index()
        all_quarters = sorted(qagg["quarter"].unique())
        matrix = []
        for q in all_quarters:
            z   = [0] * n
            sub = qagg[qagg["quarter"] == q]
            for _, row in sub.iterrows():
                i = pcode_idx.get(row["Tsp_Pcode"])
                if i is not None:
                    z[i] = int(row["events"])
            matrix.append(z)
        return {
            "mode": "animated", "frames": all_quarters,
            "frame_labels": [_quarter_label(q) for q in all_quarters],
            "matrix": matrix, "max_val": _q95(qagg["events"]),
            "pcodes": pcodes, "text": text, "region": region,
        }
    else:
        total = agg.groupby("Tsp_Pcode")["events"].sum().reset_index()
        z = [0] * n
        for _, row in total.iterrows():
            i = pcode_idx.get(row["Tsp_Pcode"])
            if i is not None:
                z[i] = int(row["events"])
        return {
            "mode": "time_range", "z": z, "max_val": _q95(pd.Series(z)),
            "pcodes": pcodes, "text": text, "region": region,
        }


# ── Neighbouring country labels ────────────────────────────────────────────────

def _add_country_labels(fig: go.Figure) -> None:
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
    fig.update_geos(lonaxis_range=[89.5, 104.0], lataxis_range=[8.0, 30.5])


# ── Static choropleth ──────────────────────────────────────────────────────────

def _build_choropleth(store: dict, geo: dict) -> go.Figure:
    active_geo = filter_geo_by_property(geo, "ST", store["region"]) if store.get("region") else geo
    fig = go.Figure(go.Choropleth(
        geojson=active_geo,
        locations=store["pcodes"],
        z=store["z"], text=store["text"],
        featureidkey="properties.TS_PCODE",
        colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
        zmin=0, zmax=store["max_val"],
        marker_line_width=0.3, marker_line_color="#ffffff",
        hovertemplate="<b>%{text}</b><br>Events: %{z:,}<extra></extra>",
        colorbar=dict(title="Events", thickness=12, len=0.5, x=1.01),
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


# ── Animated choropleth (Plotly native — zero server round-trips) ──────────────

def _build_animated_choropleth(store: dict, geo: dict) -> go.Figure:
    active_geo = filter_geo_by_property(geo, "ST", store["region"]) if store.get("region") else geo
    matrix     = store["matrix"]
    mv         = store["max_val"]
    labels     = store["frame_labels"]
    n_frames   = len(labels)
    initial_z  = matrix[0] if n_frames > 0 else [0] * len(store["pcodes"])

    base_trace = go.Choropleth(
        geojson=active_geo,
        locations=store["pcodes"],
        z=initial_z, text=store["text"],
        featureidkey="properties.TS_PCODE",
        colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
        zmin=0, zmax=mv,
        marker_line_width=0.3, marker_line_color="#ffffff",
        hovertemplate="<b>%{text}</b><br>Events: %{z:,}<extra></extra>",
        colorbar=dict(title="Events", thickness=12, len=0.4, x=1.01),
    )

    frames = [
        go.Frame(
            data=[go.Choropleth(z=z)],
            traces=[0],
            name=str(i),
            layout=go.Layout(title_text=f"<b>{lbl}</b>"),
        )
        for i, (z, lbl) in enumerate(zip(matrix, labels))
    ]

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


# ── Trend chart ────────────────────────────────────────────────────────────────

def _build_trend(al: pd.DataFrame) -> go.Figure:
    if al.empty:
        return _empty_fig(height=200)
    current_month = datetime.date.today().strftime("%Y-%m")
    al = al.copy()
    al["event_date"] = pd.to_datetime(al["event_date"], errors="coerce")
    al["month_str"] = al["event_date"].dt.to_period("M").astype(str)

    monthly = (
        al.groupby(["month_str", "type2"], observed=True)["event_id_cnty"]
        .nunique().reset_index(name="events").sort_values("month_str")
    )
    monthly["month_dt"] = pd.to_datetime(monthly["month_str"] + "-01")
    complete = monthly[monthly["month_str"] < current_month]
    partial  = monthly[monthly["month_str"] == current_month]

    color_map = {"offend": "#ef4444", "being_offended": "#3b82f6"}
    label_map = {"offend": "Offending", "being_offended": "Defending"}

    fig = go.Figure()
    for role in monthly["type2"].unique():
        color = color_map.get(role, "#94a3b8")
        label = label_map.get(role, role)
        grp_c = complete[complete["type2"] == role]
        grp_p = partial[partial["type2"] == role]

        if not grp_c.empty:
            fig.add_trace(go.Scatter(
                x=grp_c["month_dt"], y=grp_c["events"],
                mode="lines", name=label,
                line=dict(color=color, width=2.2),
                legendgroup=role, showlegend=True,
                hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:,}}<extra></extra>",
            ))
            if not grp_p.empty:
                fig.add_trace(go.Scatter(
                    x=grp_p["month_dt"], y=grp_p["events"],
                    mode="markers", name=label,
                    marker=dict(color=color, size=9, symbol="circle-open",
                                line=dict(width=2, color=color)),
                    legendgroup=role, showlegend=False,
                    hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:,}} (partial)<extra></extra>",
                ))
        elif not grp_p.empty:
            fig.add_trace(go.Scatter(
                x=grp_p["month_dt"], y=grp_p["events"],
                mode="markers", name=label,
                marker=dict(color=color, size=9, symbol="circle-open",
                            line=dict(width=2, color=color)),
                legendgroup=role, showlegend=True,
                hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:,}} (partial)<extra></extra>",
            ))

    # Milestone vertical lines — color-coded pills + hover diamonds
    all_dates = monthly["month_dt"]
    if not all_dates.empty:
        x_min = all_dates.min()
        x_max = all_dates.max()
        for date_str, title, short_name, color, desc in MILESTONES:
            mdt = pd.Timestamp(date_str)
            if x_min <= mdt <= (x_max + pd.Timedelta(days=90)):
                fig.add_shape(
                    type="line",
                    x0=mdt, x1=mdt, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color=color, width=1.5, dash="dot"),
                    opacity=0.55,
                )
                fig.add_annotation(
                    x=mdt, y=1.0, xref="x", yref="paper",
                    text=f"<b>{short_name}</b>",
                    showarrow=False,
                    font=dict(size=8, color=color),
                    xanchor="center", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=color, borderwidth=1, borderpad=2,
                )
                # Small diamond on hidden y2 axis — hover target with full description
                fig.add_trace(go.Scatter(
                    x=[mdt], y=[0.5], yaxis="y2",
                    mode="markers",
                    marker=dict(color=color, size=7, symbol="diamond",
                                line=dict(color="white", width=1), opacity=0.75),
                    showlegend=False, name="",
                    hovertemplate=(
                        f"<b>{title}</b><br>"
                        f"<span style='color:#6b7280'>{desc}</span>"
                        f"<extra></extra>"
                    ),
                ))

    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=28, b=4),
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickformat="%b %Y", tickangle=-30,
                   tickfont=dict(size=10), linecolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=True,
                   zerolinecolor="#e5e7eb", zerolinewidth=1,
                   rangemode="tozero", autorange=True,
                   tickfont=dict(size=10), title=None),
        hovermode="x unified",
    )
    return fig


# ── Association chart (replaces plain table) ───────────────────────────────────

def _build_alliance_chart(ally_pairs, actor_name, valid_ids):
    ap = ally_pairs[
        (ally_pairs["actor_name"] == actor_name) &
        (ally_pairs["event_id_cnty"].isin(valid_ids))
    ]
    if ap.empty:
        return html.Div("No association data for this selection.", className="table-empty")

    ap = ap[~ap["ally_name"].apply(_is_excluded_actor)]
    if ap.empty:
        return html.Div("No named associated actors in this selection.", className="table-empty")

    tbl = (
        ap.groupby("ally_name")["event_id_cnty"]
        .nunique().reset_index(name="events")
        .sort_values("events", ascending=False)   # highest first (leftmost)
        .head(15)
    )

    fig = go.Figure(go.Bar(
        x=tbl["ally_name"],
        y=tbl["events"],
        marker=dict(color="#3b82f6"),
        text=[f"{v:,}" for v in tbl["events"]],
        textposition="outside",
        textfont=dict(size=9, color="#6B7280"),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{y:,} shared events<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=8, t=20, b=4),
        xaxis=dict(tickfont=dict(size=9), title=None, tickangle=-40, automargin=True),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=9),
                   title=None, rangemode="tozero"),
    )
    return dcc.Graph(figure=fig, config=_chart_config("actor_associated_actors"))


# ── Highest events ─────────────────────────────────────────────────────────────

def _highest_events(al: pd.DataFrame, acled: pd.DataFrame, region):
    if al.empty:
        return "—", "—"
    if region:
        valid_pcodes = set(acled[acled["admin1"] == region]["Tsp_Pcode"].dropna().unique())
        al = al[al["Tsp_Pcode"].isin(valid_pcodes)]
        return region, f"{al['event_id_cnty'].nunique():,}"
    by_pcode = al.groupby("Tsp_Pcode")["event_id_cnty"].nunique()
    if by_pcode.empty:
        return "—", "—"
    top_pcode = by_pcode.idxmax()
    match = acled[acled["Tsp_Pcode"] == top_pcode]["admin1"]
    top_name = match.iloc[0] if not match.empty else top_pcode
    return top_name, f"{int(by_pcode.max()):,}"


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    meta        = _get_defaults()
    acled       = load_acled_main()
    actor_level = load_actor_level()
    ally_pairs  = load_ally_pairs()
    geo         = load_geojson()

    start_val   = meta["start_val"]
    end_val     = meta["end_val"]
    latest_str  = meta["latest_str"]

    actor_opts  = _actor_options(actor_level)
    admin1_opts = sorted(acled["admin1"].dropna().unique())

    default_store = {
        "start_month": start_val, "end_month": end_val,
        "region": None, "actor_name": DEFAULT_ACTOR, "preset_label": None,
    }

    # ── Pre-build initial charts ───────────────────────────────────────────────
    armed_ids  = _armed_ids(acled)
    al_default = actor_level[
        (actor_level["actor_name"] == DEFAULT_ACTOR) &
        (actor_level["event_id_cnty"].isin(armed_ids))
    ].copy()
    al_default["event_date"] = pd.to_datetime(al_default["event_date"], errors="coerce")
    al_default["month"] = al_default["event_date"].dt.to_period("M").astype(str)

    store        = _build_store(actor_level, geo, DEFAULT_ACTOR, start_val, end_val,
                                None, acled, "time_range")
    init_map     = _build_choropleth(store, geo) if store else _empty_fig()
    init_trend   = _build_trend(al_default)

    valid_ids    = set(al_default["event_id_cnty"].unique())
    n_townships  = al_default["Tsp_Pcode"].nunique()
    _off_ids     = set(al_default[al_default["type2"] == "offend"]["event_id_cnty"])
    _def_ids     = set(al_default[al_default["type2"] == "being_offended"]["event_id_cnty"]) - _off_ids
    n_offenses   = len(_off_ids)
    n_defenses   = len(_def_ids)
    h_region, h_count = _highest_events(al_default, acled, None)
    init_alliance = _build_alliance_chart(ally_pairs, DEFAULT_ACTOR, valid_ids)
    init_summary  = _build_filter_summary(start_val, end_val, None, DEFAULT_ACTOR)

    return html.Div([

        # ── hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="ac-applied-filters", data=default_store),
        dcc.Store(id="ac-mode", data="time_range"),

        # ── page header ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H4("Actor Analysis", className="page-title"),
                html.Div("Geographic footprint · alliance network · engagement trend",
                         className="page-subtitle"),
                html.Div("Armed conflict only — ground-based, air & drone attacks",
                         className="page-subtitle page-subtitle--armed"),
            ], className="page-header-left"),
            html.Div(f"Last Updated: {latest_str}", className="page-last-updated"),
        ], className="page-header"),

        # ── actor banner ────────────────────────────────────────────────────────
        html.Div([
            html.Span("Analyzing:", className="actor-banner-label"),
            html.Span(DEFAULT_ACTOR, id="ac-actor-banner-name",
                      className="actor-banner-name"),
        ], className="actor-banner"),

        # ── filter card ────────────────────────────────────────────────────────
        html.Div([
            html.Div([

                # Time Range: quick presets + date-card pickers
                html.Div([
                    html.Label("Time Range", className="filter-label"),
                    html.Div([
                        html.Button("Last 7 days",    id="ac-btn-7d",  n_clicks=0, className="quick-btn"),
                        html.Button("Last 30 days",   id="ac-btn-30d", n_clicks=0, className="quick-btn"),
                        html.Button("Since Feb 2021", id="ac-btn-all", n_clicks=0, className="quick-btn"),
                    ], className="quick-btn-row"),
                    html.Div([
                        html.Div([
                            html.Div("Date From", className="date-card-header"),
                            dcc.DatePickerSingle(
                                id="ac-from-date",
                                date=start_val,
                                display_format="DD/MM/YYYY",
                                first_day_of_week=1,
                                className="date-picker-single",
                            ),
                        ], className="date-card"),
                        html.Div([
                            html.Div("Date To", className="date-card-header"),
                            dcc.DatePickerSingle(
                                id="ac-to-date",
                                date=end_val,
                                display_format="DD/MM/YYYY",
                                first_day_of_week=1,
                                className="date-picker-single",
                            ),
                        ], className="date-card"),
                    ], className="date-card-group"),
                ], className="filter-group filter-group--datepicker"),

                html.Div([
                    html.Label("Region", className="filter-label"),
                    dcc.Dropdown(id="ac-region",
                                 options=[{"label": r, "value": r} for r in admin1_opts],
                                 multi=False, placeholder="All Regions", clearable=True),
                ], className="filter-group"),

                html.Div([
                    html.Label("Actor", className="filter-label"),
                    dcc.Dropdown(id="ac-actor", options=actor_opts,
                                 value=DEFAULT_ACTOR, clearable=False, searchable=True),
                ], className="filter-group filter-group--wide"),

                html.Div([
                    html.Label("\u00a0", className="filter-label"),
                    html.Div([
                        html.Button("Apply", id="ac-apply-btn",
                                    n_clicks=0, className="btn-apply"),
                        html.Button("Reset", id="ac-reset-btn",
                                    n_clicks=0, className="btn-reset"),
                    ], className="btn-group"),
                ], className="filter-group filter-group--btns"),

            ], className="filter-controls"),
        ], className="filter-card"),

        # ── filter summary ─────────────────────────────────────────────────────
        html.Div(init_summary, id="ac-filter-summary", className="filter-summary"),

        # ── two-column body ────────────────────────────────────────────────────
        html.Div([

            # ── LEFT: hero map ─────────────────────────────────────────────────
            html.Div([
                html.Div([

                    html.Div([
                        html.Div("Geographic Footprint", className="card-title"),
                        html.Div("Point at a township · blue = high activity",
                                 className="card-subtitle"),
                    ], className="dash-card-head"),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div("Period Total", className="mode-card-title"),
                                html.Div("Cumulative events for selected time range",
                                         className="mode-card-desc"),
                            ]),
                        ], id="ac-mode-total-btn", n_clicks=0,
                           className="mode-card mode-card--active"),
                        html.Div([
                            html.Div([
                                html.Div("Quarterly Animation", className="mode-card-title"),
                                html.Div("Watch conflict spread quarter by quarter",
                                         className="mode-card-desc"),
                            ]),
                        ], id="ac-mode-anim-btn", n_clicks=0,
                           className="mode-card"),
                    ], className="mode-toggle-cards"),

                    dcc.Loading(
                        dcc.Graph(id="ac-map", figure=init_map,
                                  config={**_chart_config("actor_geographic_footprint"),
                                          "displayModeBar": True}),
                        type="dot", color="#2563eb",
                    ),

                    html.Div([
                        html.Div("Most Active", className="highest-label"),
                        html.Div([
                            html.Div(h_region, id="ac-highest-region", className="highest-region"),
                            html.Div(h_count,  id="ac-highest-count",  className="highest-count"),
                        ], className="highest-values"),
                    ], className="highest-events"),

                ], className="dash-card"),
            ], className="col-map"),

            # ── RIGHT: KPIs + alliance table + trend ───────────────────────────
            html.Div([

                html.Div([
                    html.Div([html.Div("Townships",           className="kpi-label"),
                              html.Div(_fmt(n_townships),  id="ac-kpi-townships",  className="kpi-value"),
                              html.Div("unique locations",   className="kpi-sub")],
                             className="kpi-card kpi-accent-teal"),
                    html.Div([html.Div("Offensives",          className="kpi-label"),
                              html.Div(_fmt(n_offenses),   id="ac-kpi-offenses",   className="kpi-value"),
                              html.Div("as attacker",         className="kpi-sub")],
                             className="kpi-card kpi-accent-orange"),
                    html.Div([html.Div("Defensive",           className="kpi-label"),
                              html.Div(_fmt(n_defenses),   id="ac-kpi-defenses",   className="kpi-value"),
                              html.Div("as targeted",         className="kpi-sub")],
                             className="kpi-card kpi-accent-red"),
                ], className="kpis-row kpis-row--3 kpis-compact"),

                html.Div([
                    html.Div([
                        html.Div("Associated Actors", className="card-title"),
                        html.Div("Co-involved in the same armed conflict events",
                                 className="card-subtitle"),
                    ], className="dash-card-head"),
                    html.Div(init_alliance, id="ac-alliance-table",
                             className="panel-body"),
                ], className="dash-card"),

                html.Div([
                    html.Div([
                        html.Div("Monthly Engagement Trend", className="card-title"),
                        html.Div([
                            html.Span("Offensive", style={"color": "#ef4444", "fontWeight": "600"}),
                            html.Span(" = actor initiated the attack  ·  "),
                            html.Span("Defensive", style={"color": "#3b82f6", "fontWeight": "600"}),
                            html.Span(" = actor was targeted. Recoded from ACLED by our team."),
                        ], className="card-subtitle"),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        dcc.Graph(id="ac-trend", figure=init_trend,
                                  config=_chart_config("actor_monthly_trend")),
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

# 1. Mode card buttons → mode store + button styles
@callback(
    Output("ac-mode",           "data"),
    Output("ac-mode-total-btn", "className"),
    Output("ac-mode-anim-btn",  "className"),
    Input("ac-mode-total-btn",  "n_clicks"),
    Input("ac-mode-anim-btn",   "n_clicks"),
    prevent_initial_call=True,
)
def switch_mode(n_total, n_anim):
    if ctx.triggered_id == "ac-mode-anim-btn":
        return "animated", "mode-card", "mode-card mode-card--active"
    return "time_range", "mode-card mode-card--active", "mode-card"


# 2. Quick preset buttons → apply immediately + sync date pickers
@callback(
    Output("ac-applied-filters", "data", allow_duplicate=True),
    Output("ac-from-date",       "date",               allow_duplicate=True),
    Output("ac-to-date",         "date",               allow_duplicate=True),
    Input("ac-btn-7d",           "n_clicks"),
    Input("ac-btn-30d",          "n_clicks"),
    Input("ac-btn-all",          "n_clicks"),
    State("ac-region",           "value"),
    State("ac-actor",            "value"),
    prevent_initial_call=True,
)
def set_quick_dates(n7, n30, nall, region, actor):
    max_dt = load_acled_main()["event_date"].max()
    if ctx.triggered_id == "ac-btn-7d":
        start_dt = max_dt - pd.Timedelta(days=7)
        label = "Last 7 days"
    elif ctx.triggered_id == "ac-btn-30d":
        start_dt = max_dt - pd.Timedelta(days=30)
        label = "Last 30 days"
    else:
        start_dt = pd.Timestamp("2021-02-01")
        label = "Since Feb 2021"
    start_m    = start_dt.strftime("%Y-%m")
    end_m      = max_dt.strftime("%Y-%m")
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = max_dt.strftime("%Y-%m-%d")
    return (
        {"start_month": start_m, "end_month": end_m,
         "region": region, "actor_name": actor or DEFAULT_ACTOR,
         "preset_label": label},
        start_date, end_date,
    )


# 3. Apply → update store
@callback(
    Output("ac-applied-filters", "data"),
    Input("ac-apply-btn",        "n_clicks"),
    State("ac-from-date",        "date"),
    State("ac-to-date",          "date"),
    State("ac-region",           "value"),
    State("ac-actor",            "value"),
    prevent_initial_call=True,
)
def apply_filters(n, from_date, to_date, region, actor):
    d = _get_defaults()
    start_m = (from_date or d["start_val"])[:7]
    end_m   = (to_date   or d["end_val"])[:7]
    return {"start_month": start_m, "end_month": end_m,
            "region": region, "actor_name": actor or DEFAULT_ACTOR,
            "preset_label": None}


# 4. Reset → restore date pickers + default store
@callback(
    Output("ac-region",           "value"),
    Output("ac-actor",            "value"),
    Output("ac-from-date",        "date",               allow_duplicate=True),
    Output("ac-to-date",          "date",               allow_duplicate=True),
    Output("ac-applied-filters",  "data", allow_duplicate=True),
    Input("ac-reset-btn",         "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n):
    d = _get_defaults()
    defaults = {"start_month": d["start_val"][:7], "end_month": d["end_val"][:7],
                "region": None, "actor_name": DEFAULT_ACTOR, "preset_label": None}
    return None, DEFAULT_ACTOR, d["start_val"], d["end_val"], defaults


# 5. Filters + mode → rebuild all charts
@callback(
    Output("ac-map",              "figure"),
    Output("ac-kpi-townships",    "children"),
    Output("ac-kpi-offenses",     "children"),
    Output("ac-kpi-defenses",     "children"),
    Output("ac-alliance-table",   "children"),
    Output("ac-trend",            "figure"),
    Output("ac-highest-region",   "children"),
    Output("ac-highest-count",    "children"),
    Output("ac-filter-summary",   "children"),
    Output("ac-actor-banner-name","children"),
    Input("ac-applied-filters",   "data"),
    Input("ac-mode",              "data"),
    prevent_initial_call=True,
)
def update_actor(applied, mode):
    if not applied:
        raise PreventUpdate

    start_month  = applied.get("start_month")
    end_month    = applied.get("end_month")
    region       = applied.get("region")
    actor_name   = applied.get("actor_name") or DEFAULT_ACTOR
    preset_label = applied.get("preset_label")
    mode         = mode or "time_range"

    acled       = load_acled_main()
    actor_level = load_actor_level()
    ally_pairs  = load_ally_pairs()
    geo         = load_geojson()

    # Filter to armed events only
    ids = _armed_ids(acled)
    al = actor_level[
        (actor_level["actor_name"] == actor_name) &
        (actor_level["event_id_cnty"].isin(ids))
    ].copy()
    al["event_date"] = pd.to_datetime(al["event_date"], errors="coerce")
    al["month"] = al["event_date"].dt.to_period("M").astype(str)
    if start_month: al = al[al["month"] >= start_month]
    if end_month:   al = al[al["month"] <= end_month]
    if region:
        valid_pcodes = set(acled[acled["admin1"] == region]["Tsp_Pcode"].dropna().unique())
        al = al[al["Tsp_Pcode"].isin(valid_pcodes)]

    filter_summary = _build_filter_summary(start_month, end_month, region, actor_name,
                                           mode, preset_label)
    empty_map = _empty_fig(f"No data for '{actor_name}' in this period")

    if al.empty:
        return (
            empty_map, "—", "—", "—",
            html.Div("No data.", className="table-empty"),
            _empty_fig(height=220), "—", "—", filter_summary, actor_name,
        )

    valid_ids   = set(al["event_id_cnty"].unique())
    n_townships = al["Tsp_Pcode"].nunique()
    _off_ids    = set(al[al["type2"] == "offend"]["event_id_cnty"])
    _def_ids    = set(al[al["type2"] == "being_offended"]["event_id_cnty"]) - _off_ids
    n_offenses  = len(_off_ids)
    n_defenses  = len(_def_ids)

    store = _build_store(actor_level, geo, actor_name, start_month, end_month,
                         region, acled, mode)
    if store:
        fig_map = (_build_animated_choropleth(store, geo)
                   if mode == "animated" else _build_choropleth(store, geo))
    else:
        fig_map = empty_map

    alliance  = _build_alliance_chart(ally_pairs, actor_name, valid_ids)
    fig_trend = _build_trend(al)
    h_region, h_count = _highest_events(al, acled, region)

    return (
        fig_map,
        _fmt(n_townships), _fmt(n_offenses), _fmt(n_defenses),
        alliance, fig_trend,
        h_region, h_count, filter_summary, actor_name,
    )
