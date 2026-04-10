"""
Page 2 — Actor Analysis  (design v4)

Same card / grid design as Overview.
Time range: quick presets (Last 7 / 30 / Since Feb 2021) + inline From→To dropdowns.
Map: Period Total | Quarterly Animation — Plotly native frames (no lag, same as Overview).
Charts: Alliance network table + monthly engagement trend (with export button).
"""
import re
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate

from components.loaders   import (load_acled_main, load_actor_level,
                                   load_ally_pairs, load_geojson, load_last_checked)
from components.colors    import SEQUENTIAL_BLUES_ZERO_GREY
from components.map_utils import apply_tight_geos, add_neighbor_labels, filter_geo_by_property
from components.page_bits import data_disclaimer


DEFAULT_ACTOR = "Myanmar Military Regime"
NON_ARMED_ACTORS = {
    "Labor Group (Myanmar)",
    "Prisoners (Myanmar)",
}

# (date, hover title, short pill, line color, description)
MILESTONES = [
    ("2021-02-01", "Feb 1, 2021 – Military Coup",           "Military<br>Coup",       "#dc2626",
     "The Myanmar military seized power and arrested elected leaders, sparking nationwide protests."),
    ("2021-09-07", "Sept 7, 2021 – NUG Declaration of People's Defensive War", "People's<br>Defensive War",    "#d97706",
     "The NUG called for armed resistance against the military, escalating the conflict."),
    ("2023-10-27", "Oct 27, 2023 – Operation 1027",         "Op. 1027",   "#7c3aed",
     "Rebel alliance launched a major offensive, capturing key territory in northern Shan State."),
    ("2025-03-28", "Mar 28, 2025 – 7.7 Earthquake",         "Earthquake", "#0891b2",
     "A major earthquake near Mandalay caused heavy destruction and worsened the crisis."),
    ("2025-12-28", "Dec 28, 2025 – SAC-organized Elections", "SAC-organized<br>Elections",  "#6b7280",
     "The SAC held phased elections in controlled areas; the process was widely rejected as illegitimate."),
]

PLOTLY_FONT = "Avenir Next, Segoe UI, Arial, sans-serif"
PLOTLY_DISPLAY = "Iowan Old Style, Palatino Linotype, Book Antiqua, Georgia, serif"
PLOTLY_TEXT = "#415669"
PLOTLY_GRID = "#e6ddd0"
PLOTLY_HOVER_BG = "rgba(255,251,246,0.98)"
PLOTLY_HOVER_BORDER = "#d9cfbf"


# ── Module-level defaults ──────────────────────────────────────────────────────

def _get_defaults() -> dict:
    df = load_acled_main()
    checked = load_last_checked()
    checked_str = checked.get("display") or checked.get("date_display") or "Not yet recorded"
    checked_note = checked.get("cadence_note") or "Scheduled ACLED check: Thursday evening"
    timezone_label = checked.get("timezone_label") or "Yangon time"
    if timezone_label and timezone_label.lower() not in checked_note.lower():
        checked_note = f"{checked_note} · {timezone_label}"
    return {
        "start_val":  df["event_date"].min().strftime("%Y-%m-%d"),
        "end_val":    df["event_date"].max().strftime("%Y-%m-%d"),
        "latest_str": df["event_date"].max().strftime("%d %b %Y"),
        "checked_str": checked_str,
        "checked_note": checked_note,
    }


# ── Generic helpers ────────────────────────────────────────────────────────────

def _fmt(n) -> str:
    return f"{int(n):,}"


def _empty_fig(msg="No data for this selection", height=460):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#5a6c7e", family=PLOTLY_FONT), xref="paper", yref="paper")
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


def _is_generic_civilian(name: str) -> bool:
    return bool(re.match(r"^civilians?\s*\(", name.strip(), re.IGNORECASE))


def _is_non_armed_actor(name: str) -> bool:
    return str(name).strip() in NON_ARMED_ACTORS


def _is_excluded_actor(name: str) -> bool:
    name = str(name).strip()
    return (
        _is_generic_civilian(name)
        or _is_non_armed_actor(name)
        or "unidentified" in name.lower()
    )


def _geo_pcodes(geo: dict) -> list[str]:
    return [f["properties"]["TS_PCODE"] for f in geo["features"]]


def _tsp_text(geo: dict) -> list[str]:
    return [f["properties"].get("TS", f["properties"]["TS_PCODE"])
            for f in geo["features"]]


def _tsp_states(geo: dict) -> list[str]:
    return [f["properties"].get("ST", "") for f in geo["features"]]


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
    def _fmt_date(value: str | None) -> str:
        if value:
            try:
                return pd.to_datetime(value).strftime("%d %b %Y")
            except Exception:
                return value
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


def _filter_actor_events(actor_level, actor_name, start_date, end_date, region, acled):
    al = actor_level[actor_level["actor_name"] == actor_name].copy()
    al["event_date"] = pd.to_datetime(al["event_date"], errors="coerce")
    if start_date:
        al = al[al["event_date"] >= pd.to_datetime(start_date)]
    if end_date:
        al = al[al["event_date"] <= pd.to_datetime(end_date)]
    if region:
        valid_pcodes = set(acled[acled["admin1"] == region]["Tsp_Pcode"].dropna().unique())
        al = al[al["Tsp_Pcode"].isin(valid_pcodes)]
    return al


def _q95(series: pd.Series) -> int:
    s = series[series > 0]
    if s.empty:
        return 1
    return max(int(s.quantile(0.95)) if len(s) >= 5 else int(s.max()), 1)


def _chart_config(filename: str, *, show_modebar: bool = False) -> dict:
    """Modebar config: camera download button on hover."""
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


def _build_filter_chips(start_month, end_month, region, actor_name,
                        mode: str = "time_range",
                        preset_label: str | None = None,
                        start_date: str | None = None,
                        end_date: str | None = None):
    def fmt_m(m):
        try:
            return pd.Timestamp(m + "-01").strftime("%b %Y")
        except Exception:
            return m or ""

    time_str = preset_label or _format_selected_range(start_date, end_date, start_month, end_month)
    chips = [
        ("Date", time_str),
        ("Region", region or "All Myanmar"),
        ("Actor", actor_name),
    ]
    chips.append(("View", "Quarterly playback" if mode == "animated" else "Total period"))
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

def _build_store(actor_level, geo, actor_name, start_date, end_date,
                 region, acled, mode) -> dict | None:
    al = _filter_actor_events(actor_level, actor_name, start_date, end_date, region, acled)
    if al.empty:
        return None

    active_geo = filter_geo_by_property(geo, "ST", region) if region else geo
    pcodes     = _geo_pcodes(active_geo)
    text       = _tsp_text(active_geo)
    states     = _tsp_states(active_geo)
    n          = len(pcodes)
    pcode_idx  = {p: i for i, p in enumerate(pcodes)}
    start_month = pd.to_datetime(start_date).strftime("%Y-%m") if start_date else None
    end_month = pd.to_datetime(end_date).strftime("%Y-%m") if end_date else None

    agg = (
        al.groupby(["Tsp_Pcode", "event_date"])["event_id_cnty"]
        .nunique().reset_index(name="events")
    )

    if mode == "animated":
        agg["quarter"] = agg["event_date"].dt.to_period("Q").astype(str)
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
            "start_month": start_month,
            "end_month": end_month,
            "pcodes": pcodes, "text": text, "states": states, "region": region,
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
            "start_month": start_month,
            "end_month": end_month,
            "pcodes": pcodes, "text": text, "states": states, "region": region,
        }


# ── Static choropleth ──────────────────────────────────────────────────────────

def _build_choropleth(
    store: dict,
    geo: dict,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> go.Figure:
    active_geo = filter_geo_by_property(geo, "ST", store["region"]) if store.get("region") else geo
    fig = go.Figure(go.Choropleth(
        geojson=active_geo,
        locations=store["pcodes"],
        z=store["z"], text=store["text"],
        customdata=store.get("states", [""] * len(store["pcodes"])),
        featureidkey="properties.TS_PCODE",
        colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
        zmin=0, zmax=store["max_val"],
        marker_line_width=0.3, marker_line_color="#ffffff",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "<span style='color:#5a6c7e'>%{customdata}</span><br>"
            "Events: <b>%{z:,}</b>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(
                text="Events",
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
    apply_tight_geos(fig, active_geo, height=700)
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
        customdata=store.get("states", [""] * len(store["pcodes"])),
        featureidkey="properties.TS_PCODE",
        colorscale=SEQUENTIAL_BLUES_ZERO_GREY,
        zmin=0, zmax=mv,
        marker_line_width=0.3, marker_line_color="#ffffff",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "<span style='color:#5a6c7e'>%{customdata}</span><br>"
            "Events: <b>%{z:,}</b>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(
                text="Events",
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
    apply_tight_geos(fig, active_geo, height=740)
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


# ── Trend chart ────────────────────────────────────────────────────────────────

def _build_trend(al: pd.DataFrame, start_date: str | None, end_date: str | None) -> go.Figure:
    if al.empty:
        return _empty_fig(height=200)
    al = al.copy()
    al["event_date"] = pd.to_datetime(al["event_date"], errors="coerce")
    use_daily = (_window_days(start_date, end_date) or 999) <= 45
    bucket_col = "day_dt" if use_daily else "period_dt"
    if use_daily:
        al["day_dt"] = al["event_date"].dt.normalize()
    else:
        al["period_dt"] = al["event_date"].dt.to_period("M").dt.to_timestamp()

    trend = (
        al.groupby([bucket_col, "type2"], observed=True)["event_id_cnty"]
        .nunique().reset_index(name="events").sort_values(bucket_col)
    )

    color_map = {"offend": "#b85d57", "being_offended": "#3f698d"}
    label_map = {"offend": "Offending side", "being_offended": "Targeted side"}

    fig = go.Figure()
    for role in trend["type2"].unique():
        color = color_map.get(role, "#94a3b8")
        label = label_map.get(role, role)
        grp = trend[trend["type2"] == role]
        fig.add_trace(go.Scatter(
            x=grp[bucket_col], y=grp["events"],
            mode="lines+markers" if use_daily else "lines",
            name=label,
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=6),
            legendgroup=role, showlegend=True,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"%{{x|{'%d %b %Y' if use_daily else '%b %Y'}}}: %{{y:,}}"
                f"<extra></extra>"
            ),
        ))

    # Milestone vertical lines — color-coded pills + hover diamonds
    all_dates = trend[bucket_col]
    if not all_dates.empty:
        x_min = all_dates.min()
        x_max = all_dates.max()
        for i, (date_str, title, short_name, color, desc) in enumerate(MILESTONES):
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
                    x=mdt, y=1.13 if i % 2 == 0 else 1.04, xref="x", yref="paper",
                    text=f"<b>{short_name}</b>",
                    showarrow=False,
                    font=dict(size=10, color=color, family=PLOTLY_FONT),
                    xanchor="center", yanchor="bottom",
                    bgcolor=PLOTLY_HOVER_BG,
                    bordercolor=color, borderwidth=1, borderpad=3,
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
                        f"<span style='color:#5a6c7e'>{desc}</span>"
                        f"<extra></extra>"
                    ),
                ))

    fig.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=62, b=54),
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0,
                    font=dict(size=10, family=PLOTLY_FONT, color=PLOTLY_TEXT), bgcolor="rgba(0,0,0,0)"),
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        xaxis=dict(showgrid=False, tickformat="%d %b" if use_daily else "%b %Y", tickangle=-30,
                   tickfont=dict(size=10, family=PLOTLY_FONT, color=PLOTLY_TEXT), linecolor=PLOTLY_GRID),
        yaxis=dict(showgrid=True, gridcolor=PLOTLY_GRID, zeroline=True,
                   zerolinecolor=PLOTLY_GRID, zerolinewidth=1,
                   rangemode="tozero", autorange=True,
                   tickfont=dict(size=10, family=PLOTLY_FONT, color=PLOTLY_TEXT), title=None),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=11),
        ),
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
        .sort_values("events", ascending=False)
        .head(15)
    )

    fig = go.Figure(go.Bar(
        y=tbl["ally_name"],
        x=tbl["events"],
        orientation="h",
        marker=dict(color="#355c84", line=dict(color="rgba(56,78,99,0.08)", width=0.5)),
        text=[f"{v:,}" for v in tbl["events"]],
        textposition="outside",
        textfont=dict(size=10, color=PLOTLY_TEXT, family=PLOTLY_FONT),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,} shared events<extra></extra>",
    ))
    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=80, t=12, b=4),
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
        yaxis=dict(tickfont=dict(size=9, color=PLOTLY_TEXT, family=PLOTLY_FONT), title=None,
                   autorange="reversed", automargin=True),
        xaxis=dict(showgrid=True, gridcolor=PLOTLY_GRID, tickfont=dict(size=9, color=PLOTLY_TEXT, family=PLOTLY_FONT),
                   title=None, rangemode="tozero"),
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER_BG,
            bordercolor=PLOTLY_HOVER_BORDER,
            font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=11),
        ),
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
    match = acled[acled["Tsp_Pcode"] == top_pcode]["admin3"]
    top_name = str(match.iloc[0]) if not match.empty else top_pcode
    return top_name, f"{int(by_pcode.max()):,}"


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    meta        = _get_defaults()
    acled       = load_acled_main()
    actor_level = load_actor_level()
    # ally_pairs not loaded here — only needed by the callback (cached anyway)
    geo         = load_geojson()

    start_val   = meta["start_val"]
    end_val     = meta["end_val"]
    latest_str  = meta["latest_str"]
    checked_str = meta["checked_str"]
    checked_note = meta["checked_note"]

    actor_opts       = _actor_options(actor_level)
    admin1_opts      = sorted(acled["admin1"].dropna().unique())
    total_townships  = len(_geo_pcodes(geo))

    default_store = {
        "start_month": start_val[:7], "end_month": end_val[:7],
        "start_date": start_val, "end_date": end_val,
        "region": None, "actor_name": DEFAULT_ACTOR, "preset_label": None,
    }

    # ── Pre-compute fast KPIs (no chart building — charts filled by callback) ──
    al_default = actor_level[
        actor_level["actor_name"] == DEFAULT_ACTOR
    ].copy()
    al_default["event_date"] = pd.to_datetime(al_default["event_date"], errors="coerce")
    al_default["month"] = al_default["event_date"].dt.to_period("M").astype(str)

    n_townships  = al_default["Tsp_Pcode"].nunique()
    _off_ids     = set(al_default[al_default["type2"] == "offend"]["event_id_cnty"])
    _def_ids     = set(al_default[al_default["type2"] == "being_offended"]["event_id_cnty"]) - _off_ids
    n_offenses   = len(_off_ids)
    n_defenses   = len(_def_ids)
    h_region, h_count = _highest_events(al_default, acled, None)
    init_summary  = _build_filter_chips(
        start_val[:7],
        end_val[:7],
        None,
        DEFAULT_ACTOR,
        start_date=start_val,
        end_date=end_val,
    )

    # Empty figure placeholders — the update_actor callback fills these immediately
    # (prevent_initial_call=False) so the user sees spinners briefly, not blank cards.
    init_map   = _empty_fig(height=700)
    init_trend = _empty_fig(height=280)

    return html.Div([

        # ── hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="ac-applied-filters", data=default_store),
        dcc.Store(id="ac-mode", data="time_range"),

        # ── page header ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("Actor Analysis", className="page-title"),
                html.Div("Actor-level view of Myanmar conflict events, geographic footprint, associated actors, and engagement trends over time.",
                         className="page-subtitle"),
                html.Div([
                    html.Div("Prototype · work in progress", className="hero-pill hero-pill--prototype"),
                    html.Div([
                        html.Span("Actor", className="hero-status-key"),
                        html.Span(DEFAULT_ACTOR, id="ac-actor-banner-name",
                                  className="hero-status-value-inline"),
                    ], className="hero-status-pill"),
                    html.Div("Combat-only view", className="hero-pill"),
                ], className="hero-pill-row hero-pill-row--tight"),
            ], className="page-header-left"),
            html.Div([
                html.Div([
                    html.Span("Events up to", className="hero-status-key"),
                    html.Span(latest_str, className="hero-status-value-inline"),
                ], className="hero-status-pill"),
                html.Div([
                    html.Span("Last checked", className="hero-status-key"),
                    html.Span(checked_str, className="hero-status-value-inline"),
                ], className="hero-status-pill"),
                html.Div(checked_note, className="hero-status-note hero-status-note--inline"),
            ], className="hero-status-inline"),
        ], className="page-header overview-hero"),

        # ── filter card ────────────────────────────────────────────────────────
        html.Div([
            html.Div([

                # Time Range: quick presets + date-card pickers
                html.Div([
                    html.Label("Time Range", className="filter-label"),
                    html.Div([
                        html.Button("Last 7 days",    id="ac-btn-7d",  n_clicks=0, className="quick-btn"),
                        html.Button("Last 30 days",   id="ac-btn-30d", n_clicks=0, className="quick-btn"),
                        html.Button("Last 1 year",    id="ac-btn-1y",  n_clicks=0, className="quick-btn"),
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

            ], className="filter-controls"),
            html.Div([
                html.Div(
                    "Changes update automatically. Use Reset to return to the default actor view.",
                    className="filter-drawer-note",
                ),
                html.Button("Reset", id="ac-reset-btn",
                            n_clicks=0, className="btn-reset btn-reset--solo"),
            ], className="filter-drawer-actions"),
        ], id="ac-filter-card", className="filter-card"),

        # ── filter summary ─────────────────────────────────────────────────────
        html.Div(init_summary, id="ac-filter-summary", className="filter-summary"),

        # ── two-column body ────────────────────────────────────────────────────
        html.Div([

            # ── LEFT: hero map ─────────────────────────────────────────────────
            html.Div([
                html.Div([

                    html.Div([
                        html.Div([
                            html.H2("Geographic Footprint", className="card-title"),
                            html.Div("Point at a township. Blue indicates higher reported activity.",
                                     className="card-subtitle"),
                        ], className="map-stage-copy"),
                        html.Div([
                            html.Button(
                                "Total period",
                                id="ac-mode-total-btn",
                                n_clicks=0,
                                type="button",
                                **{"aria-label": "Show one map for the full date range",
                                   "aria-pressed": "true",
                                   "title": "One map covering the full selected date range"},
                                className="view-toggle-btn view-toggle-btn--active",
                            ),
                            html.Button(
                                "Quarterly playback",
                                id="ac-mode-anim-btn",
                                n_clicks=0,
                                type="button",
                                **{"aria-label": "Step through each quarter",
                                   "aria-pressed": "false",
                                   "title": "Animated quarter-by-quarter playback — use the play button on the map"},
                                className="view-toggle-btn",
                            ),
                        ], className="map-stage-controls", role="group",
                           **{"aria-label": "Map view mode"}),
                    ], className="dash-card-head map-stage-head"),

                    dcc.Loading(
                        dcc.Graph(id="ac-map", figure=init_map,
                                  className="map-graph",
                                  style={"height": "100%"},
                                  config={**_chart_config("actor_geographic_footprint", show_modebar=True),
                                          "displayModeBar": True}),
                        type="dot", color="#2563eb",
                    ),

                    html.Div([
                        html.Div("Most Active Township", className="highest-label"),
                        html.Div([
                            html.Div(h_region, id="ac-highest-region", className="highest-region"),
                            html.Div(h_count,  id="ac-highest-count",  className="highest-count"),
                        ], className="highest-values"),
                    ], className="highest-events"),

                ], className="dash-card map-stage"),
            ], className="col-map"),

            # ── RIGHT: KPIs + alliance table + trend ───────────────────────────
            html.Div([

                html.Div([
                    html.Div([html.Div("Townships",                        className="kpi-label"),
                              html.Div(_fmt(n_townships),  id="ac-kpi-townships", className="kpi-value"),
                              html.Div(f"of {total_townships} in Myanmar",         className="kpi-sub")],
                             className="kpi-card kpi-accent-teal"),
                    html.Div([html.Div("Offending-side Events",     className="kpi-label"),
                              html.Div(_fmt(n_offenses),   id="ac-kpi-offenses",   className="kpi-value"),
                              html.Div("actor on offending side", className="kpi-sub")],
                             className="kpi-card kpi-accent-orange"),
                    html.Div([html.Div("Targeted-side Events",       className="kpi-label"),
                              html.Div(_fmt(n_defenses),   id="ac-kpi-defenses",   className="kpi-value"),
                              html.Div("actor on targeted side", className="kpi-sub")],
                             className="kpi-card kpi-accent-red"),
                ], className="kpis-row kpis-row--3 kpis-compact"),

                html.Div([
                    html.Div([
                        html.H2("Associated Actors", className="card-title"),
                        html.Div("Co-involved in the same combat events",
                                 className="card-subtitle"),
                    ], className="dash-card-head"),
                    dcc.Loading(
                        html.Div(id="ac-alliance-table", className="panel-body"),
                        type="dot", color="#2563eb",
                    ),
                    html.Div(
                        "Note: \"associated\" means co-appearing on the same side of the same recorded combat event. "
                        "It does not by itself imply a formal alliance, command structure, or coordination.",
                        className="actor-coapp-note",
                    ),
                ], className="dash-card"),

                html.Div([
                    html.Div([
                        html.H2("Monthly Engagement Trend", className="card-title"),
                        html.Div([
                            html.Span("Offending side", style={"color": "#ef4444", "fontWeight": "600"}),
                            html.Span(" = actor recorded on the offending side in our recode  ·  "),
                            html.Span("Targeted side", style={"color": "#3b82f6", "fontWeight": "600"}),
                            html.Span(" = actor recorded on the opposing side in our recode."),
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
        data_disclaimer(),

    ], className="page-wrap")


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# 0. Filters → update actor dropdown counts to reflect current date/region scope
@callback(
    Output("ac-actor", "options"),
    Input("ac-applied-filters", "data"),
    prevent_initial_call=True,
)
def update_actor_options(applied):
    if not applied:
        raise PreventUpdate
    actor_level = load_actor_level()
    al = actor_level.copy()
    al["event_date"] = pd.to_datetime(al["event_date"], errors="coerce")
    al["month"] = al["event_date"].dt.to_period("M").astype(str)
    start_month = applied.get("start_month")
    end_month   = applied.get("end_month")
    region      = applied.get("region")
    if start_month: al = al[al["month"] >= start_month]
    if end_month:   al = al[al["month"] <= end_month]
    if region:
        acled = load_acled_main()
        valid_pcodes = set(acled[acled["admin1"] == region]["Tsp_Pcode"].dropna().unique())
        al = al[al["Tsp_Pcode"].isin(valid_pcodes)]
    return _actor_options(al)


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
        return "animated", "view-toggle-btn", "view-toggle-btn view-toggle-btn--active"
    return "time_range", "view-toggle-btn view-toggle-btn--active", "view-toggle-btn"


# 2. Quick preset buttons → update date pickers
@callback(
    Output("ac-from-date",       "date"),
    Output("ac-to-date",         "date"),
    Input("ac-btn-7d",           "n_clicks"),
    Input("ac-btn-30d",          "n_clicks"),
    Input("ac-btn-1y",           "n_clicks"),
    Input("ac-btn-all",          "n_clicks"),
    prevent_initial_call=True,
)
def set_quick_dates(n7, n30, n1y, nall):
    max_dt = load_acled_main()["event_date"].max()
    if ctx.triggered_id == "ac-btn-7d":
        start_dt = max_dt - pd.Timedelta(days=7)
    elif ctx.triggered_id == "ac-btn-30d":
        start_dt = max_dt - pd.Timedelta(days=30)
    elif ctx.triggered_id == "ac-btn-1y":
        start_dt = max_dt - pd.Timedelta(days=365)
    else:
        start_dt = pd.Timestamp("2021-02-01")
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = max_dt.strftime("%Y-%m-%d")
    return start_date, end_date


# 3. Filter controls → update store automatically
@callback(
    Output("ac-applied-filters", "data", allow_duplicate=True),
    Input("ac-from-date",        "date"),
    Input("ac-to-date",          "date"),
    Input("ac-region",           "value"),
    Input("ac-actor",            "value"),
    prevent_initial_call=True,
)
def apply_filters(from_date, to_date, region, actor):
    d = _get_defaults()
    start_m = (from_date or d["start_val"])[:7]
    end_m   = (to_date   or d["end_val"])[:7]
    return {"start_month": start_m, "end_month": end_m,
            "start_date": from_date or d["start_val"],
            "end_date": to_date or d["end_val"],
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
                "start_date": d["start_val"], "end_date": d["end_val"],
                "region": None, "actor_name": DEFAULT_ACTOR, "preset_label": None}
    return None, DEFAULT_ACTOR, d["start_val"], d["end_val"], defaults


# 5. Filters + mode → rebuild all charts
# prevent_initial_call=False so charts load immediately on page mount
# (layout() returns empty placeholders; callback fills them on first render)
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
    prevent_initial_call=False,
)
def update_actor(applied, mode):
    if not applied:
        raise PreventUpdate

    start_month  = applied.get("start_month")
    end_month    = applied.get("end_month")
    start_date   = applied.get("start_date")
    end_date     = applied.get("end_date")
    region       = applied.get("region")
    actor_name   = applied.get("actor_name") or DEFAULT_ACTOR
    if _is_excluded_actor(actor_name):
        actor_name = DEFAULT_ACTOR
    preset_label = applied.get("preset_label")
    mode         = mode or "time_range"

    acled       = load_acled_main()
    actor_level = load_actor_level()
    ally_pairs  = load_ally_pairs()
    geo         = load_geojson()

    al = _filter_actor_events(actor_level, actor_name, start_date, end_date, region, acled)

    filter_summary = _build_filter_chips(start_month, end_month, region, actor_name,
                                         mode, preset_label,
                                         start_date=start_date, end_date=end_date)
    empty_map = _empty_fig(f"No data for '{actor_name}' in this period", height=700)

    if al.empty:
        no_data_hint = html.Div([
            html.Div("No events found for this selection.", className="actor-empty-title"),
            html.Div(
                f"'{actor_name}' has no recorded events matching the current filters. "
                "Try expanding the date range, removing the region filter, or choosing a different actor.",
                className="actor-empty-hint",
            ),
        ], className="actor-empty-state")
        return (
            empty_map, "—", "—", "—",
            no_data_hint,
            _empty_fig(height=280), "—", "—", filter_summary, actor_name,
        )

    valid_ids   = set(al["event_id_cnty"].unique())
    n_townships = al["Tsp_Pcode"].nunique()
    _off_ids    = set(al[al["type2"] == "offend"]["event_id_cnty"])
    _def_ids    = set(al[al["type2"] == "being_offended"]["event_id_cnty"]) - _off_ids
    n_offenses  = len(_off_ids)
    n_defenses  = len(_def_ids)

    store = _build_store(actor_level, geo, actor_name, start_date, end_date,
                         region, acled, mode)
    if store:
        fig_map = (
            _build_animated_choropleth(store, geo)
            if mode == "animated"
            else _build_choropleth(store, geo, start_date=start_date, end_date=end_date)
        )
    else:
        fig_map = empty_map

    alliance  = _build_alliance_chart(ally_pairs, actor_name, valid_ids)
    fig_trend = _build_trend(al, start_date, end_date)
    h_region, h_count = _highest_events(al, acled, region)

    return (
        fig_map,
        _fmt(n_townships), _fmt(n_offenses), _fmt(n_defenses),
        alliance, fig_trend,
        h_region, h_count, filter_summary, actor_name,
    )
