# pages/3_temporal.py
from __future__ import annotations

import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px

from components.layout import page_shell
from components.figures import empty_fig
from components.loaders import load_anomaly, load_time_series, load_geojson
from components.utils_dates import month_options, default_range_earliest_latest, month_bounds
from components.utils_format import fmt_date
from components.colors import ANOMALY_COLORS
from components.map_utils import apply_tight_geos, ensure_full_geoindex
from components.tiles import build_region_tiles
from components.ui import section_header, panel

dash.register_page(__name__, path="/temporal", name="Temporal Analysis")
PAGE_ID = "ts"


def _wrap_text(text: str, width: int = 60) -> str:
    """Insert <br> at word boundaries so long descriptions wrap in Plotly hovers.
    Keeps words intact; collapses internal whitespace; returns a string with <br>.
    """
    if not isinstance(text, str) or not text.strip():
        return "—"
    words = text.strip().split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        lw = len(w)
        # +1 for space if not first in line
        gain = lw if cur_len == 0 else lw + 1
        if cur_len + gain <= width:
            cur.append(w)
            cur_len += gain
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = lw
    if cur:
        lines.append(" ".join(cur))
    return "<br>".join(lines)

def _geo_township_name_map(geojson) -> dict[str, str]:
    """
    Build {TS_PCODE -> township_name} from the GeoJSON.
    We try several common property keys and fall back to empty string.
    """
    name_keys_try = [
        "TS_NAME", "TS", "TS_TN", "TSN", "NAME_3", "NAME_EN", "NAME", "Tsp_Name",
    ]
    out = {}
    for feat in geojson.get("features", []):
        p = feat.get("properties", {}) or {}
        pcode = p.get("TS_PCODE")
        if not pcode:
            continue
        # pick first non-empty among candidates
        label = ""
        for k in name_keys_try:
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                label = v.strip()
                break
        out[str(pcode)] = label
    return out


def layout():
    df_ts = load_time_series()
    df_an = load_anomaly()

    # No time-range filter on this page; we always use the full data range

    header = section_header("Temporal Analysis (Armed Conflicts)", "Monthly anomalies and regional tiles")

    # Move explainer above filters to save vertical space
    explainer = html.Div(
            [
                html.Div("Tile map", style={"fontWeight": 600, "marginBottom": "4px"}),
                html.Div(
                    "Each tile shows monthly armed conflict counts for a region (rows = years, columns = months). "
                    "Darker shades indicate higher numbers. The next 3 months are forecasted and highlighted in a different color, where the values are forecasted generated using an XGBoost time-series model. "
                    "The model relies on recent lags, rolling averages, and contextual conflict signals to estimate near-future events.",
                    style={"opacity": 0.85},
                ),
                html.Br(),
                html.Div("Anomaly groups by township", style={"fontWeight": 600, "marginBottom": "4px"}),
                html.Div(
                    "Each township is shaded according to whether its recent conflict activity is statistically unusual compared to past patterns. "
                    "Grey areas show normal activity. Blue indicates an event surge (unusually high number of armed-conflict incidents). "
                    "Red highlights a fatality surge (unusually high number of deaths). Purple marks both surges occurring together.",
                    style={"opacity": 0.85},
                ),
            ],
            className="panel panel-body",
            style={"padding": "8px 10px"},
    )

    # No filters for this page
    filters = []

    left_graph = panel(
        "Township-Level Anomalies in Conflict Events and Fatalities",
        dcc.Loading(html.Div([
            dcc.Graph(id=f"{PAGE_ID}-anomaly", className="graph-map-tall"),
        ]), className="dash-loading", type="default"),
    )
    # Restore a clear panel-level title (outside the subplot)
    right_graph = panel(
        "Regional Conflict Trends with Three-Month Forecasts",
        dcc.Loading(html.Div([
            dcc.Graph(id=f"{PAGE_ID}-tiles", className="graph-map-tall"),
        ]), className="dash-loading", type="default"),
    )

    foot = html.Div(
        [
            "Data: ", html.A("ACLED", href="https://acleddata.com", target="_blank", rel="noopener noreferrer"),
            " • Notes: Map is categorical; tiles are monthly totals (blue); next 3 months are forecasted (different color).",
        ],
        style={"textAlign": "right"},
    )

    return html.Div(
        [
            header,
            dcc.Store(id=f"{PAGE_ID}-init", data=0),
            explainer,
            page_shell(
                filters=filters, kpis=None, left_map=left_graph, right_content=right_graph, footnote=foot, page_class="temporal-page"
            ),
        ]
    )

@dash.callback(
    Output(f"{PAGE_ID}-anomaly","figure"),
    Output(f"{PAGE_ID}-tiles","figure"),
    Input(f"{PAGE_ID}-init","data"),
)
def render_page(_):
    df_ts = load_time_series()
    df_an = load_anomaly()
    geo   = load_geojson()

    # Use the full data range
    mstart, mend = default_range_earliest_latest(df_ts, "event_date")
    d1, _ = month_bounds(mstart); _, d2 = month_bounds(mend)

    # --------------------
    # Build TS_PCODE -> township name from GeoJSON (authoritative)
    # --------------------
    name_map = {}
    for feat in geo.get("features", []):
        p = feat.get("properties", {}) or {}
        code = str(p.get("TS_PCODE") or "")
        if not code:
            continue
        # try a few common name keys; keep the first non-empty
        for k in ("TS_ENG","TS","TS_NAME","NAME_3","NAME","Tsp_Name"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                name_map[code] = v.strip()
                break
        if code not in name_map:
            name_map[code] = ""  # no label in GJ

    # --------------------
    # Left: anomaly choropleth (categorical)
    # --------------------
    # Keep required cols from CSV; one row per township
    an = df_an[["Tsp_Pcode","map_category","admin1","admin2","admin3","flag_description"]].copy()
    an["Tsp_Pcode"] = an["Tsp_Pcode"].astype(str).str.strip()

    # Ensure every polygon is present; use empty string filler (NOT None)
    filled = ensure_full_geoindex(
        an, geo,
        id_col="Tsp_Pcode",
        feature_key="properties.TS_PCODE",
        fill_columns=["map_category","admin1","admin2","admin3","flag_description"],
        fill_value=""     # <- IMPORTANT: avoid pandas fillna(None) error
    )

    # Only map_category gets a default when missing; normalize to ANOMALY_COLORS keys
    filled["map_category"] = (
        filled["map_category"].astype(str).str.strip().str.lower().replace("", "normal")
    )
    valid_cats = set(ANOMALY_COLORS.keys())
    filled.loc[~filled["map_category"].isin(valid_cats), "map_category"] = "normal"

    # Township label: prefer GeoJSON name; fallback to admin3; else em dash
    filled["township_label"] = (
        filled["Tsp_Pcode"].map(name_map).fillna("").replace({"nan": ""})
    )
    m = filled["township_label"].eq("")
    filled.loc[m, "township_label"] = filled.loc[m, "admin3"].fillna("")
    filled.loc[filled["township_label"].eq(""), "township_label"] = "—"

    # Flag description cleaned
    filled["flag_desc_label"] = (
        filled["flag_description"].astype(str).str.strip().replace({"nan":"","None":""})
    )
    filled.loc[filled["flag_desc_label"].eq(""), "flag_desc_label"] = "—"
    # Wrap long descriptions so hover box doesn't overflow horizontally
    filled["flag_desc_wrapped"] = filled["flag_desc_label"].apply(lambda s: _wrap_text(s, width=60))

    # Build the map
    anomaly_fig = px.choropleth(
        filled,
        geojson=geo,
        locations="Tsp_Pcode",
        featureidkey="properties.TS_PCODE",
        color="map_category",
        category_orders={"map_category": list(ANOMALY_COLORS.keys())},
        color_discrete_map=ANOMALY_COLORS,
        custom_data=["township_label","flag_desc_wrapped"],  # wrapped description for hover
        template="plotly_white",
    )
    anomaly_fig.update_traces(
        hovertemplate="Township: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        marker_line_width=0.4, marker_line_color="#666",
    )
    apply_tight_geos(anomaly_fig, geo, height=740, show_colorbar=False)
    anomaly_fig.update_layout(
        legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left"),
        legend_title_text="",
        margin=dict(l=6, r=6, t=6, b=6),
        hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"),
    )

    # --------------------
    # Right: region tiles
    # --------------------
    tiles_fig = build_region_tiles(
        df_ts, start=d1, end=d2,
        region_order=sorted(df_ts["region"].dropna().astype(str).unique().tolist()),
        ncols=3
    )
    # Preserve sizing and margins from builder so titles remain visible and grid is compact.

    return anomaly_fig, tiles_fig