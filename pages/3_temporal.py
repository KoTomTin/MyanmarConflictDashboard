# pages/3_temporal.py
from __future__ import annotations

import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px

from components.layout import page_shell
from components.filters import month_range
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

    mopts = month_options(df_ts, "event_date")
    mstart, mend = default_range_earliest_latest(df_ts, "event_date")

    header = section_header("Temporal Analysis (Armed Conflicts)", "Monthly anomalies and regional tiles")

    # Move explainer above filters to save vertical space
    explainer = html.Div(
            [
                html.Div("Tile map", style={"fontWeight": 600, "marginBottom": "4px"}),
                html.Div(
                    "Each tile shows monthly armed conflict counts for a region (rows = years, columns = months). "
                    "Darker shades indicate higher numbers. The last 3 months are highlighted in orange, where the values are forecasted generated using an XGBoost time-series model. "
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

    filters = [
        month_range(PAGE_ID, mopts, mstart, mend),
    ]

    left_graph = panel(
        "Anomaly groups by township",
        dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-anomaly", className="graph-map-tall"), className="dash-loading"),
    )
    # Restore a clear panel-level title (outside the subplot)
    right_graph = panel(
        "Region month×year tiles — totals (last 3 months highlighted)",
        dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-tiles", className="graph-map-tall"), className="dash-loading"),
    )

    foot = html.Div(
        [
            "Data: ACLED • Last updated ",
            html.Strong(fmt_date(df_ts["event_date"].max())),
            " • Notes: Map is categorical; tiles are monthly totals (blue); last 3 months highlighted (orange).",
        ],
        style={"textAlign": "right"},
    )

    return html.Div(
        [
            header,
            explainer,
            page_shell(
                filters=filters, kpis=None, left_map=left_graph, right_content=right_graph, footnote=foot, page_class="temporal-page"
            ),
        ]
    )

@dash.callback(
    Output(f"{PAGE_ID}-anomaly","figure"),
    Output(f"{PAGE_ID}-tiles","figure"),
    Input(f"{PAGE_ID}-month-start","value"),
    Input(f"{PAGE_ID}-month-end","value"),
)
def render_page(start_mm, end_mm):
    df_ts = load_time_series()
    df_an = load_anomaly()
    geo   = load_geojson()

    if not start_mm or not end_mm:
        empty = empty_fig("No data"); return empty, empty
    if end_mm < start_mm:
        start_mm, end_mm = end_mm, start_mm
    d1, _ = month_bounds(start_mm); _, d2 = month_bounds(end_mm)

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

    # Build the map
    anomaly_fig = px.choropleth(
        filled,
        geojson=geo,
        locations="Tsp_Pcode",
        featureidkey="properties.TS_PCODE",
        color="map_category",
        category_orders={"map_category": list(ANOMALY_COLORS.keys())},
        color_discrete_map=ANOMALY_COLORS,
        custom_data=["township_label","flag_desc_label"],  # bind per-point at creation
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