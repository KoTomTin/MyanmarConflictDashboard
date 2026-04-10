# components/map_utils.py
"""
Geo helpers (reusable across pages):
- apply_tight_geos(fig, geojson, ...)                   -> tight, tall map frame
- add_neighbor_labels(fig)                              -> national context labels + wider border view
- ensure_full_geoindex(df, geojson, ...)                -> add missing polygons (no holes)
- filter_geo_by_property(geojson, prop, value)          -> sub-geojson by property (e.g., Admin1 name)
- ids_from_geo(geojson, feature_key="properties.TS_PCODE") -> list of ids from geojson
"""

from __future__ import annotations
import plotly.graph_objects as go

# ---------- internal: compute lon/lat bounds ----------
def _geo_bounds(geojson: dict) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = 180.0, -180.0, 90.0, -90.0

    def _walk(x):
        nonlocal min_lon, max_lon, min_lat, max_lat
        if not isinstance(x, (list, tuple)):
            return
        if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            lon, lat = x
            min_lon = lon if lon < min_lon else min_lon
            max_lon = lon if lon > max_lon else max_lon
            min_lat = lat if lat < min_lat else min_lat
            max_lat = lat if lat > max_lat else max_lat
            return
        for y in x:
            _walk(y)

    for feat in geojson.get("features", []):
        _walk(feat.get("geometry", {}).get("coordinates", []))
    return min_lon, max_lon, min_lat, max_lat


# ---------- public: tighten map frame ----------
def apply_tight_geos(
    fig: go.Figure,
    geojson: dict,
    *,
    height: int = 720,
    pad_frac: float | None = None,
    show_colorbar: bool = True,
) -> go.Figure:
    min_lon, max_lon, min_lat, max_lat = _geo_bounds(geojson)
    dlon, dlat = max_lon - min_lon, max_lat - min_lat

    # Auto pad: use larger breathing room for small regions (< 30% of Myanmar extent).
    # Myanmar full extent is roughly 10° lon × 18° lat → area proxy ~180.
    # Anything below ~54 (30%) gets a larger pad so it doesn't appear as a thin sliver.
    if pad_frac is None:
        area_proxy = dlon * dlat
        # Full Myanmar ≈ 10 * 18 = 180; small region like Yangon ≈ 1.5 * 1.5 = 2.25
        if area_proxy < 10:
            pad_frac = 0.18   # city-scale or very small state
        elif area_proxy < 40:
            pad_frac = 0.10   # small state (Yangon, Kayah, Mon …)
        elif area_proxy < 90:
            pad_frac = 0.05   # medium state
        else:
            pad_frac = 0.014  # full country
    pad_lon, pad_lat = dlon * pad_frac, dlat * pad_frac

    fig.update_geos(
        visible=True,
        showframe=False,
        showcoastlines=False,
        coastlinecolor="#D7DFE9",
        coastlinewidth=0.55,
        showcountries=False,
        countrycolor="#D7DFE9",
        countrywidth=0.5,
        showsubunits=False,
        showland=False,
        showocean=False,
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
        fitbounds=None,
        lonaxis_range=[min_lon - pad_lon, max_lon + pad_lon],
        lataxis_range=[min_lat - pad_lat, max_lat + pad_lat],
    )
    fig.update_traces(marker_line_width=0.4, selector=dict(type="choropleth"))
    fig.update_layout(
        height=int(height),
        margin=dict(l=0, r=0, t=48, b=0),
        geo=dict(domain=dict(x=[0.0, 1.0], y=[0.015, 0.985])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=show_colorbar,
    )
    return fig


def add_neighbor_labels(fig: go.Figure, *, font_size: int = 9) -> go.Figure:
    """Neighbor context removed: keep only the Myanmar map."""
    return fig


# ---------- public: ensure full index (no holes) ----------
def ensure_full_geoindex(
    df,
    geojson: dict,
    *,
    id_col: str,
    feature_key: str = "properties.TS_PCODE",
    fill_columns: list[str] | None = None,
    fill_value = 0,
):
    import pandas as pd
    parts = feature_key.split(".")
    def _get_prop(feat):
        x = feat
        for p in parts:
            x = x.get(p) if isinstance(x, dict) else None
        return x

    all_ids = [_get_prop(ft) for ft in geojson.get("features", [])]
    full = pd.DataFrame({id_col: all_ids})
    out = full.merge(df, how="left", on=id_col)
    if fill_columns is None:
        fill_columns = [c for c in out.columns if c != id_col]
    for c in fill_columns:
        if c in out.columns:
            # Be defensive: if fill_value is None, choose a dtype-appropriate default
            fv = fill_value
            if fv is None:
                try:
                    is_str = (out[c].dtype == object) or pd.api.types.is_string_dtype(out[c])
                except Exception:
                    is_str = False
                fv = "" if is_str else 0
            out[c] = out[c].fillna(fv)
    return out


# ---------- public: filter geojson to a property value ----------
def filter_geo_by_property(
    geojson: dict,
    prop: str,
    value: str,
) -> dict:
    feats = [ft for ft in geojson.get("features", []) if ft.get("properties", {}).get(prop) == value]
    return {"type": "FeatureCollection", "features": feats}


# ---------- public: get list of ids from geojson ----------
def ids_from_geo(
    geojson: dict,
    feature_key: str = "properties.TS_PCODE"
) -> list[str]:
    parts = feature_key.split(".")
    out = []
    for ft in geojson.get("features", []):
        x = ft
        for p in parts:
            x = x.get(p) if isinstance(x, dict) else None
        out.append(x)
    return out
