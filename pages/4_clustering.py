# pages/4_clustering.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from components.layout import page_shell
from components.ui import panel, section_header
from components.filters import month_range
from components.loaders import load_acled_main, load_geojson
from components.utils_dates import month_options, default_range_earliest_latest, month_bounds
from components.utils_format import fmt_date
from components.map_utils import apply_tight_geos, ensure_full_geoindex
from components.colors import CLUSTER_COLORS

dash.register_page(__name__, path="/clustering", name="Clustering Analysis")
PAGE_ID = "cl"

def aggregate_features(df: pd.DataFrame, d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    """Build township-level features within [d1, d2]."""
    sub = df[(df["event_date"] >= d1) & (df["event_date"] <= d2)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Tsp_Pcode", "admin1", "admin2", "admin3"])

    # Base identity columns
    ident = sub.groupby("Tsp_Pcode", as_index=False).agg(
        admin1=("admin1", "first"),
        admin2=("admin2", "first"),
        admin3=("admin3", "first"),
    )

    # Get all detailed events and create count columns
    detailed_events = sorted(sub["detailed_event"].dropna().astype(str).str.strip().unique())
    
    # Create pivot table for detailed events
    sub_copy = sub.copy()
    sub_copy["detailed_event_clean"] = sub_copy["detailed_event"].astype(str).str.strip()
    
    pv = sub_copy.pivot_table(
        index="Tsp_Pcode", 
        columns="detailed_event_clean", 
        values="event_id_cnty", 
        aggfunc="count", 
        fill_value=0
    ).reset_index()
    
    # Rename columns to avoid spaces/special chars
    event_cols = {}
    for col in pv.columns:
        if col != "Tsp_Pcode":
            event_cols[col] = f"{col}_count"
    pv = pv.rename(columns=event_cols)

    # Fatalities sum
    fat = sub.groupby("Tsp_Pcode", as_index=False)["fatalities"].sum().rename(columns={"fatalities": "fatalities_sum"})

    # Civilian targeting count
    is_civ = sub["civilian_targeting"].astype(str).str.strip().str.lower() == "yes"
    ct = sub.assign(is_ct=is_civ.astype(int)).groupby("Tsp_Pcode", as_index=False)["is_ct"].sum().rename(columns={"is_ct": "civ_target_count"})

    # Merge everything
    out = ident.merge(pv, on="Tsp_Pcode", how="left").merge(fat, on="Tsp_Pcode", how="left").merge(ct, on="Tsp_Pcode", how="left")
    
    # Fill NaN values
    for col in out.columns:
        if col not in ["Tsp_Pcode", "admin1", "admin2", "admin3"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out

def get_feature_columns(agg: pd.DataFrame) -> list[str]:
    """Get all feature columns (excluding ID and admin columns)."""
    exclude = ["Tsp_Pcode", "admin1", "admin2", "admin3"]
    return [col for col in agg.columns if col not in exclude]

def create_feature_labels(feature_cols: list[str]) -> dict[str, str]:
    """Create nice labels for features."""
    labels = {}
    for col in feature_cols:
        if col == "fatalities_sum":
            labels[col] = "Fatalities (sum)"
        elif col == "civ_target_count":
            labels[col] = "Civilian targeting (count)"
        elif col.endswith("_count"):
            # Remove _count suffix and clean up
            clean_name = col[:-6].replace("_", " ").title()
            labels[col] = f"{clean_name} (count)"
        else:
            labels[col] = col
    return labels

def layout():
    df = load_acled_main()
    
    mopts = month_options(df, "event_date")
    mstart, mend = default_range_earliest_latest(df, "event_date")
    
    # Primary actor type options
    actor_types = ["All"] + sorted(df["primary_actor_type"].dropna().astype(str).str.strip().unique())
    
    # Get sample data to build feature options
    sample_agg = aggregate_features(df, df["event_date"].min(), df["event_date"].max())
    feature_cols = get_feature_columns(sample_agg)
    labels = create_feature_labels(feature_cols)
    
    # Default features: civilian targeting, fatalities, air strike
    default_features = []
    for col in feature_cols:
        if "civ_target_count" in col:
            default_features.append(col)
        elif "fatalities_sum" in col:
            default_features.append(col)
        elif "air_strike" in col.lower() or "air strike" in labels.get(col, "").lower():
            default_features.append(col)
    
    # If we don't have exactly 3 defaults, fill with first available
    if len(default_features) < 3:
        for col in feature_cols:
            if col not in default_features:
                default_features.append(col)
                if len(default_features) >= 3:
                    break

    head = section_header("Clustering Analysis", "Group townships by conflict patterns")

    filters = [
        month_range(PAGE_ID, mopts, mstart, mend),
        
        html.Div([
            html.Label("Primary actor type"),
            dcc.Dropdown(
                id=f"{PAGE_ID}-actor-type",
                options=[{"label": v, "value": v} for v in actor_types],
                value="Military Regime" if "Military Regime" in actor_types else "All",
                clearable=False,
            ),
        ]),
        
        html.Div([
            html.Label("Features (select 2-7)"),
            dcc.Dropdown(
                id=f"{PAGE_ID}-features",
                options=[{"label": labels[col], "value": col} for col in feature_cols],
                value=default_features[:3],
                multi=True,
                clearable=False,
            ),
        ]),
        
        html.Div([
            html.Label("Number of clusters"),
            dcc.Slider(
                id=f"{PAGE_ID}-k", 
                min=2, max=7, step=1, value=3, 
                marks={i: str(i) for i in range(2, 8)}
            ),
        ], style={"paddingTop": "8px"}),
        
        html.Div([
            html.Button("Run clustering", id=f"{PAGE_ID}-run", n_clicks=0, className="btn btn-primary"),
            html.Span(id=f"{PAGE_ID}-status", style={"marginLeft": "12px", "opacity": 0.85}),
            dbc.Button("ℹ️", id=f"{PAGE_ID}-info-btn", color="link", size="sm", className="p-0 ms-2", title="About silhouette score", style={"fontSize":"16px", "lineHeight":"1"}),
            dbc.Popover(
                [
                    dbc.PopoverHeader("Silhouette score"),
                    dbc.PopoverBody(
                        html.Ul([
                            html.Li("The score can range from -1 to +1."),
                            html.Li("A score close to +1 means the data point fits very well in its own cluster and is far from others."),
                            html.Li("A score close to 0 means the data point is between clusters or the clusters are overlapping."),
                            html.Li("A score close to -1 means the data point is in the wrong cluster."),
                        ])
                    ),
                ],
                id=f"{PAGE_ID}-info-popover",
                target=f"{PAGE_ID}-info-btn",
                trigger="click",
                placement="top",
            ),
        ], style={"paddingTop": "14px", "display":"flex", "alignItems":"center", "gap":"6px"}),
    ]
    
    left = panel(
        "Township Clusters of Conflict Profiles", 
        dcc.Loading(
            html.Div([
                dcc.Graph(id=f"{PAGE_ID}-map", className="graph-map-tall"),
            ]),
            className="dash-loading",
            type="default"
        )
    )
    
    right = html.Div([
        panel(
            "About this page",
            html.Div([
                html.P(
                    "Clustering groups townships that share similar conflict patterns. "
                    "Use it to explore how your selected features vary across Myanmar: choose 2–7 features (start with 2–3), "
                    "run clustering, and read the map by color—townships with the same color share broadly similar profiles."
                ),
                html.P(
                    "The map is best for comparing which areas look alike and which stand out. "
                    "Cluster borders are fuzzy by nature: townships near color boundaries often share characteristics of both sides, "
                    "and occasional outliers within a color usually reflect especially high or low values on one feature."
                ),
                html.P(
                    "The PCA chart gives another view of the same results. Each dot is a township; greater separation between colors implies more distinct profiles. "
                    "All features are min–max scaled (0–1) before clustering so each has equal weight. Some overlap between clusters is expected."
                ),
            ], style={"opacity": 0.85})
        ),
        panel(
            "Cluster Structure in Two Dimensions (PCA Projection)", 
            dcc.Loading(
                html.Div([
                    dcc.Graph(id=f"{PAGE_ID}-pca", className="graph-medium"),
                ]),
                className="dash-loading",
                type="default"
            )
        )
    ])

    last_updated = fmt_date(df["event_date"].max())
    foot = html.Div([
        "Data: ", html.A("ACLED", href="https://acleddata.com", target="_blank", rel="noopener noreferrer"),
        f" • Last updated {last_updated} • ",
        "Features: user-selected detailed event counts + fatalities sum + civilian targeting count (min-max scaled)"
    ], style={"opacity": 0.8})

    # Store feature labels for use in callback
    store = dcc.Store(id=f"{PAGE_ID}-feature-labels", data=labels)

    return html.Div([
        head, 
        store,
        page_shell(filters=filters, left_map=left, right_content=right, footnote=foot, page_class="clustering-page")
    ])

@dash.callback(
    Output(f"{PAGE_ID}-map", "figure"),
    Output(f"{PAGE_ID}-pca", "figure"),
    Output(f"{PAGE_ID}-status", "children"),
    Input(f"{PAGE_ID}-run", "n_clicks"),
    Input(f"{PAGE_ID}-month-start", "value"),
    Input(f"{PAGE_ID}-month-end", "value"),
    Input(f"{PAGE_ID}-actor-type", "value"),
    Input(f"{PAGE_ID}-features", "value"),
    Input(f"{PAGE_ID}-k", "value"),
    State(f"{PAGE_ID}-feature-labels", "data"),
)
def run_clustering(n_clicks, start_mm, end_mm, actor_type, selected_features, k, stored_labels):
    df = load_acled_main()
    geo = load_geojson()
    
    # Create empty figures
    empty_map = px.choropleth(template="plotly_white").update_layout(height=740, title="Click 'Run clustering' to start")
    empty_pca = px.scatter(template="plotly_white").update_layout(height=380, title="")
    
    if not start_mm or not end_mm:
        return empty_map, empty_pca, "Select time range"
        
    if end_mm < start_mm:
        start_mm, end_mm = end_mm, start_mm
        
    d1, _ = month_bounds(start_mm)
    _, d2 = month_bounds(end_mm)
    
    # Filter by actor type
    if actor_type and actor_type != "All":
        df = df[df["primary_actor_type"] == actor_type]
    
    if df.empty:
        return empty_map.update_layout(title="No data after filters"), empty_pca, "No data"
    
    # Aggregate features
    agg = aggregate_features(df, d1, d2)
    # Normalize ID type
    if not agg.empty:
        agg["Tsp_Pcode"] = agg["Tsp_Pcode"].astype(str).str.strip()
    if agg.empty or len(agg) < 3:
        return empty_map.update_layout(title="Not enough data points"), empty_pca, "Not enough data"
    
    # Validate selected features
    if not selected_features or len(selected_features) < 2 or len(selected_features) > 7:
        return empty_map.update_layout(title="Please select 2-7 features"), empty_pca, "Select 2-7 features"
    
    # Check if selected features exist in the data
    available_features = get_feature_columns(agg)
    valid_selected = [f for f in selected_features if f in available_features]
    
    if len(valid_selected) < 2:
        return empty_map.update_layout(title="Selected features not available in data"), empty_pca, "Features not available"
    
    # Use stored labels or create new ones if not available
    if stored_labels:
        labels = stored_labels
    else:
        labels = create_feature_labels(valid_selected)
    
    # Ensure FULL GEO COVERAGE: add missing townships (filled with 0s)
    # This prevents map holes when some features are unavailable for certain areas
    filled = ensure_full_geoindex(
        agg, geo,
        id_col="Tsp_Pcode",
        feature_key="properties.TS_PCODE",
        fill_columns=["admin1","admin2","admin3"] + available_features,
        fill_value=0,
    )
    filled["Tsp_Pcode"] = filled["Tsp_Pcode"].astype(str).str.strip()

    # Remove zero-variance features from the currently selected list (based on filled data)
    feature_data_full = filled[valid_selected].values
    feature_vars = np.var(feature_data_full, axis=0)
    final_features = [col for i, col in enumerate(valid_selected) if feature_vars[i] > 0]
    if len(final_features) < 2:
        return empty_map.update_layout(title="Selected features have no variance"), empty_pca, "No feature variance"

    # Determine which rows contain any signal (avoid clustering all-zero rows)
    has_data_mask = (filled[final_features].sum(axis=1) > 0).values
    data_rows = filled.loc[has_data_mask]
    if data_rows.empty or len(data_rows) < 3:
        # Not enough informative rows; still render a map with 'No data' everywhere
        plot_data = filled.copy()
        plot_data["cluster"] = "No data"
        palette = {"No data": "#bdbdbd"}
        # Build township name labels from GeoJSON (authoritative), fallback to admin3
        name_map = {}
        for feat in geo.get("features", []):
            p = feat.get("properties", {}) or {}
            code = str(p.get("TS_PCODE") or "")
            if not code:
                continue
            val = p.get("TS")
            name_map[code] = val.strip() if isinstance(val, str) else ""
        plot_data["township_label"] = plot_data["Tsp_Pcode"].map(name_map).fillna("")

        # Prepare custom data as named columns to avoid shape issues
        plot_data["cd_label"] = plot_data["township_label"]
        custom_cols_nd = ["cd_label"]
        map_fig = px.choropleth(
            plot_data,
            geojson=geo,
            locations="Tsp_Pcode",
            featureidkey="properties.TS_PCODE",
            color="cluster",
            color_discrete_map=palette,
            template="plotly_white",
            # Pass column names so Plotly Express pulls data from data_frame
            custom_data=custom_cols_nd,
        )
        apply_tight_geos(map_fig, geo, height=740, show_colorbar=False)
        map_fig.update_layout(
            title="Township clusters (insufficient data)",
            margin=dict(l=6, r=6, t=40, b=6),
            legend_title_text="Cluster",
            legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
            hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"),
        )
        map_fig.update_traces(hovertemplate="Township: %{customdata[0]}<br>Data: none<extra></extra>")
        return map_fig, empty_pca, "Not enough informative rows"

    # Scale features on informative rows only
    X = data_rows[final_features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure k doesn't exceed number of informative samples
    k = min(int(k), len(data_rows))

    # Run K-means on informative rows
    try:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        cluster_labels_inf = kmeans.fit_predict(X_scaled)

        # Silhouette score on informative rows
        if k > 1 and len(data_rows) > k:
            sil_score = silhouette_score(X_scaled, cluster_labels_inf)
            status = f"k={k}, silhouette={sil_score:.3f}"
        else:
            status = f"k={k}, silhouette=n/a"
    # Do not append feature list to status (user requested concise status)

    except Exception as e:
        return empty_map.update_layout(title=f"Clustering failed: {str(e)}"), empty_pca, "Clustering failed"

    # Build full plot_data with a 'No data' bucket for non-informative rows
    plot_data = filled.copy()
    cluster_col = np.array(["No data"] * len(plot_data), dtype=object)
    cluster_col[has_data_mask] = cluster_labels_inf.astype(str)
    plot_data["cluster"] = cluster_col
    
    # Create color mapping (plus 'No data' grey)
    palette = {str(i): CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)}
    palette["No data"] = "#bdbdbd"
    
    # Create choropleth map
    # Build custom hover via custom_data; no need to use hover_data dict
    
    # Build township name from GeoJSON (authoritative) and fallback to admin3 for any missing
    name_map = {}
    for feat in geo.get("features", []):
        p = feat.get("properties", {}) or {}
        code = str(p.get("TS_PCODE") or "")
        if not code:
            continue
        val = p.get("TS")
        name_map[code] = val.strip() if isinstance(val, str) else ""
    plot_data["township_label"] = plot_data["Tsp_Pcode"].map(name_map).fillna("")

    # Prepare custom_data via named columns to avoid orientation/length mismatches
    # Structure: [label] + first up-to-5 features + status
    show_feats = final_features[:5]
    data_status = np.where(plot_data["cluster"] == "No data", "none", "available")
    plot_data["cd_label"] = plot_data["township_label"]
    custom_cols = ["cd_label"]
    # Copy feature columns to named custom-data columns to keep order stable
    feat_aliases = []
    for i, feat in enumerate(show_feats):
        alias = f"cd_feat_{i}"
        plot_data[alias] = plot_data[feat]
        feat_aliases.append(alias)
    plot_data["cd_status"] = data_status
    custom_cols = ["cd_label", *feat_aliases, "cd_status"]
    map_fig = px.choropleth(
        plot_data,
        geojson=geo,
        locations="Tsp_Pcode",
        featureidkey="properties.TS_PCODE",
        color="cluster",
        color_discrete_map=palette,
        template="plotly_white",
        # Pass column names so Plotly handles lengths correctly
        custom_data=custom_cols,
    )
    
    # Update map styling
    apply_tight_geos(map_fig, geo, height=740, show_colorbar=False)
    map_fig.update_layout(
        title=f"Township clusters (k={k})",
        margin=dict(l=6, r=6, t=40, b=6),
        legend_title_text="Cluster",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
        hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999")
    )
    
    # Custom hover template showing township and actual values
    # Hover: township label + up to first 5 selected features + data status
    hover_template = "Township: %{customdata[0]}<br>"
    for i, col in enumerate(show_feats):
        # feature values are placed after label at positions 1..n
        hover_template += f"{labels.get(col, col)}: %{{customdata[{i+1}]}}<br>"
    idx_status = len(show_feats) + 1
    hover_template += f"Data: %{{customdata[{idx_status}]}}<extra></extra>"
    map_fig.update_traces(hovertemplate=hover_template)
    
    # Create PCA plot
    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=cluster_labels_inf.astype(str),
            color_discrete_map=palette,
            template="plotly_white",
            labels={"x": "PC 1", "y": "PC 2"},
        )
        
        pca_fig.update_layout(
            title=f"PCA projection (k={k})",
            margin=dict(l=6, r=6, t=36, b=6),
            height=380,
            legend_title_text="Cluster",
            hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999")
        )
        
        # Add township names to PCA hover
        pca_fig.update_traces(
            customdata=data_rows["Tsp_Pcode"].map(name_map).fillna(data_rows["admin3"]).fillna("—").values,
            hovertemplate="Township: %{customdata}<br>PC 1: %{x:.3f}<br>PC 2: %{y:.3f}<extra></extra>"
        )
        
    except Exception as e:
        pca_fig = empty_pca.update_layout(title=f"PCA failed: {str(e)}")
    
    return map_fig, pca_fig, status