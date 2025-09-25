from __future__ import annotations
import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px

from components.layout import page_shell
from components.ui import panel, section_header
from components.cards import kpi_card
from components.filters import month_range
from components.figures import empty_fig
from components.loaders import load_acled_main, load_geojson
from components.utils_dates import month_options, default_range_earliest_latest, month_bounds
from components.utils_format import fmt_date
from components.colors import SEQUENTIAL_BLUES_ZERO_GREY, KEY_EVENT_COLORS
from components.map_utils import apply_tight_geos, ensure_full_geoindex, filter_geo_by_property

dash.register_page(__name__, path="/", name="Overview")
PAGE_ID = "ov"

def _fmt_compact(n):
    try: n = float(n)
    except: return "0"
    s = "-" if n < 0 else ""; n = abs(n)
    return (f"{s}{n/1e9:.2f}B" if n>=1e9 else
            f"{s}{n/1e6:.2f}M" if n>=1e6 else
            f"{s}{n/1e3:.2f}K" if n>=1e3 else
            f"{s}{int(n):,}")

def layout():
    df = load_acled_main()

    # options
    mopts = month_options(df, "event_date")
    mstart, mend = default_range_earliest_latest(df, "event_date")
    actor_types = ["All"] + sorted(df["primary_actor_type"].dropna().astype(str).str.strip().unique().tolist())
    key_events  = ["All"] + sorted(df["key_event"].dropna().astype(str).str.strip().unique().tolist())
    civ_tgts    = ["All"] + (sorted(df["civilian_targeting"].dropna().astype(str).str.strip().unique().tolist())
                             if "civilian_targeting" in df.columns else [])
    admin1s     = ["All"] + sorted(df["admin1"].dropna().astype(str).str.strip().unique().tolist())

    header = section_header("Overview", "Explore conflict patterns in Myanmar through interactive filters, maps, KPIs, and trend charts.")

    # Provide raw filter controls (page_shell will wrap in .row-filters)
    filters = [
        month_range(PAGE_ID, mopts, mstart, mend),
        html.Div([html.Label("Primary actor type"),
                  dcc.Dropdown(id=f"{PAGE_ID}-actor-type",
                               options=[{"label":v,"value":v} for v in actor_types],
                               value="All", clearable=False)]),
        html.Div([html.Label("Key event"),
                  dcc.Dropdown(id=f"{PAGE_ID}-key-event",
                               options=[{"label":v,"value":v} for v in key_events],
                               value="All", clearable=False)]),
        html.Div([html.Label("Civilian targeting"),
                  dcc.Dropdown(id=f"{PAGE_ID}-civ-tgt",
                               options=[{"label":v,"value":v} for v in civ_tgts],
                               value="All", clearable=False)]),
        html.Div([html.Label("Region (Admin1)"),
                  dcc.Dropdown(id=f"{PAGE_ID}-admin1",
                               options=[{"label":v,"value":v} for v in admin1s],
                               value="All", clearable=False)]),
    ]

    # Provide KPI cards directly (page_shell will wrap in .row-kpis)
    kpis = [
        kpi_card("Event count", f"{PAGE_ID}-kpi-events"),
        kpi_card("Townships affected", f"{PAGE_ID}-kpi-tsp"),
        kpi_card("Population affected", f"{PAGE_ID}-kpi-pop"),
        kpi_card("Fatalities", f"{PAGE_ID}-kpi-fat"),
    ]

    left = panel(
        "Events by township",
        dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-map", className="graph-map-tall"), className="dash-loading"),
    )

    right = html.Div(
        [
            panel(
                "Weekly events by key event",
                dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-weekly", className="graph-medium"),
                            className="dash-loading"),
            ),
            panel(
                "Top detailed event types (sorted)",
                dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-detailbar", className="graph-short"),
                            className="dash-loading"),
            ),
        ]
    )

    last_updated = fmt_date(df["event_date"].max())
    foot = html.Div(["Data: ACLED • Last updated ", html.Strong(last_updated)])

    return html.Div([header, page_shell(filters=filters, kpis=kpis, left_map=left, right_content=right, footnote=foot)])

@dash.callback(
    Output(f"{PAGE_ID}-kpi-events", "children"),
    Output(f"{PAGE_ID}-kpi-tsp", "children"),
    Output(f"{PAGE_ID}-kpi-pop", "children"),
    Output(f"{PAGE_ID}-kpi-fat", "children"),
    Output(f"{PAGE_ID}-map", "figure"),
    Output(f"{PAGE_ID}-weekly", "figure"),
    Output(f"{PAGE_ID}-detailbar", "figure"),
    Input(f"{PAGE_ID}-month-start", "value"),
    Input(f"{PAGE_ID}-month-end", "value"),
    Input(f"{PAGE_ID}-actor-type", "value"),
    Input(f"{PAGE_ID}-key-event", "value"),
    Input(f"{PAGE_ID}-civ-tgt", "value"),
    Input(f"{PAGE_ID}-admin1", "value"),
)
def update_overview(start_mm, end_mm, actor_type, key_event, civ_target, admin1):
    df = load_acled_main()
    geo = load_geojson()

    if not start_mm or not end_mm:
        empty = empty_fig("No data"); return "—","—","—","—", empty, empty, empty
    if end_mm < start_mm:
        start_mm, end_mm = end_mm, start_mm
    d1, _ = month_bounds(start_mm); _, d2 = month_bounds(end_mm)

    f = df[(df["event_date"] >= d1) & (df["event_date"] <= d2)].copy()
    if actor_type and actor_type != "All" and "primary_actor_type" in f.columns:
        f = f[f["primary_actor_type"] == actor_type]
    if key_event and key_event != "All":
        f = f[f["key_event"] == key_event]
    if civ_target and civ_target != "All" and "civilian_targeting" in f.columns:
        f = f[f["civilian_targeting"] == civ_target]
    if admin1 and admin1 != "All":
        f = f[f["admin1"] == admin1]

    if f.empty:
        empty = empty_fig("No data in this range"); return "0","0","0","0", empty, empty, empty

    events_n = len(f)
    tsp_affected = f["Tsp_Pcode"].nunique() if "Tsp_Pcode" in f.columns else 0
    pop_sum = int(f.drop_duplicates("Tsp_Pcode")["population_size"].sum()) if "population_size" in f.columns else 0
    fat_sum = int(f["fatalities"].sum()) if "fatalities" in f.columns else 0

    # map data (no holes)
    geo_used = filter_geo_by_property(geo, prop="ST", value=admin1) if admin1 and admin1 != "All" else geo
    counts = f.groupby("Tsp_Pcode").size().rename("events").reset_index()
    tnames = f[["Tsp_Pcode","admin3"]].dropna().drop_duplicates("Tsp_Pcode")
    counts = counts.merge(tnames, on="Tsp_Pcode", how="left")
    full = ensure_full_geoindex(counts, geo_used,
                                id_col="Tsp_Pcode", feature_key="properties.TS_PCODE",
                                fill_columns=["events","admin3"], fill_value=0)
    full["admin3"] = full["admin3"].replace(0, None).fillna("—")

    map_fig = px.choropleth(
        full, geojson=geo_used, locations="Tsp_Pcode", featureidkey="properties.TS_PCODE",
        color="events", color_continuous_scale=SEQUENTIAL_BLUES_ZERO_GREY,
        range_color=(0, max(1, int(full["events"].max()))),
        hover_data={"Tsp_Pcode":False,"admin3":True,"events":True},
        template="plotly_white",
    )
    map_fig.update_traces(
        hovertemplate="Township: %{customdata[0]}<br>Total count: %{z}<extra></extra>",
        customdata=full[["admin3"]], marker_line_width=0.4, marker_line_color="#666",
    )
    apply_tight_geos(map_fig, geo_used, height=700, show_colorbar=True)
    map_fig.update_layout(margin=dict(l=4,r=4,t=6,b=6), coloraxis_colorbar=dict(title="Events"))

    # weekly line
    wf = f.copy()
    wf["week"] = wf["event_date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = wf.groupby(["week","key_event"]).size().reset_index(name="count")
    weekly_fig = px.line(weekly, x="week", y="count", color="key_event",
                         color_discrete_map=KEY_EVENT_COLORS, template="plotly_white")
    weekly_fig.update_layout(legend=dict(title_text="", orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
                             margin=dict(l=6,r=6,t=12,b=10), height=340)
    weekly_fig.update_xaxes(title="", tickformat="%b %Y", nticks=10, showgrid=False)
    weekly_fig.update_yaxes(title="", showgrid=True, zeroline=True)
    weekly_fig.update_traces(hovertemplate="Week: %{x|%b %d, %Y}<br>Total count: %{y}<extra></extra>")

    # detailed bar
    if "detailed_event" in f.columns:
        de = f["detailed_event"].astype(str).str.strip().value_counts().head(10).reset_index()
        de.columns = ["detailed_event","count"]
        detail_fig = px.bar(de.sort_values("count", ascending=False), x="detailed_event", y="count",
                            template="plotly_white")
        detail_fig.update_traces(hovertemplate="Detailed event: %{x}<br>Total count: %{y}<extra></extra>", marker_color="#4c78a8")
        detail_fig.update_layout(margin=dict(l=6,r=6,t=10,b=90), xaxis=dict(categoryorder="total descending", automargin=True, title="Detailed event"),
                                 yaxis_title="", showlegend=False, height=340)
        detail_fig.update_xaxes(tickangle=35, tickfont=dict(size=10))
    else:
        detail_fig = empty_fig("No detailed_event field")

    return (_fmt_compact(events_n), _fmt_compact(tsp_affected), _fmt_compact(pop_sum), _fmt_compact(fat_sum),
            map_fig, weekly_fig, detail_fig)
