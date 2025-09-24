# pages/2_actor.py
from __future__ import annotations
import dash
from dash import html, dcc, Input, Output

from components.layout import page_shell
from components.filters import month_range
from components.figures import empty_fig
from components.loaders import load_acled_main
from components.utils_dates import month_options, default_range_earliest_latest, month_bounds
from components.ui import section_header, panel
from components.actor_utils import build_network_figure, build_tactic_tiles

dash.register_page(__name__, path="/actor-network", name="Actor Analysis")
PAGE_ID = "actor"

def layout():
    df = load_acled_main()
    mopts = month_options(df, "event_date")
    mstart, mend = default_range_earliest_latest(df, "event_date")

    header = section_header("Actor Analysis (Armed Conflicts)", "Interactions & tactics")

    # Filters row: time range + explainer box filling remaining space
    filters = [
        month_range(PAGE_ID, mopts, mstart, mend),
        html.Div(
            [
                html.Div("About this page", style={"fontWeight":"600","marginBottom":"4px"}),
                html.Div(
                    "Explore which actors interact most (left) and their common tactics (right). "
                    "Tiles show percentages: the top chart normalizes by tactic (columns), "
                    "the bottom normalizes by actor (rows).",
                    style={"opacity":0.8}
                ),
            ],
            className="panel panel-body",  # lightweight styled note
            style={"padding":"8px 10px"}
        ),
    ]

    left = panel(
        "Actor interaction network (top 10 connections)",
        body=dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-net", className="graph-map-tall"), className="dash-loading")
    )

    right = html.Div(
        [
            panel(
                "Which tactics are most common overall? (share within each tactic)",
                body=dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-heat-col", className="graph-medium"), className="dash-loading")
            ),
            html.Div(style={"height":"10px"}),
            panel(
                "Which tactics each actor uses most? (share within actor)",
                body=dcc.Loading(dcc.Graph(id=f"{PAGE_ID}-heat-row", className="graph-short"), className="dash-loading")
            ),
        ]
    )

    foot = html.Div("Data: ACLED • Top 10 interactions; 'Unidentified' actors excluded; PDF variants collapsed.",
                    style={"textAlign":"right","opacity":0.8})

    return html.Div([header, page_shell(filters=filters, left_map=left, right_content=right, footnote=foot, page_class="actor-page")])

@dash.callback(
    Output(f"{PAGE_ID}-net","figure"),
    Output(f"{PAGE_ID}-heat-col","figure"),
    Output(f"{PAGE_ID}-heat-row","figure"),
    Input(f"{PAGE_ID}-month-start","value"),
    Input(f"{PAGE_ID}-month-end","value"),
)
def render_page(start_mm, end_mm):
    df = load_acled_main()
    if not start_mm or not end_mm:
        empty = empty_fig("No data"); return empty, empty, empty
    if end_mm < start_mm:
        start_mm, end_mm = end_mm, start_mm
    d1, _ = month_bounds(start_mm); _, d2 = month_bounds(end_mm)

    f = df[(df["event_date"] >= d1) & (df["event_date"] <= d2)].copy()
    f = f[f["key_event"] == "Armed conflict"].copy()

    if f.empty:
        empty = empty_fig("No data in this range"); return empty, empty, empty

    net_fig = build_network_figure(f)
    heat_col, heat_row = build_tactic_tiles(f)
    # Remove internal figure titles (panel headers already provide them) and wrap x tick labels
    net_fig.update_layout(height=700, margin=dict(l=6,r=6,t=6,b=6), template="plotly_white", title=None,
                          hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"))
    heat_col.update_layout(height=360, margin=dict(l=6,r=6,t=6,b=40), template="plotly_white", title=None,
                           hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"))
    heat_row.update_layout(height=360, margin=dict(l=6,r=6,t=6,b=40), template="plotly_white", title=None,
                           hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"))
    heat_col.update_xaxes(tickangle=35, automargin=True)
    heat_row.update_xaxes(tickangle=35, automargin=True)
    return net_fig, heat_col, heat_row