import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.SANDSTONE],
    suppress_callback_exceptions=True,
    title="Conflict Dashboard",
)
server = app.server

def sidebar():
    # Enforce a custom, user-requested order
    desired = ["/", "/actor-network", "/temporal", "/clustering", "/guide"]
    all_pages = list(dash.page_registry.values())
    # Map by path for quick lookup
    by_path = {p.get("path", ""): p for p in all_pages}
    ordered = [by_path[p] for p in desired if p in by_path]
    # Append any others (if present) at the end to avoid losing pages
    remaining = [p for p in all_pages if p not in ordered]
    pages = ordered + remaining
    links = [html.Div(dcc.Link(p["name"], href=p["path"], className="sidebar-link")) for p in pages]
    return html.Div(
        [
            html.H2("MCD", style={"margin":"0 0 6px 0","fontWeight":"800","letterSpacing":"1px"}),
            html.Div("Myanmar Conflict Dashboard", style={"color":"#6b7280","marginBottom":"10px","fontSize":"0.9rem"}),
            html.Hr(),
            html.Nav(links, style={"display":"grid","gap":"8px"}),
            html.Hr(),
            html.Div(
                [
                    html.Div("Created by: Ko Thomas"),
                    html.Div(["Contact: ", html.A("kothomasgye@gmail.com", href="mailto:kothomasgye@gmail.com")]),
                ],
                style={"fontSize":"0.9rem","color":"#6b7280","marginTop":"auto"},
            ),
        ],
        className="sidebar",
    )

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar(), width=2, className="g-0"),
                dbc.Col(
                    html.Div(
                        [
                            html.Main(dash.page_container, className="main"),
                        ]
                    ),
                    width=10, className="g-0"
                ),
            ],
            className="g-0",
        )
    ],
    fluid=True,
)

if __name__ == "__main__":
    # Explicitly enable reloader and hot reload to ensure page edits reflect immediately
    app.run(debug=True, port=8050, use_reloader=True, dev_tools_hot_reload=True)