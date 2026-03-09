import importlib
import traceback
import dash
from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc
from flask import request as flask_request

_ov = importlib.import_module("pages.1_overview")
_ac = importlib.import_module("pages.2_actor")
_ab = importlib.import_module("pages.3_about")

PAGE_MAP = {
    "/":      _ov.layout,
    "/actor": _ac.layout,
    "/about": _ab.layout,
}

NAV_ITEMS = [
    {"path": "/",      "label": "Overview"},
    {"path": "/actor", "label": "Actor Analysis"},
    {"path": "/about", "label": "About"},
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Myanmar Conflict Dashboard",
)
server = app.server


def mobile_topnav():
    links = [
        dcc.Link(
            item["label"],
            href=item["path"],
            id=f"mob-nav-{i}",
            className="mob-nav-link",
        )
        for i, item in enumerate(NAV_ITEMS)
    ]
    return html.Div([
        html.Span("MCD", className="mob-brand"),
        html.Div(links, className="mob-nav-links"),
    ], className="mobile-topnav d-flex d-md-none")


def sidebar():
    links = [
        dcc.Link(
            html.Div(item["label"], className="nav-item"),
            href=item["path"],
            id=f"nav-link-{i}",
            className="nav-link-wrap",
        )
        for i, item in enumerate(NAV_ITEMS)
    ]
    return html.Div([
        html.Div([
            html.Div("MCD", className="brand-abbr"),
            html.Div("Myanmar Conflict", className="brand-name"),
            html.Div("Dashboard", className="brand-name"),
            html.Div("Tracking conflict since Feb 2021", className="brand-tagline"),
        ], className="brand-block"),
        html.Div(className="sidebar-divider"),
        html.Nav(links, className="sidebar-nav"),
        html.Div(className="sidebar-divider"),
        html.Div([
            html.Div("Data: ACLED", className="sidebar-meta"),
            html.Div("Myanmar · Feb 2021 – present", className="sidebar-meta"),
            html.Div("Contact: kothomasgye@gmail.com", className="sidebar-meta"),
        ], className="sidebar-footer"),
    ], className="sidebar")


def serve_layout():
    try:
        path = flask_request.path or "/"
    except Exception:
        path = "/"

    fn = PAGE_MAP.get(path, _ov.layout)

    try:
        page_content = fn()
    except Exception as e:
        print(f"[serve_layout] ERROR rendering {path}: {e}")
        traceback.print_exc()
        page_content = html.Div(f"Error loading page: {e}",
                                style={"color": "red", "padding": "20px"})

    return dbc.Container([
        dcc.Location(id="url", refresh=False),
        mobile_topnav(),
        dbc.Row([
            dbc.Col(sidebar(), md=2, className="g-0 d-none d-md-flex"),
            dbc.Col(
                html.Main(
                    dcc.Loading(
                        html.Div(page_content, id="page-content"),
                        id="page-loading",
                        type="dot",
                        color="#2563eb",
                        delay_show=120,   # only show spinner if load takes >120ms
                    ),
                    className="main",
                ),
                xs=12, md=10, className="g-0",
            ),
        ], className="g-0"),
    ], fluid=True)


app.layout = serve_layout


# ── Navigate to page ──────────────────────────────────────────────────────────
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def render_page(pathname):
    print(f"[render_page] navigating to: {pathname!r}")
    try:
        fn = PAGE_MAP.get(pathname or "/")
        if fn:
            result = fn()
            print(f"[render_page] {pathname!r} rendered OK")
            return result
        print(f"[render_page] unknown path {pathname!r}, falling back to overview")
        return _ov.layout()
    except Exception as e:
        print(f"[render_page] ERROR for {pathname!r}: {e}")
        traceback.print_exc()
        return html.Div(f"Error: {e}", style={"color": "red", "padding": "20px"})


# ── Highlight active nav link ─────────────────────────────────────────────────
@callback(
    [Output(f"nav-link-{i}", "className") for i in range(len(NAV_ITEMS))]
    + [Output(f"mob-nav-{i}", "className") for i in range(len(NAV_ITEMS))],
    Input("url", "pathname"),
)
def highlight_active_nav(pathname):
    pathname = pathname or "/"
    desktop = [
        "nav-link-wrap nav-link-wrap--active" if item["path"] == pathname
        else "nav-link-wrap"
        for item in NAV_ITEMS
    ]
    mobile = [
        "mob-nav-link mob-nav-link--active" if item["path"] == pathname
        else "mob-nav-link"
        for item in NAV_ITEMS
    ]
    return desktop + mobile


if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, port=8050)
