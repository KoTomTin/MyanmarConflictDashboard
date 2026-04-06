import importlib
import traceback
import dash
from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc
from flask import request as flask_request

_ov = importlib.import_module("pages.1_overview")
_ac = importlib.import_module("pages.2_actor")
_ab = importlib.import_module("pages.3_about")
_al = importlib.import_module("pages.4_alerts")

PAGE_MAP = {
    "/":      _ov.layout,
    "/actor": _ac.layout,
    "/alerts": _al.layout,
    "/about": _ab.layout,
}

NAV_ITEMS = [
    {"path": "/",      "label": "Overview"},
    {"path": "/actor", "label": "Actor Analysis"},
    {"path": "/alerts", "label": "Township Alerts"},
    {"path": "/about", "label": "About"},
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Myanmar Conflict Dashboard",
    update_title=None,
)
server = app.server


def topnav():
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
        html.Div([
            html.Div("Myanmar Conflict Dashboard", className="brand-abbr"),
            html.Div("Tracking conflict since Feb 2021", className="brand-tagline"),
        ], className="topnav-brand"),
        html.Div([
            html.Span("Pages", className="topnav-nav-label"),
            html.Nav(links, className="topnav-links"),
        ], className="topnav-nav"),
    ], className="topnav-shell")


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
        topnav(),
        html.Main(
            html.Div(page_content, id="page-content"),
            className="main",
        ),
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
    [Output(f"mob-nav-{i}", "className") for i in range(len(NAV_ITEMS))],
    Input("url", "pathname"),
)
def highlight_active_nav(pathname):
    pathname = pathname or "/"
    mobile = [
        "mob-nav-link mob-nav-link--active" if item["path"] == pathname
        else "mob-nav-link"
        for item in NAV_ITEMS
    ]
    return mobile


if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, port=8050)
