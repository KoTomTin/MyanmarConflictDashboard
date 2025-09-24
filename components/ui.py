from __future__ import annotations
from dash import html

def section_header(title: str, subtitle: str | None = None):
    return html.Div(
        [
            html.H2(title, style={"margin":"0 0 2px 0"}),
            html.Div(subtitle or "", style={"opacity":0.75}),
        ],
        className="page-header",
    )

def panel(title: str, body):
    return html.Div(
        [
            html.Div(title, className="panel-head"),
            html.Div(body, className="panel-body"),
        ],
        className="panel",
    )