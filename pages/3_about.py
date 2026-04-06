"""
Page 3 — About
Loads content from docs/ABOUT.md.
"""
from pathlib import Path
from dash import dcc, html

ROOT         = Path(__file__).resolve().parents[1]
CONTENT_FILE = ROOT / "docs" / "ABOUT.md"


def layout():
    try:
        content = CONTENT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "_About content not found. Please check `docs/ABOUT.md`._"

    return html.Div([
        html.Div([
            html.Div([
                html.H4("About", className="page-title"),
                html.Div("Methods, sources, recoding choices, and contact information.",
                         className="page-subtitle"),
            ], className="page-header-left"),
        ], className="page-header"),

        html.Div([
            dcc.Markdown(
                content,
                dangerously_allow_html=False,
                className="about-markdown",
            ),
        ], className="panel panel-body about-panel"),

    ], className="page-wrap")
