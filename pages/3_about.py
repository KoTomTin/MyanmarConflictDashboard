"""
Page 3 — About
Loads content from docs/ABOUT.md.
"""
from pathlib import Path
from dash import dcc, html
from components.page_bits import data_disclaimer

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
                html.H1("About", className="page-title"),
                html.Div("Purpose, methods, project status, limitations, and academic context for the Myanmar Conflict Dashboard.",
                         className="page-subtitle"),
                html.Div([
                    html.Span("Also see", className="page-link-label"),
                    html.A("Overview", href="/"),
                    html.A("Actor Analysis", href="/actor"),
                    html.A("Township Alerts", href="/alerts"),
                ], className="page-link-row"),
                html.Div([
                    html.Div("Prototype · work in progress", className="hero-pill hero-pill--prototype"),
                ], className="hero-pill-row hero-pill-row--tight"),
            ], className="page-header-left"),
        ], className="page-header"),

        html.Div([
            dcc.Markdown(
                content,
                dangerously_allow_html=False,
                className="about-markdown",
            ),
        ], className="panel panel-body about-panel"),

        data_disclaimer(),

    ], className="page-wrap")
