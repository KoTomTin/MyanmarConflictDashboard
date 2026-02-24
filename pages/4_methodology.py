"""
Page 4 — Methodology
Loads content from methodology_content.txt (edit that file without touching code).
"""
from pathlib import Path
from dash import dcc, html

ROOT         = Path(__file__).resolve().parents[1]
CONTENT_FILE = ROOT / "methodology_content.txt"


def layout():
    try:
        content = CONTENT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "_Methodology content not found. Please check `methodology_content.txt` in the project root._"

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Methodology", className="page-title"),
                html.Div("How data is sourced, cleaned and transformed for this dashboard",
                         className="page-subtitle"),
            ], className="page-header-left"),
        ], className="page-header"),

        html.Div([
            dcc.Markdown(
                content,
                dangerously_allow_html=False,
                className="method-markdown",
            ),
        ], className="panel panel-body method-panel"),

    ], className="page-wrap")
