"""Shared Plotly theme constants and helpers used across all pages.

Single source of truth so the dashboard's editorial look (cream + slate)
stays consistent and we can re-theme in one place.
"""
from __future__ import annotations

import plotly.graph_objects as go

# ── Typography ────────────────────────────────────────────────────────────────
PLOTLY_FONT = "Avenir Next, Segoe UI, Arial, sans-serif"
PLOTLY_DISPLAY = "Iowan Old Style, Palatino Linotype, Book Antiqua, Georgia, serif"

# ── Palette ───────────────────────────────────────────────────────────────────
PLOTLY_TEXT = "#2f4257"          # bumped from #415669 for stronger contrast
PLOTLY_TEXT_MUTED = "#5a6c7e"
PLOTLY_GRID = "#e6ddd0"
PLOTLY_HOVER_BG = "rgba(255,251,246,0.98)"
PLOTLY_HOVER_BORDER = "#d9cfbf"

# Tick / annotation defaults — bumped slightly for legibility
TICK_FONT_SIZE = 11               # was 9–10
ANNOTATION_FONT_SIZE = 10         # was 7.4 (!)


def empty_fig(msg: str = "No data for this selection", height: int = 400) -> go.Figure:
    """Lightweight placeholder figure used while charts hydrate or when filters return nothing."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5, y=0.5, showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=13, color=PLOTLY_TEXT_MUTED, family=PLOTLY_FONT),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_visible=False,
        yaxis_visible=False,
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT),
    )
    return fig


def chart_config(filename: str, *, show_modebar: bool = False) -> dict:
    """Modebar config that exposes only the PNG download button on hover."""
    return {
        "displayModeBar": "hover" if show_modebar else False,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        ],
        "toImageButtonOptions": {
            "format": "png", "filename": filename,
            "height": 700, "width": 1200, "scale": 2,
        },
    }


def hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(79,122,163,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def hover_label() -> dict:
    return dict(
        bgcolor=PLOTLY_HOVER_BG,
        bordercolor=PLOTLY_HOVER_BORDER,
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=12),
    )
