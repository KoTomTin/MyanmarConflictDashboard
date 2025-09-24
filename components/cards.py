# components/cards.py
from dash import html

def kpi_card(title: str, value_id: str):
    """Small, reusable KPI card."""
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(id=value_id, children="—", className="kpi-value"),
        ],
        className="kpi-card",
    )