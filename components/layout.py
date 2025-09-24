from __future__ import annotations
from dash import html

def page_shell(*, filters=None, kpis=None, left_map=None, right_content=None, footnote=None, page_class: str | None = None):
    """Shared page wrapper with a strict 2-column body (left long, right stacked).
    page_class lets pages add a CSS hook (e.g., 'actor-page') for custom widths.
    """
    return html.Div(
        [
            html.Div(filters, className="row-filters"),
            html.Div(kpis, className="row-kpis"),
            html.Div(
                [
                    html.Div(left_map, className="col-left"),
                    html.Div(right_content, className="col-right"),
                ],
                className="row-body",
            ),
            html.Div(footnote, className="row-foot"),
        ],
        className="page-wrap" + (f" {page_class}" if page_class else ""),
    )