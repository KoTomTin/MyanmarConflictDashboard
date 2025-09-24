# components/filters.py
from dash import html, dcc

# components/filters.py  (inside month_range)
from dash import html, dcc

def month_range(page_id, month_opts, start_val, end_val):
    return html.Div(
        [
            html.Label("Time range"),
            html.Div(  # ↓ wrapper makes the two inputs sit close
                [
                    dcc.Dropdown(
                        id=f"{page_id}-month-start",
                        options=month_opts,
                        value=start_val,
                        clearable=False,
                        style={"width": "120px"},
                    ),
                    dcc.Dropdown(
                        id=f"{page_id}-month-end",
                        options=month_opts,
                        value=end_val,
                        clearable=False,
                        style={"width": "120px"},
                    ),
                ],
                className="month-range-group"
            ),
        ]
    )