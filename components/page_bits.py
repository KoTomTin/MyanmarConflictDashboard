from dash import html


def data_disclaimer():
    return html.Div([
        "Source: ",
        html.A(
            "ACLED",
            href="https://acleddata.com",
            target="_blank",
            style={"color": "inherit", "textDecoration": "underline"},
        ),
        " (Armed Conflict Location & Event Data Project). Our team reviews and analytically recodes parts of the data using local field knowledge where possible, but the dashboard remains a simplified view of reported events. Displayed figures may differ from official ACLED outputs, other published sources, or on-the-ground reports. Interpret it alongside local context and corroborating reporting, not as a complete or real-time account of events.",
    ], className="data-disclaimer")
