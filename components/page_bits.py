from dash import html


def data_disclaimer(compact: bool = False):
    acled_link = html.A(
        "ACLED",
        href="https://acleddata.com",
        target="_blank",
        style={"color": "inherit", "textDecoration": "underline"},
    )
    if compact:
        # One line for space-constrained pages; the full caveat lives on About.
        from dash import dcc
        return html.Div([
            "Source: ", acled_link,
            ", reviewed and recoded by our team · a simplified view of reported events, "
            "not a complete or real-time account — ",
            dcc.Link("read how to interpret it", href="/about", className="disclaimer-link"),
        ], className="data-disclaimer data-disclaimer--compact")
    return html.Div([
        "Source: ", acled_link,
        " (Armed Conflict Location & Event Data Project). Our team reviews and analytically recodes parts of the data using local field knowledge where possible, but the dashboard remains a simplified view of reported events. Displayed figures may differ from official ACLED outputs, other published sources, or on-the-ground reports. Interpret it alongside local context and corroborating reporting, not as a complete or real-time account of events.",
    ], className="data-disclaimer")
