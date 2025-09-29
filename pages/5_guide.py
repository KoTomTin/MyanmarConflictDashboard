# pages/5_guide.py
from __future__ import annotations

import dash
from dash import html, dcc

from components.ui import section_header, panel
from components.layout import page_shell


dash.register_page(__name__, path="/guide", name="Dashboard Guide")
PAGE_ID = "guide"


GUIDE_MD = """
### Purpose of the Dashboard

The Myanmar Conflict Dashboard provides an interactive platform for exploring conflict trends in Myanmar. It was designed to address the limitations of existing tools such as ACLED Explorer by offering township-level detail, Myanmar-specific event categories, and analytical modules. The dashboard aims to support humanitarian actors, journalists, civil society organisations, academics, and policymakers in monitoring conflict, planning responses, and raising awareness.

---

### Data Sources

The dashboard integrates the following datasets:
- ACLED (Armed Conflict Location & Event Data Project): event-level data on political violence, protests, and related fatalities.
- MIMU (Myanmar Information Management Unit): official shapefiles, region and township codes.

Coverage spans from February 2021 to August 2025. Updates are planned on a monthly basis, depending on ACLED’s public release schedule.

---

### Key Labels and Categories

To improve contextual relevance, some categories were refined:
- Drone strikes separated from general airstrikes.
- Displacement separated from “other” events.
- People’s Defence Forces (PDFs) coded as a distinct actor group.

Event types include armed conflict, arrests, displacement, protests, and violence against civilians, consistent with ACLED’s methodology but tailored for Myanmar.

---

### Dashboard Structure and Pages
- Overview: KPI cards, choropleth map, time series, and top event-type bar chart.
- Anomalies: township-level surges identified using Median Absolute Deviation (MAD).
- Clustering: township profiles grouped by conflict patterns with PCA scatterplots and maps.
- Forecasting: regional-level short-term projections using XGBoost.
- Actor/Tactic Analysis: network graph of actor relationships and tile charts linking actors to tactics.

---

### How to Use the Dashboard
- Filters: select by time range (month/year), actor type, event type, region, and civilian targeting.
- Navigation: reset button restores default settings.
- Hover text: provides simple explanations of terms and values.
- Downloads: figures can be exported as PNG for reporting.

Note: performance may be slower when loading large datasets or using multiple analytical modules simultaneously.

---

### Good Practices for Users
- Treat dashboard outputs as signals, not exact truths.
- Use insights to identify patterns and trends, then verify with field reports or other sources.
- Township-level maps can guide aid planning, but results should be contextualised with local knowledge.

---

### Limitations
- Reliance on ACLED data, which may under-report events due to internet shutdowns or security risks.
- Static dataset during project phase; monthly updates will be applied in future iterations.
- Forecasting limited to regional level.

---

### Disclaimer

The Myanmar Conflict Dashboard is a non-commercial, academic project developed for research and educational purposes only. All event data is sourced from the Armed Conflict Location & Event Data Project (ACLED), and all geographic data is sourced from the Myanmar Information Management Unit (MIMU). The dashboard does not claim ownership of these datasets, and all rights remain with the original providers.

While every effort has been made to clean and contextualise the data, conflict information in Myanmar is often incomplete or under-reported due to internet shutdowns, security risks for reporters, and political sensitivities. The dashboard should therefore be used as a tool for exploration and awareness, not as a definitive record of events. Users are strongly advised to verify insights with other sources before making operational, political, or advocacy decisions.

The author bears no responsibility for how outputs are interpreted or applied outside the intended research context. The tool is not designed for commercial use, nor does it make political endorsements. Its aim is to support humanitarian awareness, academic study, and peace-building efforts.

---

### Further Information
- ACLED Website: https://acleddata.com 
- ACLED Codebook: https://acleddata.com/methodology/acled-codebook
- ACLED Myanmar Coverage: https://acleddata.com/methodology/myanmar
- MIMU Data: https://themimu.info/
"""


def layout():
    head = section_header("Dashboard Guide", "How to get the most from this dashboard")

    content = panel(
        "About this dashboard",
        dcc.Markdown(GUIDE_MD, link_target="_blank")
    )

    # Use page_shell with a special page_class that makes the body single-column and wider
    body = page_shell(
        filters=None,
        kpis=None,
        left_map=content,
        right_content=None,
        footnote=None,
        page_class="guide-page",
    )
    return html.Div([head, body])
