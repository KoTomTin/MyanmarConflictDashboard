# Design Benchmarks

Last updated: 2026-04-04

## Purpose

This note captures what we can learn from professional dashboard systems and mature open-source dashboard codebases without copying any one product's visual style.

The goal is to define *transferable design principles* for the Myanmar Conflict Dashboard.

## What Professional Dashboards Do Better

Across strong public dashboards and mature dashboard platforms, the same patterns repeat:

- one dominant focal surface appears immediately
- controls support the analysis instead of dominating the first screen
- supporting charts live in a clearly subordinate rail or lower band
- typography, spacing, and card structure follow one system
- filters update linked views predictably and with low friction
- annotations clarify interpretation without covering the data

In practice, the dashboard should feel like one composed canvas, not a page of stacked sections.

## Relevant Benchmarks

### IBM Carbon dashboard guidance

Most useful takeaways:

- establish a strong hierarchy
- keep spacing and layout consistent across charts
- link views so interactions update related views
- use annotations carefully and avoid obscuring data

Why it matters here:

- our map should be the primary surface
- our side charts should support the map, not compete with it
- milestone labels should never sit on top of the trend line in a way that blocks reading

Source:

- https://carbondesignsystem.com/data-visualization/dashboards/

### Plotly Dash sample apps

Most useful takeaways:

- production-grade Dash apps usually keep the shell simple
- filters stay close to the visual they affect
- pages avoid large dead zones above the first chart
- map-heavy apps tend to privilege the chart canvas over explanatory chrome

Why it matters here:

- our current Overview still spends too much of the first screen on shell and controls
- the first visible analytical object should be the map, not header/filter furniture

Source:

- https://github.com/plotly/dash-sample-apps

### Ant Design Pro

Most useful takeaways:

- compact shell
- stable spacing system
- consistent card rhythm
- filters and metrics are presented as product UI, not raw form elements

Why it matters here:

- our dashboard needs a tighter, more deliberate shell
- the current page still reads like a prototype assembled from good parts

Source:

- https://github.com/ant-design/ant-design-pro

### Apache Superset and Metabase

Most useful takeaways:

- dashboards feel professional when the data canvas wins over the chrome
- controls are compact and composable
- the interface supports exploration without making the screen look like a form

Why it matters here:

- the Overview page should act like a real dashboard, not a report header plus filters plus charts
- compact control bars are preferable to large filter cards for desktop

Sources:

- https://github.com/apache/superset
- https://github.com/metabase/metabase

## Design Direction For This Project

We should not visually imitate IISS or any other single product.

We *should* adopt the structural qualities that make professional dashboards feel finished:

- compact page header
- compact freshness row
- one-line or side-rail controls
- immediate visual payoff above the fold
- one dominant map stage
- supporting metrics and trend views clearly secondary to that map stage
- consistent naming and microcopy across pages

## Specific Implications For Overview

### Layout

- the map should begin on the first screen
- the filter experience should be a slim toolbar or left control rail, not a large form card
- the KPI system should support the map rather than pushing it downward
- the sidebar should be reevaluated because it currently consumes a lot of width relative to the content

### Visual hierarchy

- the map is primary
- event trend is secondary
- event-type composition is tertiary
- trust/freshness cues should be visible but quiet

### Copy

- prefer `Event Types` or `Incident Types`, not `Conflict Types`

Reason:

- the Overview includes protests, arrests, displacement, looting/property destruction, and strategic developments, not only violent conflict categories

Current recommendation:

- `Event Types` is the clearest public-facing label
- `Incident Types` is acceptable
- `Conflict Types` should be avoided

### Controls

- auto-apply is preferable to `Apply` for this product
- `Reset` should remain
- explanatory notes should appear only when needed, as close to the relevant control as possible

## Note On `Others`

The added `acled_cleaned.csv` suggests that `Others` in that export is dominated by strategic-development records, especially:

- changes to group activity
- disrupted weapons use
- headquarters or base establishment
- non-violent transfer of territory
- agreements
- a small number of riot records

However, the current pipeline code in `pipeline/pipeline.py` classifies some categories differently, including `Looting/property destruction` as its own dashboard category. That means the CSV and the current code may not represent exactly the same export logic.

For UI wording, we should therefore:

- avoid saying `Others` includes any category that is already visible as its own filter option
- describe `Others` as a residual bucket for activity not shown separately in the named dashboard groups

## Working Rule

Do not ask: "How do we make this look like IISS?"

Ask instead:

- what is the primary visual?
- what can we remove from the first screen?
- what belongs in controls versus interpretation?
- what makes the page feel composed rather than stacked?

