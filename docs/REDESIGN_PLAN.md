# Dashboard Redesign Plan

Last updated: 2026-04-03

## Why We Are Restarting

We are treating the next redesign as a clean restart rather than incremental tweaking. The previous local UI iterations surfaced real issues:

- the page hierarchy was not stable
- the map container/layout balance kept fighting the data story
- trust and methodology cues were still too weak
- mobile and desktop needs were being blended rather than designed separately
- design decisions were being made without a fixed benchmark set

This plan resets the process and ties every major design decision to either:

- evidence from the literature in [LITERATURE_REVIEW.md](/Users/kaykay/Desktop/MyanmarConflictDashboard/docs/LITERATURE_REVIEW.md)
- official Dash capabilities and constraints
- strong Dash exemplars and public-facing analytical apps

## Benchmark Set We Will Use

### Literature and methodology

- [Dashboard Literature Review](/Users/kaykay/Desktop/MyanmarConflictDashboard/docs/LITERATURE_REVIEW.md)
- public-health dashboard usability checklist work
- conflict-event data methodology and uncertainty literature
- choropleth and animated-map design literature

### Official Dash and Plotly references

- [Dash sample apps repository](https://github.com/plotly/dash-sample-apps)
- [Plotly Dash app examples](https://plotly.com/examples/)
- [Dash Layout docs](https://dash.plotly.com/layout)
- [Dash multi-page apps docs](https://dash.plotly.com/urls?trk=public_post_comment-text)
- [Dash `dcc.Graph` docs](https://dash.plotly.com/dash-core-components/graph)
- [Dash `dcc.Store` docs](https://dash.plotly.com/dash-core-components/store)
- [Dash clientside callbacks docs](https://dash.plotly.com/clientside-callbacks)
- [Dash performance docs](https://dash.plotly.com/performance)
- [Dash background callbacks docs](https://dash.plotly.com/background-callbacks?tab=bi-templates)

### Dash exemplar apps reviewed

The following official/public examples are useful because they show mature Dash patterns, not because we want to copy their appearance directly.

- [OECD Pensions Explorer](https://plotly.com/examples/)
  Why it matters:
  strong framing question, cross-country comparison, narrative analytics

- [Femicide in Bolivia](https://plotly.com/examples/)
  Why it matters:
  public-interest violence mapping, morally serious tone, geographic storytelling

- [Centre de Controle des Incidents](https://plotly.com/examples/)
  Why it matters:
  live operations dashboard pattern, monitoring-oriented hierarchy

- [Uber Rides Geospatial Data Viz](https://plotly.com/examples/)
  Why it matters:
  large-scale geospatial exploration, map-first interaction model

- [Coffee Flavor Analysis Dashboard](https://plotly.com/examples/)
  Why it matters:
  structured drill-down, hierarchy and taxonomy exploration

- [Dash sample apps repo](https://github.com/plotly/dash-sample-apps)
  Why it matters:
  reusable implementation patterns for layout, callbacks, stores, loading, maps, and multi-page architecture

### Additional quality signals from Plotly’s community/explore guidance

Plotly’s own Explore Page guidance is useful because it effectively states what the platform highlights as strong examples:

- the app should tell a unique story
- it should look polished
- practical real-world use cases are favored
- live data is encouraged
- strong analytics beyond raw exploration are encouraged
- content should be easy to access without unnecessary gates

Source:
- [Share Your App - Explore Page - March 2026](https://community.plotly.com/t/share-your-app-explore-page-march-2026/96324)

## Working Design Principles For This Project

These are the design rules we will use unless we consciously decide to break one.

### 0. Public explainer first

- The redesign should optimize for public explanation before expert exploration.
- A first-time visitor should understand the basic message without needing dashboard fluency.
- Advanced interaction should support the story, not define the product.

### 1. One page, one primary job

- Overview answers: what is happening spatially and temporally in Myanmar right now or in the selected period?
- Actor Analysis answers: where and how is one actor engaged, and with whom?
- About answers: what this dashboard is, who it is for, and how to interpret it responsibly.

### 2. Static first, animation second

- The default map mode should be static.
- Animation is an optional exploratory layer, not the main explanatory surface.

### 3. Map is important, but not self-sufficient

- The map should answer “where.”
- The right-hand or supporting panels should answer “how much,” “when,” and “what kind.”
- No page should depend on the user correctly interpreting the choropleth alone.

### 4. Trust must be visible, not buried

- `Data through` and `Last checked with ACLED` stay separate.
- Fatality language must remain method-aware.
- Source, methodology, and limits should be visible at the page level.

### 5. Performance is part of design

- Slow pages are bad UX even if they are visually strong.
- Heavy callbacks and oversized figure payloads should be treated as design failures, not only engineering issues.

### 6. Mobile is a separate product surface

- Filters, hierarchy, and chart density must be designed separately for mobile.
- Desktop patterns should not simply collapse into stacked versions of the same UI.

## Restart Plan

### Phase 1. Reset the product brief

Goal:
Define exactly what this dashboard is for before changing the UI.

Deliverables:

- short audience definition
- page-by-page purpose statement
- list of top user questions each page should answer
- list of things the dashboard should explicitly not try to do

Output file:

- `docs/PRODUCT_BRIEF.md`

### Phase 2. Re-audit the data and method layer

Goal:
Make sure the information architecture reflects what the data can responsibly support.

Tasks:

- audit all metric labels against ACLED methodology
- check all fatality wording
- define standard trust copy used across pages
- define map caveats for township-level choropleths
- define terminology for actor roles, incidents, fatalities, and “activity”

Output files:

- `docs/METRIC_GLOSSARY.md`
- update `docs/METHODOLOGY.md`

### Phase 3. Benchmark the current app against the rubric

Goal:
Understand exactly where the current app fails before redesigning.

Tasks:

- audit Overview against the literature-based rubric
- audit Actor Analysis separately
- review desktop and mobile as separate experiences
- record page-level issues by severity

Output file:

- `docs/UI_AUDIT.md`

### Phase 4. Redesign the information architecture

Goal:
Decide the page structure before styling details.

Tasks:

- choose what lives above the fold on each page
- define the role of hero area, filter area, map area, and support rail
- decide which content stays on Overview and which moves elsewhere
- decide whether some right-rail cards should be merged or removed

Output file:

- `docs/IA_AND_WIREFRAMES.md`

### Phase 5. Create low-fidelity wireframes first

Goal:
Settle the page composition without getting distracted by polish.

Tasks:

- wireframe Overview desktop
- wireframe Overview mobile
- wireframe Actor desktop
- wireframe Actor mobile
- wireframe About improvements if needed

Rule:

- no visual polish work until wireframes are accepted

### Phase 6. Build a new design system for this app

Goal:
Create a small but deliberate visual system rather than continuing ad hoc CSS edits.

Tasks:

- define typography scale
- define spacing scale
- define card system
- define chips, controls, and filter patterns
- define trust/status badge patterns
- define map container rules
- define mobile-specific layout rules

Primary implementation file:

- `assets/style.css`

### Phase 7. Rebuild the Overview page from scratch locally

Goal:
Treat Overview as the flagship page and solve it well before touching Actor Analysis again.

Sequence:

1. rebuild layout skeleton
2. add trust and status layer
3. add filters
4. add static map
5. add support charts
6. add microcopy and interpretation help
7. reintroduce animated mode only if the static version is already strong

Primary implementation file:

- `pages/1_overview.py`

### Phase 8. Rebuild Actor Analysis to match the new system

Goal:
Bring Actor Analysis into alignment with the new logic and visual system, but preserve its distinct analytical purpose.

Sequence:

1. adapt the approved Overview system
2. keep actor-specific KPIs and allied-actor logic
3. simplify the map/support relationship
4. standardize labels and trust cues

Primary implementation file:

- `pages/2_actor.py`

### Phase 9. Performance and responsiveness pass

Goal:
Make the redesigned app feel fast and stable.

Tasks:

- profile initial payload size
- profile major callbacks
- add or improve caching where useful
- use clientside updates where interaction is UI-only
- check graph responsiveness and container sizing
- test cold-load and repeat interaction feel

Primary references:

- Dash performance docs
- Dash clientside callback docs
- Dash `dcc.Graph` responsive guidance

### Phase 10. Accessibility and trust pass

Goal:
Make the app legible, credible, and robust.

Tasks:

- review contrast and type sizes
- review keyboard and focus behavior where practical
- review wording for overclaiming precision
- review explanation of event/fatality limits
- review page-level source and date communication

### Phase 11. Local evaluation before any push

Goal:
Only push after the redesign works locally as a product, not just as code.

Checklist:

- first impression clear in 10 seconds
- map interpretation clear
- filters understandable
- mobile usable
- load feels acceptable
- no major trust-language problems
- no obvious visual imbalance across pages

## Implementation Order I Recommend

This is the exact order I recommend we follow.

1. Create `PRODUCT_BRIEF.md`
2. Create `METRIC_GLOSSARY.md`
3. Audit current UI into `UI_AUDIT.md`
4. Produce low-fidelity wireframes in `IA_AND_WIREFRAMES.md`
5. Rebuild Overview locally
6. Review Overview together
7. Rebuild Actor Analysis locally
8. Run performance/mobile/trust pass
9. Only then consider push and redeploy

## What We Should Not Do

- do not keep tweaking CSS without a page model
- do not redesign Overview and Actor simultaneously from the start
- do not use the animated map as the main storytelling surface
- do not treat performance fixes as something to do only at the end
- do not push unfinished UI experiments to production

## Recommendation For The Next Working Session

Start with Phase 1 and Phase 2 only.

That means:

1. define the dashboard’s audiences and page jobs
2. define the metric language and trust copy

Only after those are settled should we touch layout and styling again.

## Release Policy

- redesign work happens locally first
- nothing is pushed or deployed until explicitly approved
- local review and iteration are part of the design process, not a final afterthought
