# Metric Glossary

Last updated: 2026-04-03

## Purpose

This file defines the standard language the dashboard should use for metrics, labels, trust cues, and interpretation notes.

The goal is to keep the app method-aware and internally consistent. If a label in the UI conflicts with this glossary, the glossary should guide the next revision.

## Naming Principles

- Prefer `reported` when describing event and fatality totals.
- Prefer `estimate` when describing fatalities.
- Keep `Data through` and `Last checked with ACLED` separate.
- Avoid language that implies causality, aggression, victimhood, or precise attribution unless the underlying data support it directly.
- Use shorter UI labels only when a fuller definition exists nearby in supporting copy, a tooltip, or documentation.

## Canonical Freshness Terms

| Term | Standard meaning | Guidance |
| --- | --- | --- |
| `Data through` | The latest event date represented in the processed dataset currently loaded by the app | Use for event coverage, not for pipeline refresh time |
| `Last checked with ACLED` | The most recent successful dashboard check against the ACLED source used by the pipeline | Show in Yangon time when available; use for freshness, not event coverage |

Recommended status line:

`Data through: {latest_event_date} · Last checked with ACLED: {last_successful_check}`

## Canonical Count Terms

| Preferred label | Meaning | Guidance |
| --- | --- | --- |
| `Reported Events` | Count of dashboard event records in the selected scope | Prefer over `Total Conflicts` or `Conflict Total` |
| `Reported Fatality Estimate` | Sum of ACLED reported fatalities in the selected scope | Prefer over `Total Fatalities` when space allows |
| `Most Active Township` | Shorthand for the township with the highest reported event count in the selected scope | Acceptable short label; in docs use fuller explanation |
| `Selected Period` | The current filtered time window shown in the page | Prefer over `Period Total` |

## Canonical Overview Terms

| Preferred label | Meaning | Guidance |
| --- | --- | --- |
| `Conflict Geography` | Township-level map of reported events or reported fatality estimates | Keep map framing geographic, not causal |
| `Event Trend` | Time series of reported events in the selected scope | Be explicit when the chart excludes the current partial month or includes a partial point |
| `Incident Types` | Composition of dashboard event categories in the selected scope | Prefer over `Conflict Types` because the dashboard mixes political violence and protests on Overview |
| `Cumulative View` | Static map summarizing the selected period in one view | Prefer over `Period Total` |
| `Animated View` | Quarter-by-quarter exploratory map animation | Treat as a secondary exploratory mode |

## Canonical Actor Analysis Terms

| Preferred label | Meaning | Guidance |
| --- | --- | --- |
| `Geographic Footprint` | Township-level distribution of filtered actor event records | Avoid implying territorial control |
| `Associated Actors` | Actors recorded on the same side of the same armed-conflict event after dashboard recoding | Do not present as proof of alliance or command relationship |
| `Monthly Engagement Trend` | Time series of the selected actor's event-side records over time | Keep the subtitle explicit about what the side labels mean |

## Actor-Side Terminology

This is the most method-sensitive part of the current app.

### Current problem

The current UI uses labels such as:

- `Offensive Events`
- `Defensive Events`
- `as attacker`
- `as targeted`

These labels are too strong for the underlying ACLED actor fields alone.

ACLED explicitly states that `Actor1` and `Actor2` do not identify the aggressor, the victim, or the side that suffered more casualties. Our current actor-level dataset is built from dashboard-side assignments based on the recoded `primary_actor / assoc_actor_1` side and `secondary_actor / assoc_actor_2` side. That is a dashboard analytical construct, not a native ACLED aggressor variable.

### Recommended standard

For redesign work, prefer:

| Preferred label | Meaning |
| --- | --- |
| `Primary-side Event Records` | Event records in which the selected actor appears on the dashboard's primary actor side |
| `Secondary-side Event Records` | Event records in which the selected actor appears on the dashboard's secondary actor side |

If shorter labels are needed in the UI, they must be paired with a visible explanatory note.

Terms to retire:

- `Offensive Events`
- `Defensive Events`
- `as attacker`
- `as targeted`

## Event Category Terms

| Preferred label | Meaning | Guidance |
| --- | --- | --- |
| `Ground-based attack` | Dashboard recode for battles, shelling, and other ground fighting | Dashboard category, not native ACLED field |
| `Air attack` | Dashboard recode for aircraft-delivered strikes | Dashboard category |
| `Drone attack` | Dashboard recode for unmanned aerial strikes | Dashboard category derived partly from notes and actor context |
| `Massacres` | Dashboard recode for civilian-targeting events with 5+ reported fatalities | Not a native ACLED field; always note this in methodology/trust copy |
| `Violence against civilians` | Dashboard category based on ACLED event type | Can overlap conceptually with other harm not captured in this category |
| `Protests` | Reported demonstration events in ACLED | Overview includes these; Actor Analysis does not |
| `Others` | Broad residual category for remaining activity not shown in the named dashboard groups | In the current dashboard export this is dominated by strategic developments such as changes to group activity, disrupted weapons use, base establishment, agreements, and non-violent transfers of territory |

## Standard Trust Copy

### Short page-level note

Use when space is limited:

`ACLED records reported events, not every event that occurred. Fatalities are reported estimates and may be revised.`

### Standard map caveat

Use near choropleths or in supporting methodology:

`Township shading shows the distribution of reported events or reported fatality estimates in the selected scope. It does not show territorial control, complete event coverage, or the full on-the-ground intensity of conflict.`

### Standard actor-side caveat

Use on Actor Analysis and in methodology:

`Actor-side metrics in this dashboard are analytical groupings derived from participant-side coding. They should not be read as definitive proof of who initiated violence, who was targeted first, or who caused or suffered all fatalities in an event.`

### Standard recode caveat

Use where `Massacres` or other local categories appear:

`Some categories are dashboard analytical recodes rather than native ACLED fields. In particular, Massacres refers here to civilian-targeting events with 5 or more reported fatalities.`

## Terms To Avoid

- `Total Conflicts`
- `Exact fatalities`
- `Deaths caused by [actor]`
- `Aggressor` unless separately coded and documented
- `Victim side` unless separately coded and documented
- `Territorial control` when only event occurrence data are shown

## Source Basis

This glossary is based on:

- ACLED codebook guidance on reported fatalities
- ACLED fatality methodology FAQ
- ACLED FAQ stating that fatalities are not generally attributable to specific groups
- ACLED FAQ stating that `Actor1` and `Actor2` do not imply aggressor status
- the current dashboard pipeline and actor-level dataset construction

Key references:

- https://acleddata.com/knowledge-base/codebook/
- https://acleddata.com/knowledge-base/faqs-acled-fatality-methodology/
- https://acleddata.com/faq/it-possible-identify-number-people-killed-specific-group-or-total-number-civilians-killed
- https://acleddata.com/faq/can-it-be-assumed-actor-1-aggressor
- `pipeline/pipeline.py`
