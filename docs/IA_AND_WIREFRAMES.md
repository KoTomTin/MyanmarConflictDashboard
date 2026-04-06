# Information Architecture And Wireframes

Last updated: 2026-04-03

## Purpose

This document translates the product brief, metric glossary, and UI audit into low-fidelity page structure.

It is intentionally layout-first, not style-first.

The goal is to decide:

- what each page says first
- what content belongs above the fold
- what should be primary versus secondary
- how desktop and mobile should differ

This is a local-first planning artifact. It does not authorize any push or deployment.

## Design Direction

The dashboard is being redesigned as a **public explainer with analytical depth**.

That means:

- the first screen should explain before it asks users to interpret
- maps stay important, but they are not left to speak alone
- trust cues and limitations must be visible close to the analysis
- expert-level flexibility remains, but it should not define first impression

## Global IA Rules

### 1. Header layer

Each analytical page should begin with:

- page title
- one-sentence explanatory subtitle
- freshness status

This layer should answer:

- what page am I on?
- what is it for?
- how current is the data?

### 2. Trust layer

Each analytical page should surface a compact trust band near the top containing:

- source
- reported-events caveat
- fatality-estimate caveat when relevant

This should not be buried only at the bottom.

### 3. Filter layer

Filters should be available, but not feel like the main story.

Desktop:

- one compact filter strip or filter card below the header/trust layer

Mobile:

- collapsed filter drawer by default

### 4. Primary analysis layer

Each page should have one dominant analytical surface.

- Overview: national situation picture
- Actor Analysis: selected actor footprint and role profile

### 5. Support layer

Supporting charts should clarify the main view, not compete with it.

## Page-Level Architecture

## Overview

### Primary job

Explain what the selected Myanmar conflict picture looks like in geography, volume, and composition.

### Primary message

`Here is the national conflict picture for the selected period, where it is concentrated, and what kind of incidents dominate it.`

### What the page should do first

1. establish scope and freshness
2. explain what the map shows
3. give one or two headline numbers
4. then allow exploration

### Overview desktop wireframe

```text
+--------------------------------------------------------------------------------------+
| OVERVIEW                               Data through · Last checked with ACLED |
| Understand where reported conflict is concentrated and how it changes over time      |
+--------------------------------------------------------------------------------------+
| ACLED source · reported-events caveat · fatality-estimate caveat                     |
+--------------------------------------------------------------------------------------+
| Filters: Date | Region | Incident type | Apply | Reset                               |
+--------------------------------------------------------------------------------------+
| Scope chips: Date | Region | Type | View                                              |
+--------------------------------------------------------------------------------------+
| NATIONAL PICTURE                                                                     |
| Short explainer sentence:                                                            |
| "This map shows the distribution of reported events in the selected period."         |
|                                                                                      |
| +-------------------------------------------+ +------------------------------------+ |
| | MAP: Conflict Geography                   | | HEADLINES                          | |
| | subtitle + map-specific caveat            | | Reported Events                   | |
| |                                           | | Reported Fatality Estimate        | |
| | [static choropleth map]                   | | Most Active Township              | |
| |                                           | | small interpretation note         | |
| | [view switch: static / animated]          | +------------------------------------+ |
| +-------------------------------------------+ +------------------------------------+ |
|                                             | | Incident Types                     | |
|                                             | | short explanatory subtitle         | |
|                                             | | [horizontal bar chart]             | |
|                                             | +------------------------------------+ |
+--------------------------------------------------------------------------------------+
| EVENT TREND                                                                          |
| Use the timeline to see when reported events rose or fell in the selected scope      |
| [time-series chart with milestone annotations]                                       |
+--------------------------------------------------------------------------------------+
| FOOTER TRUST / SOURCE NOTE                                                           |
+--------------------------------------------------------------------------------------+
```

### Overview interpretation

- The map remains the hero, but the right rail becomes explicitly explanatory.
- Headline cards are no longer just KPI ornaments; they are part of interpretation.
- `Incident Types` moves closer to the map because it helps explain what the map is made of.
- `Event Trend` becomes a full-width explanatory layer beneath the national picture.

### Overview mobile wireframe

```text
+---------------------------------------------------------------+
| OVERVIEW                                      Data through... |
| Understand where reported conflict is concentrated            |
+---------------------------------------------------------------+
| ACLED source · caveat summary                                 |
+---------------------------------------------------------------+
| [Filters]                                                     |
+---------------------------------------------------------------+
| Scope chips                                                   |
+---------------------------------------------------------------+
| Reported Events                                               |
| Reported Fatality Estimate                                    |
| Most Active Township                                          |
+---------------------------------------------------------------+
| Conflict Geography                                            |
| short explainer + caveat                                      |
| [static choropleth map]                                       |
| [view switch: static / animated]                              |
+---------------------------------------------------------------+
| Incident Types                                                |
| [bar chart]                                                   |
+---------------------------------------------------------------+
| Event Trend                                                   |
| [line chart]                                                  |
+---------------------------------------------------------------+
| Footer trust note                                             |
+---------------------------------------------------------------+
```

### Why mobile differs

- Headline numbers come before the map because they are faster to read on a small screen.
- The map still matters, but it should not dominate before the user knows what they are seeing.
- Filter controls stay hidden until requested.

## Actor Analysis

### Primary job

Explain where a selected actor appears and how that actor's recorded participation profile is distributed across dashboard-coded event sides and associated actors.

### Primary message

`Here is where this actor appears, which event-side grouping they most often appear in, and which actors most often co-appear on the same side.`

### What the page should do first

1. identify the selected actor clearly
2. explain the actor-side interpretation caveat
3. show footprint and side profile
4. then show associated actors and trend

### Actor desktop wireframe

```text
+--------------------------------------------------------------------------------------+
| ACTOR ANALYSIS                         Data through · Last checked with ACLED |
| Understand where one actor appears and how its event-side profile changes            |
+--------------------------------------------------------------------------------------+
| Selected actor: Myanmar Military Regime                                              |
| actor-side caveat: dashboard analytical grouping, not native aggressor/victim field  |
+--------------------------------------------------------------------------------------+
| Filters: Date | Region | Actor | Apply | Reset                                       |
+--------------------------------------------------------------------------------------+
| Scope chips: Date | Region | Actor | View                                            |
+--------------------------------------------------------------------------------------+
| ACTOR FOOTPRINT                                                                      |
| Short explainer sentence                                                             |
|                                                                                      |
| +-------------------------------------------+ +------------------------------------+ |
| | MAP: Geographic Footprint                 | | ACTOR PROFILE                      | |
| | map-specific caveat                       | | Primary-side Event Records         | |
| |                                           | | Secondary-side Event Records       | |
| | [static choropleth map]                   | | Townships                          | |
| |                                           | | Most Active Township               | |
| | [view switch: static / animated]          | | small interpretation note          | |
| +-------------------------------------------+ +------------------------------------+ |
|                                             | | Associated Actors                  | |
|                                             | | co-appearance caveat               | |
|                                             | | [bar chart/table]                  | |
|                                             | +------------------------------------+ |
+--------------------------------------------------------------------------------------+
| MONTHLY ENGAGEMENT TREND                                                             |
| Primary-side vs Secondary-side event records over time                               |
| [time-series chart with milestone annotations]                                       |
+--------------------------------------------------------------------------------------+
| FOOTER TRUST / SOURCE NOTE                                                           |
+--------------------------------------------------------------------------------------+
```

### Actor mobile wireframe

```text
+---------------------------------------------------------------+
| ACTOR ANALYSIS                               Data through...  |
| Understand where one actor appears                           |
+---------------------------------------------------------------+
| Selected actor: Myanmar Military Regime                      |
| actor-side caveat                                            |
+---------------------------------------------------------------+
| [Filters]                                                    |
+---------------------------------------------------------------+
| Scope chips                                                  |
+---------------------------------------------------------------+
| Primary-side Event Records                                   |
| Secondary-side Event Records                                 |
| Townships                                                    |
| Most Active Township                                         |
+---------------------------------------------------------------+
| Geographic Footprint                                         |
| map explainer + caveat                                       |
| [static choropleth map]                                      |
| [view switch]                                                |
+---------------------------------------------------------------+
| Associated Actors                                            |
| co-appearance caveat                                         |
| [chart/table]                                                |
+---------------------------------------------------------------+
| Monthly Engagement Trend                                     |
| [line chart]                                                 |
+---------------------------------------------------------------+
| Footer trust note                                            |
+---------------------------------------------------------------+
```

### Why mobile differs

- The selected actor and caveat should appear before anything else.
- The side-profile summary becomes the quick entry point, not the map.
- Associated actors should stay above the trend because it answers a more immediate “who with?” question.

## About

### Primary job

Help a first-time visitor understand what the dashboard is and how to use it responsibly.

### About wireframe

```text
+--------------------------------------------------------------------------------------+
| ABOUT                                                                                |
| What this dashboard is, who it is for, and how to interpret it responsibly          |
+--------------------------------------------------------------------------------------+
| QUICK START                                                                          |
| - what the dashboard shows                                                           |
| - how to read the pages                                                              |
| - what the data can and cannot support                                               |
+--------------------------------------------------------------------------------------+
| DATA AND METHODS                                                                     |
| source, recoding, actor-side logic, map caveats                                      |
+--------------------------------------------------------------------------------------+
| KEY TERMS                                                                            |
| glossary table                                                                       |
+--------------------------------------------------------------------------------------+
| LIMITATIONS                                                                          |
| short, high-visibility caveats                                                       |
+--------------------------------------------------------------------------------------+
| CONTACT                                                                              |
+--------------------------------------------------------------------------------------+
```

## Content Moves Recommended By This IA

### Move closer to the top

- short trust copy
- map interpretation cue
- actor-side caveat
- “who this page is for” signal on About

### Move lower or visually demote

- heavy filter detail
- secondary mode controls
- long generic footers

### Merge or simplify

- combine freshness and trust into one more coherent top information layer
- reduce repeated explanatory phrases that currently appear as separate mini-bands

## Control Hierarchy Recommendations

### Primary controls

- date range
- actor selector on Actor Analysis
- region selector

### Secondary controls

- incident type selector on Overview
- static/animated switch
- events/fatality metric switch

The secondary controls should not dominate above-the-fold attention.

## Open Decisions For The Next Session

- Should `Reported Fatality Estimate` be a top-card label or a slightly shorter form in the final UI?
- Should `Incident Types` stay in the right rail on desktop or move below the map as a full-width section?
- Should Actor Analysis keep map and associated actors side-by-side on desktop, or should the associated-actor view move beneath the map?
- Should the bottom source note become a compact top trust band plus a lighter footer, instead of one strong footer disclaimer?

## Recommended Next Implementation Sequence

1. Approve or revise these wireframes.
2. Translate them into a small page-level layout spec.
3. Rebuild Overview first in local-only mode.
4. Review locally.
5. Rebuild Actor Analysis after Overview is approved.
