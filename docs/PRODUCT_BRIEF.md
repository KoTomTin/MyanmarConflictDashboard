# Product Brief

Last updated: 2026-04-03

## Product Name

Myanmar Conflict Dashboard

## Product Type

A public-facing analytical web dashboard built with Dash and Plotly for exploring reported conflict and protest patterns in Myanmar at township level using ACLED-based data and local recoding.

## Product Posture

This product should behave primarily as a public explainer with analytical depth, not as an expert-only research workbench.

That means:

- first-time users should be able to understand the main message quickly
- explanation and interpretation support should be visible in the interface
- analytical depth should remain available, but it should not dominate first impression

## Product Purpose

The dashboard exists to help users quickly understand:

- where conflict incidents are concentrated
- how conflict patterns change over time
- how conflict differs by incident type
- how one actor's geographic footprint and engagement pattern compare across Myanmar

It should help users move from raw event counts to a more structured, responsible overview of conflict patterns without requiring them to work directly with raw ACLED exports.

## Core Product Promise

This dashboard should make Myanmar conflict data:

- easier to navigate
- faster to interpret
- more transparent about limits
- more useful for descriptive analysis

It is not meant to replace deeper field reporting, formal academic analysis, or full event-level investigation.

## Primary Audience

The redesign will optimize first for:

- interested public users seeking a clear explanation of conflict patterns in Myanmar
- journalists, students, and educators who need a credible descriptive overview
- policy, humanitarian, and civil-society users who need fast interpretation without reading raw data exports

## Secondary Audience

The dashboard should remain understandable for:

- researchers and analysts who want a quick descriptive starting point
- donors and partner organizations
- teaching and presentation use

## Audience Assumptions

We will design for users who:

- care about Myanmar specifically, not a generic global conflict dashboard
- may know the broad political context, but may not know ACLED methodology or conflict data practice in detail
- want trustworthy descriptive patterns quickly
- may use the dashboard on laptop first, but mobile access still matters

We will not assume that users:

- understand all event-type definitions automatically
- know the difference between reported events and reported fatalities
- know the limitations of township-level choropleths
- can infer methodology from the charts alone

## Main Use Cases

### Use Case 1. National overview

A user wants a fast answer to:

- where conflict is concentrated in Myanmar
- whether the selected period is calmer or more intense than earlier periods
- which incident types dominate the selected view

### Use Case 2. Geographic pattern exploration

A user wants to filter by date, region, or incident type and see how the geographic footprint changes.

### Use Case 3. Actor-focused analysis

A user wants to inspect one actor and answer:

- where this actor appears most often
- whether this actor appears more often on the dashboard's primary side or secondary side
- which actors most often appear alongside this actor
- how this actor's engagement changes over time

### Use Case 4. Responsible public reference

A user wants to cite or show the dashboard in a presentation and needs:

- clear dates
- visible data source and methodology cues
- language that does not overclaim precision

## Page Jobs

### Overview

Primary job:
Give a fast, credible national picture of conflict geography and trend for a selected scope.

It should answer:

- Where are reported incidents concentrated?
- What does the selected period look like overall?
- Are reported events or fatalities higher or lower than expected?
- What kinds of incidents dominate the selected view?

It should not try to do:

- deep actor-network analysis
- event-level forensic investigation
- causal explanation of why conflict is happening

### Actor Analysis

Primary job:
Show the geographic footprint and engagement profile of one actor within armed conflict events.

It should answer:

- Where does this actor appear?
- Is this actor more often recorded on the dashboard's primary side or secondary side?
- Which actors are most associated with this actor in the same events?
- How does this actor's engagement shift over time?

It should not try to do:

- estimate actor-specific fatality totals from ACLED
- serve as a complete network analysis tool
- imply command relationships or alliances beyond co-appearance logic

### About

Primary job:
Explain what the dashboard is, what data it uses, who it is for, and how to interpret it responsibly.

It should answer:

- What is this dashboard?
- What data and methods sit behind it?
- What are the main caveats?
- How can someone contact the project?

## Product Boundaries

This dashboard is for descriptive analysis.

It is in scope to:

- summarize reported events
- compare time periods
- compare incident types
- explore actor involvement patterns
- provide downloadable charts and transparent source cues

It is out of scope to:

- provide real-time tactical intelligence
- estimate true conflict prevalence
- identify exact culpability for fatalities from ACLED alone
- support event-level case adjudication
- replace full methodology documentation

## Content Principles

### 1. Clarity over density

The dashboard should answer a few questions well rather than show every possible chart.

### 1a. Public explanation before expert exploration

The interface should explain what matters before it asks users to operate many controls or decode specialist terms.

### 2. Static interpretation first

The default view should help users interpret the selected period immediately. Animation is secondary.

### 3. Trust is part of the interface

Data freshness, source, and limits should be visible close to the analysis, not buried in a methodology page.

### 4. Maps need support

Maps should be paired with trend, ranking, and type-composition views so users do not overread the choropleth.

### 5. Method-aware language

Terms like `fatalities`, `reported`, `most active`, and actor-side role labels should match what the data can actually support.

### 6. Mobile is not an afterthought

The mobile layout should be designed as a deliberate small-screen experience, not only a collapsed desktop layout.

## Tone And Presentation

The dashboard should feel:

- serious
- trustworthy
- clear
- restrained
- analytically strong
- publicly legible

It should not feel:

- sensational
- militaristic
- overly decorative
- excessively technical for first-time users

## Key Product Risks

- Users may interpret mapped density as complete ground truth rather than reported event coverage.
- Users may overinterpret fatality numbers as precise or actor-attributable.
- Animation may feel impressive but communicate less clearly than static views.
- Heavy filters and too many panels may weaken the first impression.
- Performance problems may undermine trust even if the visuals are improved.

## Success Criteria

The redesign is successful if a first-time visitor can:

- state what each page is for within 10 seconds
- identify the time coverage and refresh status without hunting
- understand what the map colors represent
- use filters without confusion
- distinguish reported event counts from reported fatality estimates
- understand Actor Analysis without needing prior knowledge of the codebase

The redesign is also successful if the app:

- feels fast on initial load
- remains responsive under repeated filtering
- works cleanly on desktop and mobile
- uses consistent language across pages

## Non-Goals For This Phase

We are not trying to:

- expand the dashboard to many new routes
- add advanced predictive analytics
- build a full research portal or document repository
- introduce heavy custom front-end frameworks
- push unfinished design experiments to production

## Immediate Implications For Design Work

- Overview should be rebuilt first.
- The page should lead with a clear national picture and a public-facing explanation, not with control density.
- Actor Analysis should inherit the approved system rather than invent a separate visual language.
- Trust copy and metric terminology should be standardized before any major UI rebuild.
- Every new layout decision should be checked against this brief before implementation.

## Release Policy For Current Redesign

All redesign work in this phase is local-first.

- redesign iterations happen locally
- nothing should be pushed or deployed until explicitly approved
- production launch happens only after local review confirms the product is satisfactory
