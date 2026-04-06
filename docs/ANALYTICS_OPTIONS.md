# Analytics Options

## Why This Note Exists

This note captures high-value analytical features the dashboard can add next, based on:

- the current processed datasets used by the app
- the anomaly prototype in `research/anomaly_month.ipynb`
- the dashboard's public-explainer product brief
- the project's methodology and metric constraints

The goal is to increase analytical value without overclaiming what the underlying data can support.

## Available Analytical Data

The current workspace already supports more analysis than the live dashboard currently shows.

### Event-level

`data/processed/acled_cleaned.parquet` supports:

- event date
- township code
- admin1 / admin2 / admin3
- fatalities
- dashboard recoded `key_event`
- dashboard recoded `detailed_event`
- actor-side fields and notes

Current `key_event` categories in the canonical Parquet:

- `Ground-based attack`
- `Air attack`
- `Drone attack`
- `Massacres`
- `Violence against civilians`
- `Protests`
- `Arrests`
- `Looting/property destruction`
- `Displacement`
- `Others`

### Township-month

`data/processed/monthly_township.parquet` supports fast monthly aggregation by:

- township
- region
- month
- key event
- event count
- fatality count

This is the best source for anomaly detection, momentum, and event-mix change features.

### Actor-level

`data/processed/acled_actor_level.parquet` supports:

- actor presence by event
- township-level actor footprint
- event date
- participant-side analytical grouping

`data/processed/acled_actor_ally_pairs.parquet` supports:

- same-side co-appearance networks
- associated-actor ranking
- recurring combinations

## What The Anomaly Notebook Already Does

The notebook in `research/anomaly_month.ipynb` already provides a strong prototype idea:

- monthly aggregation per township
- 2-month rolling totals
- township-specific local baseline
- township-specific long-term baseline
- robust MAD-based surge flags
- emergence rules when historical baseline is 0 or 1
- combined categories:
  - `normal`
  - `Event surge`
  - `Fatality surge`
  - `Both surges`

That is a good direction because it compares each township against its own history rather than against the whole country only.

## Important Fixes Before Productizing Anomalies

### 1. Use canonical dashboard data, not the old CSV

The notebook currently reads a standalone `acled_cleaned.csv` and even hard-codes an old working directory. That is not the dashboard's canonical runtime source anymore.

For production dashboard use, anomaly logic should read:

- `data/processed/acled_cleaned.parquet`, or
- preferably `data/processed/monthly_township.parquet`

### 2. Align the metric with the wording

The notebook's current event count logic aggregates **all events**, but the explanation text says `armed-conflict events`.

If the dashboard says:

- `increased armed conflict activity`

then the anomaly feature should filter to the relevant dashboard conflict categories first, for example:

- `Ground-based attack`
- `Air attack`
- `Drone attack`
- `Massacres`
- `Violence against civilians`

If the dashboard instead wants to detect unusual change across the full event spectrum, the wording should say:

- `reported event activity`

not `armed conflict activity`.

### 3. Keep the explanation visible

For a public-facing dashboard, anomaly outputs should not appear as a black-box score. Each flagged township should show:

- current value
- recent baseline
- long-term baseline
- whether the unusual change is in events, fatalities, or both
- whether it is an emergence from near-zero activity or a surge above a prior pattern

## Recommended Anomaly Feature

The next analytical feature I would build is:

### Township Alert Map

Flag townships with unusual recent increases relative to their own history.

Suggested logic:

- use 2-month rolling totals
- compare against:
  - recent local baseline
  - longer-term township baseline
- classify:
  - `No unusual change`
  - `Event surge`
  - `Fatality surge`
  - `Both surges`
  - optional: `Emerging activity`

Suggested presentation:

- map layer with anomaly category coloring
- ranked table of flagged townships
- one-line explanation per township
- click-through sparkline for the selected township

This is the highest-value next analytical module because it adds interpretation, not just another descriptive count.

### How The Current Alerts Page Works

The local `Township Alerts` page now uses this logic:

- it works from event-level `acled_cleaned.parquet`, not the monthly township table
- the current alert window is the latest available **30-day** window
- that window ends with the latest event date represented in the processed data, not today's calendar date
- the page is meant to support transparent public alerting, not immediate-response operations

Why 30 days:

- it is more responsive than the earlier 2-month version
- it still smooths very short daily spikes
- it fits the public-facing goal better than a heavily lagged multi-month window

Definitions used in the page:

- `Recent baseline`: the median of the previous **6** 30-day windows for that township
- `Long-term baseline`: the median of **all** earlier 30-day windows for that township
- `Recent threshold`: recent baseline plus `2.5 x MAD`
- `Long-term threshold`: long-term baseline plus `2.5 x MAD`
- when a baseline is `0` or `1`, the page uses emergence thresholds from the historical 95th percentile instead of the MAD rule

Trend chart line meanings:

- solid lines: the rolling 30-day totals
- dashed lines: the recent baseline
- dotted lines: the long-term baseline

This is still an analytical design choice rather than a law of nature, but it is a better fit for a public alerting page than the earlier monthly 2-month prototype.

## Other High-Value Analytics From Current Data

### 1. Event Mix Shift

Question:

- which townships are not just busier, but behaving differently from their own usual event mix?

Use:

- monthly township data
- event-type shares instead of only totals

Example value:

- a township may not have a huge event surge, but may show a sharp shift from protests to armed attacks or from clashes to air/drone strikes

Good visuals:

- slope chart of previous vs current event mix
- diverging bar chart of share change by event type
- small stacked bars for baseline vs current period

### 2. Sustained Hotspot Score

Question:

- which townships are persistently high-activity, not just spiking this month?

Use:

- count how many months a township sits above a chosen percentile or above its own baseline

Good visuals:

- ranked bar chart
- map of `persistent hotspot` vs `temporary surge`

This is useful because it distinguishes long-running conflict centres from short-term flare-ups.

### 3. Volatility Score

Question:

- which townships are unstable or erratic over time?

Use:

- rolling coefficient of variation
- month-to-month absolute change
- count of reversals or spikes

Good visuals:

- scatter plot:
  - x = average activity
  - y = volatility
- or a ranked list with sparklines

This helps separate steady high-burden areas from suddenly changing areas.

### 4. Severity Shift

Question:

- where are fatality estimates rising faster than event counts?

Use:

- fatalities per event
- current vs historical fatality intensity

Good visuals:

- quadrant scatter:
  - x = change in events
  - y = change in fatalities per event
- bubble size = current total fatalities

This adds analytical value without implying actor-specific fatality attribution.

### 5. Townships Driving The National Change

Question:

- which townships explain most of the national increase or decrease in the selected period?

Use:

- difference between current period and comparison period
- contribution share by township

Good visuals:

- waterfall chart
- ranked contribution bars
- map plus top-contributors table

This is more interpretable than only showing national totals.

### 6. Regional Burden Decomposition

Question:

- how much of the selected national total comes from each region?

Use:

- region share of events
- region share of fatalities
- share of anomaly-flagged townships

Good visuals:

- stacked share bars
- waffle or treemap if kept simple

This is useful for public explainers because it helps users move from national picture to subnational burden.

### 7. Actor Concentration In Flagged Townships

Question:

- when a township is flagged, which actors appear there most often in armed conflict events?

Use:

- anomaly township list
- actor-level dataset filtered to the same time window and township

Good visuals:

- top actors bar chart for selected township
- small linked table beneath anomaly ranking

This would create a bridge between Overview and Actor Analysis.

Important caveat:

- use actor presence, not actor-attributed fatalities
- avoid `attacker` / `victim` language

### 8. Associated Actor Shifts

Question:

- for a selected actor, are the actors most commonly appearing alongside them changing over time?

Use:

- `acled_actor_ally_pairs.parquet`

Good visuals:

- before/after rank comparison
- top associated actors in two selected periods
- streamgraph or bump chart if kept readable

### 9. First Appearance / Resurgence

Question:

- where is conflict newly appearing after long dormancy?

Use:

- emergence logic from the anomaly notebook
- optionally require several months of prior near-zero activity

Good visuals:

- map of `newly active`, `re-emerging`, `persistently active`
- ranked table with last active month and current month totals

This is especially useful for a public explainer because it tells a geographic change story directly.

### 10. Civilian Harm Profile

Question:

- where is activity increasingly concentrated in `Violence against civilians` and `Massacres` rather than in other categories?

Use:

- current share of events in those two categories
- change from township baseline

Good visuals:

- two-axis comparison
- lollipop ranking
- map of civilian-harm share

This is strong analytically, but needs especially careful wording around reported fatalities and event coding.

## Best Visual Additions To Prioritize

If the goal is to add analytical value without overcomplicating the dashboard, I would prioritize:

1. anomaly alert map
2. anomaly ranking table with explanation
3. township sparkline on click
4. event mix shift view
5. sustained hotspot vs temporary surge classification

That set would already make the dashboard feel much more analytical.

## Suggested Product Sequence

### Phase 1

Add one anomaly module to Overview:

- anomaly map
- anomaly ranking table
- selected-township sparkline

### Phase 2

Add one structural change metric:

- event mix shift

### Phase 3

Add one persistence metric:

- sustained hotspot vs temporary surge

### Phase 4

Bridge Overview to Actor Analysis:

- top actors present in flagged townships

## What To Avoid

These ideas would look analytical but would not be method-safe enough for the current product:

- actor-specific fatality attribution
- predictive conflict forecasting without a separate modeling workflow
- opaque composite `risk scores` with no explanation
- territorial-control claims from the township choropleth
- causal language like `why violence increased` without supporting data

## Recommendation

The best next analytical feature is:

### Build a township anomaly module

But do it with three corrections:

1. read canonical processed data, not the old CSV
2. decide clearly whether the metric is `all reported events` or `armed conflict activity`
3. pair every flag with a visible explanation and baseline comparison

If we do that well, the dashboard will gain real analytical value without breaking the public-explainer purpose.
