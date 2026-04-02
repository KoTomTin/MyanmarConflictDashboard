# Methodology

## Data Scope

The dashboard covers Myanmar conflict events from `2021-02-01` onward using ACLED as the base source. The current processed dataset in this repository extends through `2026-03-27`.

## Pipeline Overview

The pipeline lives in `pipeline/pipeline.py` and supports two modes:

- Full rebuild: process the historical Excel source, then top up from the ACLED API
- Incremental refresh: load the existing Parquet dataset and fetch a rolling overlap from the latest stored event date

The incremental update path now uses timestamp-based syncing, which is the method ACLED documents for catching newly added rows, edits to older rows, and deletions.

## Output Files

The pipeline writes these canonical outputs to `data/processed/`:

- `acled_cleaned.parquet`
- `acled_actor_level.parquet`
- `acled_actor_ally_pairs.parquet`
- `monthly_township.parquet`
- `last_updated.txt`

These Parquet files are the application source of truth.

The pipeline also maintains `acled_sync_state.json`, which stores the last sync cursor used for timestamp-based incremental refreshes.

## Cleaning And Recoding

The pipeline applies the following main transformations:

- parses and validates event dates
- drops future-dated rows
- normalizes `admin1` names
- matches township P-codes
- recodes `key_event`
- recodes `detailed_event`
- standardizes actor labels and actor-type groupings
- normalizes civilian-targeting flags
- coerces fatalities to integers
- recodes civilian-targeting events with `fatalities >= 5` as `Massacres`

## Key Event Logic

The dashboard groups raw ACLED event data into higher-level analytical categories including:

- Ground-based attack
- Air attack
- Drone attack
- Massacres
- Violence against civilians
- Protests
- Arrests
- Looting/property destruction
- Displacement
- Others

The air versus drone split is based on the `Air/drone strike` subtype plus contextual checks in the notes and actor classification.

## Geographic Matching

Township P-codes are matched using a district-plus-township lookup. The pipeline prefers the external lookup workbook when present and falls back to rebuilding the lookup from the existing cleaned Parquet dataset when necessary.

Some city-level rows require hard-coded fallback handling for places such as Mandalay, Nay Pyi Taw, and Yangon when ACLED does not provide a useful township field.

## Actor-Level Datasets

`acled_actor_level.parquet` is derived from armed conflict event categories, including `Massacres`. Each event is expanded into actor-role rows so the dashboard can analyze:

- who appeared in the event
- whether they were on the offending or defending side
- which actors appeared alongside them on the same side

`acled_actor_ally_pairs.parquet` is then built by exploding same-side actor relationships into pair rows.

## Pre-Aggregation For Performance

The dashboard does not build all map inputs from the full event-level dataset on every interaction. Instead, the pipeline pre-aggregates township-month metrics into `monthly_township.parquet`, which the Overview page uses for much faster choropleth and chart updates.

## Map Rendering Strategy

The animated choropleths use Plotly native animation frames bundled into the figure, rather than per-frame server callbacks. This keeps quarter-by-quarter playback responsive in the browser.

To reduce the size of the initial map callback, the app prefers a simplified web geometry file, `data/shapes/boundaries_web.geojson`, instead of the full raw township boundary file when that optimized asset is available.

## Limits And Biases

- ACLED reflects reported events, not every event that happened.
- Fatality values can be revised over time.
- Local actor naming can vary across reports and may not map perfectly onto standardized labels.
- Drone detection is heuristic and depends partly on event notes.
