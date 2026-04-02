# Architecture

## Runtime Overview

The application is a multi-page Dash app served from `app.py`. It uses a custom sidebar and mobile top navigation rather than Dash Pages auto-discovery. The current runtime routes are:

- `/`: Overview
- `/actor`: Actor Analysis
- `/about`: About

## Page Responsibilities

- `pages/1_overview.py`: national and regional filters, choropleth map, KPI cards, trend chart, and conflict-type bar chart
- `pages/2_actor.py`: actor selector, actor footprint map, associated actor chart, offensive/defensive trend, and actor KPIs
- `pages/3_about.py`: markdown-backed informational page sourced from `docs/ABOUT.md`

## Shared Components

- `components/loaders.py`: cached Parquet and GeoJSON loaders
- `components/map_utils.py`: geo bounds, filtering, and choropleth framing helpers
- `components/colors.py`: centralized palettes and color scales

## Data Flow

1. `pipeline/pipeline.py` builds or refreshes processed datasets.
2. The pipeline writes canonical Parquet files to `data/processed/`.
3. The Dash app loads those Parquet files through `components/loaders.py`.
4. Page callbacks filter and aggregate the loaded data into Plotly figures.

## Canonical Data Outputs

- `acled_cleaned.parquet`: event-level table
- `acled_actor_level.parquet`: one row per actor-role per event
- `acled_actor_ally_pairs.parquet`: actor-to-ally event pairs
- `monthly_township.parquet`: pre-aggregated township-month metrics for fast charts and maps
- `last_updated.txt`: pipeline refresh date

## Non-Runtime Material

- `docs/`: living project and operational documentation
- `research/`: paper draft, screenshots, notes, and other report artifacts

Files in `research/` should not be treated as application runtime dependencies.

## Deliberate Constraints

- The repo is Dash-only. There is no Streamlit runtime to maintain.
- The methodology currently lives as project documentation in `docs/METHODOLOGY.md`, not as a routed app page.
- Duplicate full-project snapshots should not be kept in the workspace root.
