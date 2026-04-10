# Architecture

## Runtime Overview

The application is a multi-page Dash app served from `app.py`. It uses a custom sidebar and mobile top navigation rather than Dash Pages auto-discovery. The current runtime routes are:

- `/`: Overview
- `/actor`: Actor Analysis
- `/alerts`: Township Alerts
- `/about`: About

## Page Responsibilities

- `pages/1_overview.py`: national and regional filters, choropleth map, KPI cards, trend chart, and conflict-type bar chart
- `pages/2_actor.py`: actor selector, actor footprint map, associated actor chart, offensive/defensive trend, and actor KPIs
- `pages/4_alerts.py`: township anomaly prototype using recent versus historical township baselines for armed conflict activity and fatalities
- `pages/3_about.py`: markdown-backed informational page sourced from `docs/ABOUT.md`

## Shared Components

- `components/loaders.py`: cached Parquet and GeoJSON loaders
- `components/map_utils.py`: geo bounds, filtering, and choropleth framing helpers
- `components/colors.py`: centralized palettes and color scales (also exposes `ALERT_CATEGORY_COLORS`, the single source of truth shared with the `--alert-*` CSS variables in `assets/style.css`)
- `components/plot_theme.py`: shared Plotly font, color, and layout constants for the editorial cream-and-slate look. Pages are migrating to it incrementally

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

## Boundary Geometry

- `data/shapes/boundaries.geojson`: source township boundary file
- `data/shapes/boundaries_web.geojson`: web-optimized simplified boundary file used by the app when present
- `data/shapes/neighbor_borders.geojson`: curated neighboring-country border context derived from Natural Earth and clipped to Myanmar's immediate surroundings

The runtime loader prefers the simplified web file so choropleth callbacks do not have to ship the full raw geometry on every render.

## Non-Runtime Material

- `docs/`: living project and operational documentation
- `research/`: paper draft, screenshots, notes, and other report artifacts

Files in `research/` should not be treated as application runtime dependencies.

## Server-Level Behavior

- `Flask-Compress` (brotli + gzip) is wired in `app.py` and applies to HTML, JSON, JS, CSS, SVG, and plain text. This is where the perceived "fast" feel of repeated callbacks comes from — Plotly figure JSON compresses 70–85%.
- `_dash-component-suites/*` is served with `Cache-Control: public, max-age=31536000, immutable`. Dash bundles are content-hashed so this is safe.
- `/assets/*` is served with `Cache-Control: public, max-age=86400`.
- `/healthz` returns `200 ok` for uptime monitors.
- `app.run(debug=False)` by default. Set `MCD_DEBUG=1` to enable the dev reloader.

## Deliberate Constraints

- The repo is Dash-only. There is no Streamlit runtime to maintain.
- The methodology currently lives as project documentation in `docs/METHODOLOGY.md`, not as a routed app page.
- Duplicate full-project snapshots should not be kept in the workspace root.
- `dbc.themes.FLATLY` is intentionally retained even though it adds CSS weight: pages reference Bootstrap utility classes (`d-flex`, `me-`, `mt-`, etc.) in 100+ places. Removing it is a behavior-change refactor and should not be bundled with theming work.
