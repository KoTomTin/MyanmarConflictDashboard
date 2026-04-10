# Project Status

Last reviewed: `2026-04-10`

## Current Facts

- Runtime framework: Dash
- Deployment entrypoint: `app.py` (debug off by default; `MCD_DEBUG=1` re-enables the dev reloader)
- Public runtime pages: `Overview`, `Actor Analysis`, `Township Alerts`, `About`
- Operational endpoint: `/healthz` (returns `200 ok`) for uptime monitors
- Server-side compression: `Flask-Compress` (brotli + gzip) on all JSON/HTML/CSS/JS/SVG
- Static asset cache: `_dash-component-suites/*` `immutable` for one year, `assets/*` for one day
- Current processed data coverage: `2021-02-01` to `2026-03-27`
- Last recorded pipeline update: `2026-04-02`
- Current processed dataset sizes:
  - `acled_cleaned.parquet`: 91,581 rows
  - `acled_actor_level.parquet`: 125,706 rows
  - `acled_actor_ally_pairs.parquet`: 182,738 rows
  - `monthly_township.parquet`: 29,796 rows

## Current Workspace Policy

- Runtime code stays at the repository root.
- Documentation lives under `docs/`.
- Research and report artifacts live under `research/`.
- Key project facts must be written into markdown files and updated with the codebase.
- Current redesign work is local-only until explicitly approved for push/deployment.

## Known Intentional Choices

- The About page is markdown-backed through `docs/ABOUT.md`.
- Methodology is maintained as documentation in `docs/METHODOLOGY.md` rather than a live app route.
- Research PDFs and interview notes are treated as local artifacts, not core runtime files.
- `dbc.themes.FLATLY` is kept even though it adds CSS payload — page templates reference Bootstrap utility classes (`d-flex`, `me-`, `mt-`, etc.) in 100+ places. Removing it is a behavior-change refactor.
- Alert chip colors live in two mirrored locations on purpose: `--alert-*` CSS variables in `assets/style.css` (for HTML chips) and `ALERT_CATEGORY_COLORS` in `components/colors.py` (for Plotly choropleth fills). The two must be kept in sync.

## Update Triggers

Update this file when any of the following change:

- runtime routes or navigation
- data coverage or last refresh date
- canonical dataset names or shapes
- deployment entrypoints
- workspace layout or documentation policy
