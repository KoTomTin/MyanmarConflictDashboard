# Project Status

Last reviewed: `2026-04-02`

## Current Facts

- Runtime framework: Dash
- Deployment entrypoint: `app.py`
- Public runtime pages: `Overview`, `Actor Analysis`, `About`
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

## Known Intentional Choices

- The About page is markdown-backed through `docs/ABOUT.md`.
- Methodology is maintained as documentation in `docs/METHODOLOGY.md` rather than a live app route.
- Research PDFs and interview notes are treated as local artifacts, not core runtime files.

## Update Triggers

Update this file when any of the following change:

- runtime routes or navigation
- data coverage or last refresh date
- canonical dataset names or shapes
- deployment entrypoints
- workspace layout or documentation policy
