# Myanmar Conflict Dashboard

The Myanmar Conflict Dashboard is a Dash/Plotly application for exploring township-level conflict patterns in Myanmar using ACLED data. The runtime app is Dash-only; there is no Streamlit app in this repository.

## Current Snapshot

- Framework: Dash + Plotly + Dash Bootstrap Components
- Active runtime routes: `Overview`, `Actor Analysis`, `About`
- Data coverage in the current processed dataset: `2021-02-01` to `2026-03-27`
- Last recorded pipeline refresh: `2026-04-02`
- The UI now distinguishes `Data through` (latest event date) from `Synced` (pipeline refresh date)
- Deployment entrypoints: `app.py` and `Procfile`

## Repository Layout

- `app.py`: Dash entrypoint and navigation shell
- `pages/`: routed dashboard pages
- `components/`: shared loaders, color scales, and geo helpers
- `data/processed/`: canonical Parquet outputs used by the app
- `data/shapes/`: township boundary GeoJSON
- `pipeline/`: data update and transformation pipeline
- `docs/`: living project documentation and source markdown for the About page
- `research/`: paper/report artifacts and screenshots; not required to run the dashboard

## Run Locally

1. Create or activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the app with `python app.py`.
4. Open `http://localhost:8050`.

## Refresh Data

- Full rebuild: `python pipeline/pipeline.py`
- Incremental refresh: `python pipeline/pipeline.py --update-only`
- Optional CSV exports for inspection: add `--export-csv`

The pipeline writes:

- `data/processed/acled_cleaned.parquet`
- `data/processed/acled_actor_level.parquet`
- `data/processed/acled_actor_ally_pairs.parquet`
- `data/processed/monthly_township.parquet`
- `data/processed/last_updated.txt`

For runtime map performance, the app prefers `data/shapes/boundaries_web.geojson`
when present. You can rebuild it with `python pipeline/build_web_geojson.py`.

## Documentation Map

- [Architecture](docs/ARCHITECTURE.md)
- [Project Status](docs/PROJECT_STATUS.md)
- [Maintenance](docs/MAINTENANCE.md)
- [About Content](docs/ABOUT.md)
- [Methodology](docs/METHODOLOGY.md)
- [Changelog](CHANGELOG.md)

## Workspace Rules

- Keep runtime code and data at the repo root.
- Keep research, paper, and report artifacts under `research/`.
- Keep project facts in markdown files, not only in chat history.
- When routes, data shape, or workflow change, update the docs in `docs/` and `CHANGELOG.md` in the same change.
