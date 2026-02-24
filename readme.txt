Myanmar Conflict Dashboard (Dash/Plotly)

Overview
- Interactive dashboard for exploring township-level conflict patterns in Myanmar.
- Built with Dash (Plotly) and Dash Bootstrap Components.
- Entry point: app server in [app.py](app.py). Sidebar enforces a custom page order and uses Dash Pages.

Folder structure
- [app.py](app.py): Dash app, sidebar, layout container.
- [requirements.txt](requirements.txt): Python dependencies.
- [assets/](assets): Static assets (CSS). Main stylesheet: [assets/style.css](assets/style.css).
- [components/](components): Reusable UI and data modules:
  - [components/loaders.py](components/loaders.py): Data/GeoJSON loaders with caching.
  - [components/map_utils.py](components/map_utils.py): Map helpers like [`components.map_utils.apply_tight_geos`](components/map_utils.py), [`components.map_utils.ensure_full_geoindex`](components/map_utils.py).
  - [components/tiles.py](components/tiles.py): Regional tiles builder [`components.tiles.build_region_tiles`](components/tiles.py).
  - [components/actor_utils.py](components/actor_utils.py): Actor network + tactic tiles (`networkx`, Plotly) — e.g., [`components.actor_utils.build_network_figure`](components/actor_utils.py).
  - [components/colors.py](components/colors.py): Centralized color palettes.
  - [components/layout.py](components/layout.py), [components/ui.py](components/ui.py), [components/cards.py](components/cards.py), [components/figures.py](components/figures.py), [components/filters.py](components/filters.py), [components/utils_dates.py](components/utils_dates.py), [components/utils_format.py](components/utils_format.py).
- [pages/](pages): Dash Pages (routed views):
  - [pages/1_overview.py](pages/1_overview.py): Overview map, KPIs, weekly trends, top events.
  - [pages/2_actor.py](pages/2_actor.py): Actor interaction network and tactic tiles.
  - [pages/3_temporal.py](pages/3_temporal.py): Anomaly choropleth + regional tiles (forecast highlights).
  - [pages/4_clustering.py](pages/4_clustering.py): K-means clustering + PCA. Feature aggregation via [`pages.4_clustering.aggregate_features`](pages/4_clustering.py).
  - [pages/5_guide.py](pages/5_guide.py): User guide.
- [data/processed/](data/processed): Input CSVs
  - acled_cleaned.csv, anomaly_detection.csv, time_series.csv
- [data/shapes/](data/shapes): GeoJSON boundary file
  - boundaries.geojson

Prerequisites
- Python 3.10+ recommended.
- Install dependencies:
  - Windows/macOS/Linux (venv optional):
    - python -m venv .venv
    - source .venv/bin/activate  (Windows: .venv\Scripts\activate)
    - pip install -r requirements.txt

Data
- Expected files:
  - [data/processed/acled_cleaned.csv](data/processed/acled_cleaned.csv)
  - [data/processed/anomaly_detection.csv](data/processed/anomaly_detection.csv)
  - [data/processed/time_series.csv](data/processed/time_series.csv)
  - [data/shapes/boundaries.geojson](data/shapes/boundaries.geojson)
- Loaders validate required columns and normalize types. See [components/loaders.py](components/loaders.py).

Run
- Start the dev server:
  - python app.py
- Open http://localhost:8050 (hot reload is enabled in debug mode).
- Pages (sidebar order): Overview (/), Actor Analysis (/actor-network), Temporal (/temporal), Clustering (/clustering), Guide (/guide).

License and data sources
- ACLED event data and MIMU boundaries credited to original providers; see the Guide page [pages/5_guide.py](pages/5_guide.py).