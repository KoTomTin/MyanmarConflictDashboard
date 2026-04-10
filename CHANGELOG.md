# Changelog

## 2026-04-10

### Performance
- Enabled `Flask-Compress` (brotli + gzip) on the Flask server. Plotly figure JSON, Dash bundles, CSS, and HTML are now compressed in transit (HTML payload drops ~70% on first load).
- Added long-lived `Cache-Control` headers: `_dash-component-suites/*` cached `immutable` for one year, `assets/*` cached for one day. Repeat visits no longer re-download Plotly bundles.
- Default `app.run(debug=False)` so the dev reloader and unminified bundles do not slow down local sessions. Set `MCD_DEBUG=1` to opt back into the reloader.
- New `requirements.txt` entries: `Flask-Compress==1.24`, `brotli==1.2.0`.
- Overview page filtering refactor: `_filter_overview_events` now returns a boolean-mask view instead of `df.copy()`, and the filtered frame is computed once per callback and shared across the map, KPIs, trend, and ranking builders. Previously each builder re-ran the same filter on a 90k-row frame.

### Accessibility
- Added a keyboard skip-link (`Skip to main content`) in `app.py` and a global `:focus-visible` outline so the entire app is keyboard-navigable.
- Replaced clickable `<div>` map view-mode toggles in Overview and Actor Analysis with real `<button>` elements carrying `aria-label`, `aria-pressed`, and a `role="group"` container.
- Bumped `--text-muted` from `#6b7c89` to `#54677a` directly in `:root` so muted body text passes WCAG AA on the cream background.
- Honored `prefers-reduced-motion`: shimmer and transition animations are disabled for users who request reduced motion.

### Visual / charts
- New `/healthz` endpoint for uptime checks.
- Township and state name now appear together in every choropleth hover (Overview static + animated, Actor static + animated).
- Centralized alert chip colors as CSS custom properties (`--alert-no-change-*`, `--alert-activity-*`, `--alert-fatality-*`, `--alert-both-*`) and mirrored them into `components/colors.py` as `ALERT_CATEGORY_COLORS`. The Plotly side and CSS side now share one source of truth.
- Bumped the zero stop in `SEQUENTIAL_BLUES_ZERO_GREY` and `SEQUENTIAL_REDS_ZERO_GREY` from `#d9d9d9` to `#dde3ea`. Empty townships now read as "measured zero" rather than "missing data" against the cream page background.
- Standardized stray Plotly text greys (`#7a8895`, `#708192`, `#5b6d80`, `#6b7280`, `#6b7c89`) to `#5a6c7e` across Overview, Actor, and Alerts so muted chart copy is consistent.
- Fixed milestone label legibility on trend charts: font size 7.4 → 10, more headroom, padded label boxes.
- Fixed Actor Analysis trend overlap: legend moved from `y=1.15` (which collided with the milestone labels) to `y=-0.28` below the x-axis, chart height bumped from 280 to 310 px.
- Fixed Township Alerts Sankey label alignment. Labels are now placed at flow-weighted node centers (computed from `actor_flow` and `event_flow` totals) using `arrangement="fixed"` with explicit `node.x` and `node.y`, and rendered as paper-coordinate annotations in left/right gutters so they never collide with the colored ribbons.
- Added a co-appearance disclaimer under the Actor Analysis "Associated Actors" card so readers do not over-interpret co-occurrence as a formal alliance.

### Repo hygiene
- Deleted a stray 46 MB `acled_cleaned.csv` left in the repo root. The canonical dataset stays in `data/processed/acled_cleaned.parquet`.
- Added `components/plot_theme.py` as a future single source of truth for Plotly visual constants. Pages still use local copies; the migration is incremental.

## 2026-04-02

- Reorganized the workspace so runtime code stays at the repo root and research artifacts live under `research/`.
- Removed the duplicate `MyanmarConflictDashboard-main/` snapshot and generated cache/junk files from the working tree.
- Replaced the stale `readme.txt` with a canonical `README.md`.
- Moved About and Methodology content into markdown files under `docs/`.
- Updated the About page to load its content from `docs/ABOUT.md`.
- Removed the unused runtime `pages/4_methodology.py` page and its old text source.
- Added living project documentation so key operational facts do not live only in chat history.
- Switched ACLED refreshes toward timestamp-based incremental syncing and added sync-state tracking.
- Increased the update workflow cadence to every 6 hours so published ACLED changes reach the repo sooner.
- Reduced the Overview page's initial load weight by hydrating heavy graphs after the page mounts instead of embedding them in the first layout payload.
- Clarified the UI header so it shows both the latest event date and the last sync date, avoiding the old misleading `Last Updated` label.
- Added a web-optimized township boundary GeoJSON and updated the loader to prefer it, cutting the default Overview map callback payload from roughly 11.5 MB to roughly 1.2 MB locally.
