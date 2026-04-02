# Changelog

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
