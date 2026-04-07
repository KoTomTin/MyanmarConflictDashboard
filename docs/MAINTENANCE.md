# Maintenance

## Documentation Rule

Project facts should not live only in chat history. When the project changes, update the relevant markdown files in the same change.

## Minimum Docs To Review On Each Meaningful Change

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/PROJECT_STATUS.md`
- `CHANGELOG.md`

Also update these when their content changes:

- `docs/ABOUT.md`
- `docs/METHODOLOGY.md`
- `research/README.md`

## Workspace Rules

- Do not keep duplicate full-project copies in the root directory.
- Do not leave loose screenshots or report exports in the root directory.
- Keep runtime code separate from research/report material.
- Prefer markdown for persistent project knowledge.

## Data Refresh Workflow

1. Run `python pipeline/pipeline.py --update-only` for normal refreshes.
2. Confirm `data/processed/last_updated.txt` changed as expected.
3. Confirm `data/processed/last_checked.json` records the latest Yangon-time check.
4. If production is hosted separately from GitHub (for example Render or a self-managed Hetzner box), configure a redeploy trigger so data-only bot commits also refresh the live site.
5. For deploy hooks, set `RENDER_DEPLOY_HOOK_URL` or `DEPLOY_HOOK_URL`.
6. For Hetzner / self-managed SSH deploys, set:
   - `HETZNER_DEPLOY_HOST`
   - `HETZNER_DEPLOY_USER`
   - `HETZNER_DEPLOY_KEY`
   - `HETZNER_DEPLOY_COMMAND`
   - optional `HETZNER_DEPLOY_PORT`
   Recommended: keep a fixed redeploy script on the server and call that command over SSH, rather than downloading and executing a remote script on every run.
7. Confirm `data/processed/acled_sync_state.json` advanced as expected.
8. Spot-check record counts and latest event date.
9. Update `docs/PROJECT_STATUS.md` if the coverage window changed materially.
10. Add a short entry to `CHANGELOG.md` for meaningful operational changes.

## App Change Workflow

1. Make the code change.
2. Run a lightweight verification step such as `python -m py_compile ...` or a local app launch.
3. If map boundary geometry changed, rebuild `data/shapes/boundaries_web.geojson` with `python pipeline/build_web_geojson.py`.
4. Update docs if routes, workflows, or directory structure changed.
5. Keep `research/` references valid if files were moved.

## Cleanup Checklist

- Remove `__pycache__/` directories before finishing major cleanup work.
- Remove stray `.DS_Store` files outside protected tool directories.
- Keep `.env` and other secrets local.
- Keep generated PDFs and interview notes out of the core runtime footprint.
