#!/bin/bash
# ==========================================================
# vps-update-data.sh — daily ACLED fetch from the VPS itself.
#
# Why: ACLED's Imunify360 bot-protection intermittently blocks GitHub's
# shared runner IPs (403 "IPs used for automation should be whitelisted").
# The VPS has one static IP, which is normal-looking traffic and can be
# whitelisted by ACLED if ever needed. GitHub Actions remains as a
# freshness monitor and manual fallback.
#
# Cron entry (as root):
#   30 0 * * * /opt/apps/MyanmarConflictDashboard/scripts/vps-update-data.sh
# (server is UTC; 00:30 UTC = 07:00 Yangon)
#
# Requires (installed by scripts/provision-vps-data-update.sh):
#   - /opt/apps/MyanmarConflictDashboard/.env  (ACLED credentials, dashapp:600)
#   - dashapp deploy key registered on GitHub with write access
#   - git push remote set to git@github.com:KoTomTin/MyanmarConflictDashboard.git
# ==========================================================
set -euo pipefail

APP_USER="dashapp"
PROJECT_DIR="/opt/apps/MyanmarConflictDashboard"
LOG_FILE="/var/log/dashboard-data-update.log"
# Shared with check-and-deploy.sh: serializes against a concurrent redeploy,
# which does a `git reset --hard` that would clobber freshly written parquets.
DEPLOY_LOCK="/var/lock/dashboard-deploy.lock"

exec 9>"$DEPLOY_LOCK"
if ! flock -w 300 9; then
  echo "[$(date -Iseconds)] data update skipped: could not acquire deploy lock" >> "$LOG_FILE"
  exit 1
fi

{
  echo ""
  echo "================================================="
  echo "[$(date -Iseconds)] VPS data update starting"

  BEFORE_SHA=$(sudo -u "$APP_USER" git -C "$PROJECT_DIR" rev-parse HEAD)

  sudo -u "$APP_USER" bash -c "
    set -euo pipefail
    cd $PROJECT_DIR
    git fetch --quiet origin main
    git rebase --quiet origin/main
    venv/bin/python pipeline/pipeline.py --update-only
    git add data/processed/
    git diff --cached --quiet || git commit -q -m \"data: auto-update ACLED data \$(date -u +%Y-%m-%d) (vps)\"
    git pull --rebase --quiet origin main
    git push --quiet origin main
  "

  # If the rebase above pulled in commits that touch anything besides data,
  # the running app is now stale and check-and-deploy will never notice
  # (local == remote after our push) — trigger the full redeploy ourselves.
  NON_DATA=$(sudo -u "$APP_USER" git -C "$PROJECT_DIR" diff --name-only "$BEFORE_SHA" HEAD \
               | grep -v "^data/processed/" || true)
  if [ -n "$NON_DATA" ]; then
    echo "[$(date -Iseconds)] non-data changes arrived during update — running full redeploy"
    bash "$PROJECT_DIR/scripts/redeploy.sh"
  fi

  echo "[$(date -Iseconds)] VPS data update finished OK"
} >> "$LOG_FILE" 2>&1
