#!/bin/bash
# ==========================================================
# check-and-deploy.sh — polls origin/main for new commits and
# runs redeploy.sh when one is found. Designed for root cron.
# ==========================================================
#
# Cron entry (as root):
#   */2 * * * * /opt/apps/MyanmarConflictDashboard/scripts/check-and-deploy.sh
#
# Logs to /var/log/dashboard-deploy.log
# ==========================================================

set -euo pipefail

APP_USER="dashapp"
PROJECT_DIR="/opt/apps/MyanmarConflictDashboard"
BRANCH="main"
LOG_FILE="/var/log/dashboard-deploy.log"
LOCK_FILE="/var/lock/dashboard-deploy.lock"
REDEPLOY_SCRIPT="${PROJECT_DIR}/scripts/redeploy.sh"

# Prevent overlapping runs (a deploy can take several minutes; without this,
# a slow deploy + 2-min cron interval would queue up duplicate runs).
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] Skipped: previous deploy still running" >> "$LOG_FILE"
  exit 0
fi

cd "$PROJECT_DIR"

# Fetch as the app user so .git ownership stays consistent.
sudo -u "$APP_USER" git fetch --quiet origin "$BRANCH"

LOCAL_SHA=$(sudo -u "$APP_USER" git rev-parse "$BRANCH")
REMOTE_SHA=$(sudo -u "$APP_USER" git rev-parse "origin/$BRANCH")

if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
  exit 0
fi

# Data-only commits (the daily ACLED auto-update) don't need a restart:
# the app's loaders re-read the parquets when their mtime changes, so a
# git update alone is a zero-downtime deploy. Anything outside
# data/processed/ still goes through the full redeploy.
# Note: capture-then-test (not `grep -qv`) — BSD and GNU grep disagree on -q
# combined with -v, and early-exit -q can interact with pipefail; this form
# behaves identically everywhere. `|| true` covers grep's exit 1 when every
# changed path is under data/processed/.
NON_DATA_CHANGES=$(sudo -u "$APP_USER" git diff --name-only "$LOCAL_SHA" "$REMOTE_SHA" \
                     | grep -v "^data/processed/" || true)
if [ -n "$NON_DATA_CHANGES" ]; then
  DEPLOY_MODE="full"
else
  DEPLOY_MODE="data-only"
fi

{
  echo ""
  echo "================================================="
  echo "[$(date -Iseconds)] New commit on $BRANCH ($DEPLOY_MODE)"
  echo "  local : $LOCAL_SHA"
  echo "  remote: $REMOTE_SHA"
  echo "================================================="
  if [ "$DEPLOY_MODE" = "data-only" ]; then
    sudo -u "$APP_USER" git reset --hard "origin/$BRANCH"
    sleep 2
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/healthz || echo "000")
    echo "[$(date -Iseconds)] data-only update applied, healthz: $HTTP_STATUS"
    if [ "$HTTP_STATUS" != "200" ]; then
      echo "[$(date -Iseconds)] healthz not OK after data update — running full redeploy"
      bash "$REDEPLOY_SCRIPT" 2>&1
    fi
  else
    bash "$REDEPLOY_SCRIPT" 2>&1
  fi
  echo "[$(date -Iseconds)] check-and-deploy run finished"
} >> "$LOG_FILE"
