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

{
  echo ""
  echo "================================================="
  echo "[$(date -Iseconds)] New commit on $BRANCH"
  echo "  local : $LOCAL_SHA"
  echo "  remote: $REMOTE_SHA"
  echo "================================================="
  bash "$REDEPLOY_SCRIPT" 2>&1
  echo "[$(date -Iseconds)] check-and-deploy run finished"
} >> "$LOG_FILE"
