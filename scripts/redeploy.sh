#!/bin/bash
# ==========================================================
# MyanmarConflictDashboard Production Redeploy Script
# Ubuntu 24.04
# Domain: myanmarconflictdashboard.com
# App user: dashapp
# Project path: /opt/apps/MyanmarConflictDashboard
# Service: myanmardashboard
# ==========================================================

set -euo pipefail

# ================= CONFIG =================
APP_USER="dashapp"
APP_BASE="/opt/apps"
PROJECT_NAME="MyanmarConflictDashboard"
PROJECT_DIR="${APP_BASE}/${PROJECT_NAME}"
SERVICE="myanmardashboard"
DEFAULT_BRANCH="main"
FORCE_RECLONE=false     # set true to delete repo and reclone
RESET_VENV=false        # set true to rebuild the venv from scratch
HEALTHCHECK_URL="http://127.0.0.1:8000/healthz"
# ===========================================

echo "========================================="
echo "Starting redeploy: $(date)"
echo "Project: $PROJECT_DIR"
echo "========================================="

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory does not exist."
  exit 1
fi

# NOTE: the service keeps running while we pull and install — gunicorn holds
# the old code in memory, so the only downtime is the restart at the end.
# The old flow stopped the service first and rebuilt the venv from scratch,
# which meant multiple minutes of 502 on every commit (including the daily
# data auto-commit).

if [ "$FORCE_RECLONE" = true ]; then
  echo "Stopping service for reclone..."
  systemctl stop $SERVICE || true
  echo "Force reclone enabled. Removing project..."
  rm -rf "$PROJECT_DIR"
  sudo -u $APP_USER git clone https://github.com/KoTomTin/MyanmarConflictDashboard.git "$PROJECT_DIR"
fi

echo "Detecting branch..."
BRANCH=$(sudo -u $APP_USER bash -c "cd $PROJECT_DIR && git rev-parse --abbrev-ref HEAD" || echo "$DEFAULT_BRANCH")

echo "Using branch: $BRANCH"

echo "Pulling latest code..."
sudo -u $APP_USER bash -c "
cd $PROJECT_DIR
git fetch --all
git reset --hard origin/$BRANCH
"

if [ "$RESET_VENV" = true ]; then
  echo "Resetting virtual environment..."
  rm -rf "$PROJECT_DIR/venv"
fi

if [ ! -d "$PROJECT_DIR/venv" ]; then
  echo "Creating virtual environment..."
  sudo -u $APP_USER bash -c "
  cd $PROJECT_DIR
  python3 -m venv venv
  venv/bin/pip install --upgrade pip
  "
fi

# With an existing venv this is a fast no-op when nothing changed in
# requirements.txt — pip resolves already-satisfied pins in seconds.
echo "Installing dependencies..."
sudo -u $APP_USER bash -c "
cd $PROJECT_DIR
source venv/bin/activate
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install dash gunicorn
fi
"

echo "Restarting service..."
systemctl restart $SERVICE

echo "Checking service status..."
systemctl --no-pager status $SERVICE || true

echo "Running health check (up to 30s)..."
HTTP_STATUS="000"
for _ in $(seq 1 15); do
  sleep 2
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $HEALTHCHECK_URL || echo "000")
  if [ "$HTTP_STATUS" = "200" ]; then
    break
  fi
done

if [ "$HTTP_STATUS" = "200" ]; then
  echo "Health check PASSED (HTTP 200)"
else
  echo "Health check FAILED (HTTP $HTTP_STATUS)"
  echo "Check logs:"
  echo "  journalctl -u $SERVICE -n 50"
  exit 1
fi

echo "========================================="
echo "Redeploy completed successfully."
echo "Time: $(date)"
echo "========================================="