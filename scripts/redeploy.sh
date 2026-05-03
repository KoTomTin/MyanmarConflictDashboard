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
RESET_VENV=true         # set false to keep existing venv
HEALTHCHECK_URL="http://127.0.0.1:8000"
# ===========================================

echo "========================================="
echo "Starting redeploy: $(date)"
echo "Project: $PROJECT_DIR"
echo "========================================="

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory does not exist."
  exit 1
fi

echo "Stopping service..."
systemctl stop $SERVICE || true

if [ "$FORCE_RECLONE" = true ]; then
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
  "
fi

echo "Installing dependencies..."
sudo -u $APP_USER bash -c "
cd $PROJECT_DIR
source venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install dash gunicorn
fi
"

echo "Starting service..."
systemctl start $SERVICE

echo "Waiting for app to boot..."
sleep 3

echo "Checking service status..."
systemctl --no-pager status $SERVICE || true

echo "Running health check..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $HEALTHCHECK_URL || echo "000")

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