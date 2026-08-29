#!/bin/bash
# ==========================================================
# provision-vps-data-update.sh — ONE-TIME setup for VPS-side data updates.
# Run this from the project directory on the machine that has:
#   - SSH root access to the VPS (key in ~/.ssh)
#   - the ACLED .env in the repo root
#   - GitHub credentials stored for git (used to register the deploy key)
#
#   bash scripts/provision-vps-data-update.sh
#
# What it does (idempotent, safe to re-run):
#   1. copies .env to the VPS (dashapp-owned, mode 600)
#   2. creates a dashapp SSH keypair and registers it on GitHub as a
#      WRITE deploy key scoped to this repository only
#   3. points the repo's push remote at SSH (fetch stays anonymous https)
#   4. installs the 00:30 UTC (07:00 Yangon) root cron entry
#   5. waits for the VPS to have scripts/vps-update-data.sh, runs it once,
#      and verifies a data commit reached GitHub
# ==========================================================
set -euo pipefail

VPS="root@46.225.233.211"
PROJECT_DIR="/opt/apps/MyanmarConflictDashboard"
REPO="KoTomTin/MyanmarConflictDashboard"
CRON_LINE="30 0 * * * $PROJECT_DIR/scripts/vps-update-data.sh"

cd "$(dirname "$0")/.."
[ -f .env ] || { echo "ERROR: .env not found in repo root"; exit 1; }

echo "[1/5] Copying ACLED credentials to the VPS..."
scp -q .env "$VPS:$PROJECT_DIR/.env"
ssh "$VPS" "chown dashapp:dashapp $PROJECT_DIR/.env && chmod 600 $PROJECT_DIR/.env"

echo "[2/5] Creating dashapp deploy key (if missing) and configuring git..."
PUBKEY=$(ssh "$VPS" '
  sudo -u dashapp mkdir -p /home/dashapp/.ssh
  sudo -u dashapp chmod 700 /home/dashapp/.ssh
  if [ ! -f /home/dashapp/.ssh/id_ed25519 ]; then
    sudo -u dashapp ssh-keygen -t ed25519 -N "" -q -f /home/dashapp/.ssh/id_ed25519 -C "mcd-vps-data-push"
  fi
  ssh-keyscan -t ed25519 github.com 2>/dev/null | sudo -u dashapp tee -a /home/dashapp/.ssh/known_hosts >/dev/null
  sudo -u dashapp sort -u /home/dashapp/.ssh/known_hosts -o /home/dashapp/.ssh/known_hosts
  sudo -u dashapp git -C '"$PROJECT_DIR"' config user.name  "mcd-vps-bot"
  sudo -u dashapp git -C '"$PROJECT_DIR"' config user.email "dashapp@myanmarconflictdashboard.com"
  sudo -u dashapp git -C '"$PROJECT_DIR"' remote set-url --push origin git@github.com:'"$REPO"'.git
  cat /home/dashapp/.ssh/id_ed25519.pub
')
echo "    deploy key: $PUBKEY"

echo "[2/5] Registering the deploy key on GitHub (write access, this repo only)..."
TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential fill | grep "^password=" | cut -d= -f2)
HTTP=$(curl -s -o /tmp/mcd_key_resp.json -w "%{http_code}" --max-time 20 \
  -X POST -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/keys" \
  -d "{\"title\":\"VPS data push (dashapp)\",\"key\":\"$PUBKEY\",\"read_only\":false}")
if [ "$HTTP" = "201" ]; then
  echo "    registered."
elif grep -q "key is already in use" /tmp/mcd_key_resp.json 2>/dev/null; then
  echo "    already registered — OK."
else
  echo "ERROR: GitHub API returned $HTTP:"; cat /tmp/mcd_key_resp.json; exit 1
fi

echo "[3/5] Installing root cron entry (00:30 UTC = 07:00 Yangon)..."
ssh "$VPS" "crontab -l 2>/dev/null | grep -qF 'vps-update-data.sh' || (crontab -l 2>/dev/null; echo '$CRON_LINE') | crontab -"
ssh "$VPS" "crontab -l | tail -2"

echo "[4/5] Waiting for the VPS checkout to contain an EXECUTABLE scripts/vps-update-data.sh..."
for i in $(seq 1 20); do
  if ssh "$VPS" "[ -x $PROJECT_DIR/scripts/vps-update-data.sh ]"; then
    break
  fi
  sleep 15
done
# Strictly require the execute bit — cron runs the path directly, and a
# plain -f fallback here once masked a missing +x for 12 days.
ssh "$VPS" "[ -x $PROJECT_DIR/scripts/vps-update-data.sh ]" || { echo "ERROR: script missing or not executable on VPS — check git file mode (needs 100755)."; exit 1; }

echo "[5/5] Test run on the VPS..."
BEFORE=$(git ls-remote https://github.com/$REPO.git refs/heads/main | cut -f1)
ssh "$VPS" "bash $PROJECT_DIR/scripts/vps-update-data.sh"
ssh "$VPS" "tail -8 /var/log/dashboard-data-update.log"
AFTER=$(git ls-remote https://github.com/$REPO.git refs/heads/main | cut -f1)
echo ""
echo "origin/main before: $BEFORE"
echo "origin/main after:  $AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "SUCCESS: the VPS fetched, committed, and pushed on its own."
else
  echo "NOTE: no new commit — fine if there was genuinely nothing new to record;"
  echo "check the log lines above for 'finished OK'."
fi
echo "healthz: $(curl -s --max-time 10 https://myanmarconflictdashboard.com/healthz)"
echo ""
echo "Provisioning complete. Tell Claude it's done so the GitHub workflow"
echo "can be switched to monitor-only."
