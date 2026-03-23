#!/usr/bin/env bash
# =============================================================================
# Deploy trading bot to Hetzner VM
# =============================================================================
# Usage:
#   bash scripts/deploy.sh              # deploy + restart bot
#   bash scripts/deploy.sh --no-restart # deploy code only
# =============================================================================

set -euo pipefail

SERVER="${SERVER:-89.167.102.41}"
USER="${SERVER_USER:-root}"
DEST="${DEST:-/opt/trading/IBKR}"
VENV="$DEST/.venv"

RESTART=true
for arg in "$@"; do
    [[ "$arg" == "--no-restart" ]] && RESTART=false
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploying to $USER@$SERVER:$DEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Sync code ─────────────────────────────────────────────────────────────────
echo "[1/3] Syncing code ..."
rsync -az --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.env' \
    --exclude='data/cache/' \
    --exclude='logs/' \
    /mnt/d/Trading-Finance/IBKR/ \
    "$USER@$SERVER:$DEST/"

# ── Install/update Python deps ────────────────────────────────────────────────
echo "[2/3] Updating Python dependencies ..."
ssh "$USER@$SERVER" "
    cd $DEST
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    .venv/bin/pip install -q -r requirements.txt
"

# ── Restart services ──────────────────────────────────────────────────────────
if $RESTART; then
    echo "[3/3] Restarting trading-bot service ..."
    ssh "$USER@$SERVER" "systemctl restart trading-bot && sleep 2 && systemctl status trading-bot --no-pager -l"
else
    echo "[3/3] Skipping restart (--no-restart)"
fi

echo ""
echo "  Done. Monitor with:"
echo "    ssh $USER@$SERVER 'journalctl -fu trading-bot'"
echo "  Or:"
echo "    python scripts/monitor.py --host $SERVER"
echo ""
