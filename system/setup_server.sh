#!/usr/bin/env bash
# =============================================================================
# ICT Trading Bot — One-Shot Server Setup
# =============================================================================
# Run this ONCE on a fresh Hetzner (Ubuntu 22.04) VM.
# What it does:
#   1. Installs system deps (Java, Python, Xvfb)
#   2. Installs IB Gateway (stable channel)
#   3. Installs IBC (headless login automation)
#   4. Installs Python deps for the trading bot
#   5. Creates systemd services (ibkr-gateway + trading-bot)
#   6. Sets up log rotation
#
# Usage:
#   scp system/setup_server.sh root@YOUR_SERVER_IP:/tmp/
#   ssh root@YOUR_SERVER_IP "bash /tmp/setup_server.sh"
# =============================================================================

set -euo pipefail

# ── Configurable ──────────────────────────────────────────────────────────────
BOT_USER="${BOT_USER:-trader}"
BOT_DIR="${BOT_DIR:-/opt/trading/IBKR}"
IBC_DIR="${IBC_DIR:-/opt/ibc}"
GATEWAY_DIR="${GATEWAY_DIR:-/opt/ibgateway}"
IB_GATEWAY_VERSION="${IB_GATEWAY_VERSION:-10.26}"   # check latest at ibkr.com/en/trading/tws-latest-stable.php
IBC_VERSION="${IBC_VERSION:-3.19.0}"                # check https://github.com/IbcAlpha/IBC/releases
PYTHON="${PYTHON:-python3}"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn ]${NC} $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Must run as root ──────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || die "Run as root: sudo bash $0"

# =============================================================================
# 1. System packages
# =============================================================================
info "Updating packages ..."
apt-get update -qq
apt-get install -y -qq \
    openjdk-11-jre \
    python3 python3-pip python3-venv \
    xvfb x11-utils \
    curl wget unzip git \
    supervisor \
    logrotate \
    jq

# =============================================================================
# 2. Create bot user (unprivileged)
# =============================================================================
if ! id "$BOT_USER" &>/dev/null; then
    info "Creating user: $BOT_USER"
    useradd -m -s /bin/bash "$BOT_USER"
fi

# =============================================================================
# 3. Install IB Gateway (stable)
# =============================================================================
info "Installing IB Gateway $IB_GATEWAY_VERSION ..."
GATEWAY_INSTALLER="/tmp/ibgateway-installer.sh"

# Download stable installer from IBKR
curl -fsSL \
    "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh" \
    -o "$GATEWAY_INSTALLER" || \
warn "Could not auto-download IB Gateway. Download manually from IBKR website
  → https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
  → Linux x64 (offline installer)
  → Place it at: $GATEWAY_INSTALLER
  → Re-run this script."

if [[ -f "$GATEWAY_INSTALLER" ]]; then
    chmod +x "$GATEWAY_INSTALLER"
    # Install headlessly — IBC will manage authentication
    echo -e "\n\n\n" | bash "$GATEWAY_INSTALLER" \
        -q \
        -dir "$GATEWAY_DIR" \
        -java_home /usr/lib/jvm/java-11-openjdk-amd64 \
        2>/dev/null || true
    info "IB Gateway installed at $GATEWAY_DIR"
else
    warn "Skipping IB Gateway install — file not found at $GATEWAY_INSTALLER"
fi

# =============================================================================
# 4. Install IBC
# =============================================================================
info "Installing IBC $IBC_VERSION ..."
IBC_ZIP="/tmp/IBC-${IBC_VERSION}.zip"
curl -fsSL \
    "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" \
    -o "$IBC_ZIP"

mkdir -p "$IBC_DIR"
unzip -q -o "$IBC_ZIP" -d "$IBC_DIR"
chmod +x "$IBC_DIR"/*.sh "$IBC_DIR"/scripts/*.sh 2>/dev/null || true

chown -R "$BOT_USER:$BOT_USER" "$IBC_DIR"
info "IBC installed at $IBC_DIR"

# =============================================================================
# 5. IBC config (filled from environment / prompts)
# =============================================================================
info "Creating IBC config ..."

IBC_CONFIG="$IBC_DIR/config.ini"
if [[ ! -f "$IBC_CONFIG" ]]; then
    cat > "$IBC_CONFIG" << 'IBCCONF'
# IBC Configuration
# See: https://github.com/IbcAlpha/IBC/blob/master/userguide.md

[IBController]
FIX=no

# ── Credentials (set these before starting) ──
IbLoginId=FILL_YOUR_IBKR_USERNAME
IbPassword=FILL_YOUR_IBKR_PASSWORD

# Paper account credentials (if different from live)
# IbLoginId2=
# IbPassword2=

# trading mode: paper | live
TradingMode=paper

ReadonlyLogin=no
AcceptNonBrokerageAccountWarning=yes
AcceptBidirectionalOrderConfirmation=yes
DismissNSEComplianceNotice=yes
ReloginAfterSecondFactorAuthenticationTimeout=yes
SecondFactorAuthenticationExitInterval=60
IBCCONF
    warn "Edit $IBC_CONFIG and fill IbLoginId / IbPassword before starting."
fi

# =============================================================================
# 6. Trading bot
# =============================================================================
info "Setting up trading bot at $BOT_DIR ..."
mkdir -p "$BOT_DIR"
chown -R "$BOT_USER:$BOT_USER" "$BOT_DIR"

# Python virtual environment
if [[ ! -d "$BOT_DIR/.venv" ]]; then
    su - "$BOT_USER" -c "python3 -m venv $BOT_DIR/.venv"
fi

# Install Python deps (requirements.txt will be rsync'd from local)
if [[ -f "$BOT_DIR/requirements.txt" ]]; then
    su - "$BOT_USER" -c "$BOT_DIR/.venv/bin/pip install -q -r $BOT_DIR/requirements.txt"
    info "Python dependencies installed."
fi

# =============================================================================
# 7. Systemd services
# =============================================================================
info "Installing systemd services ..."

# Copy service files
cp "$(dirname "$0")/ibkr-gateway.service" /etc/systemd/system/ 2>/dev/null || \
    warn "ibkr-gateway.service not found — copy manually from system/ directory"

cp "$(dirname "$0")/trading-bot.service" /etc/systemd/system/ 2>/dev/null || \
    warn "trading-bot.service not found — copy manually from system/ directory"

systemctl daemon-reload

# Enable but don't start yet — credentials need to be filled first
systemctl enable ibkr-gateway trading-bot 2>/dev/null || true

# =============================================================================
# 8. Log rotation
# =============================================================================
info "Setting up log rotation ..."
cat > /etc/logrotate.d/trading-bot << LOGROTATE
$BOT_DIR/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 $BOT_USER $BOT_USER
    postrotate
        systemctl kill -s HUP trading-bot.service 2>/dev/null || true
    endscript
}
LOGROTATE

# =============================================================================
# 9. UFW firewall (allow SSH, block everything else inbound)
# =============================================================================
if command -v ufw &>/dev/null; then
    info "Configuring firewall ..."
    ufw --force reset >/dev/null
    ufw default deny incoming >/dev/null
    ufw default allow outgoing >/dev/null
    ufw allow ssh >/dev/null
    # IB Gateway API port — only from localhost
    # (the bot connects to 127.0.0.1:4002, no external access needed)
    ufw --force enable >/dev/null
    info "Firewall: SSH allowed, all other inbound blocked."
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete.${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo "  1. Edit IBC credentials:"
echo "       nano $IBC_CONFIG"
echo "       (fill IbLoginId and IbPassword)"
echo ""
echo "  2. Deploy bot code from local machine:"
echo "       bash scripts/deploy.sh"
echo ""
echo "  3. Create .env on the server:"
echo "       cp $BOT_DIR/.env.example $BOT_DIR/.env"
echo "       nano $BOT_DIR/.env"
echo ""
echo "  4. Start services:"
echo "       systemctl start ibkr-gateway"
echo "       sleep 30"
echo "       systemctl start trading-bot"
echo ""
echo "  5. Monitor:"
echo "       journalctl -fu trading-bot"
echo "       python3 $BOT_DIR/scripts/monitor.py"
echo ""
