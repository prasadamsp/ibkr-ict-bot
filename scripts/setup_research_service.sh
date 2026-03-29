#!/bin/bash
# setup_research_service.sh — Install research daemon as systemd service on the VM
# Run as root: bash scripts/setup_research_service.sh

set -euo pipefail

SERVICE_FILE=/etc/systemd/system/research-daemon.service
IBKR_DIR=/opt/trading/IBKR

cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=ICT Research Daemon (weekly algo grid search)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trader
Group=trader
WorkingDirectory=/opt/trading/IBKR

ExecStart=/opt/trading/IBKR/.venv/bin/python research/run_research.py

Restart=on-failure
RestartSec=300
StartLimitIntervalSec=3600
StartLimitBurst=3

Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-/opt/trading/IBKR/.env

MemoryMax=1G
CPUQuota=80%

StandardOutput=append:/opt/trading/IBKR/logs/research_daemon.log
StandardError=append:/opt/trading/IBKR/logs/research_daemon.log

TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable research-daemon.service
systemctl start  research-daemon.service

echo "research-daemon.service installed and started."
echo "Check status: systemctl status research-daemon"
echo "View logs:    tail -f /opt/trading/IBKR/logs/research_daemon.log"
echo ""
echo "To run immediately (one-shot):"
echo "  sudo -u trader /opt/trading/IBKR/.venv/bin/python research/run_research.py --once"
