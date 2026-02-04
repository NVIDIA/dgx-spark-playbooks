#!/bin/bash
set -euo pipefail

# DGX Spark MCP Server - Playbook Installation Script
# Installs the server from NPM and configures systemd

# Configuration
PACKAGE_NAME="dgx-spark-mcp"
SERVICE_NAME="dgx-spark-mcp"
CONFIG_DIR="/etc/dgx-spark-mcp"
LOG_DIR="/var/log/dgx-spark-mcp"
USER="dgx"
GROUP="dgx"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root" 
   exit 1
fi

# 1. Install Node.js (if missing) - Brief check
if ! command -v node &> /dev/null; then
    log_info "Node.js not found. Please install Node.js 18+."
    exit 1
fi

# 2. Install Package
log_info "Installing $PACKAGE_NAME from registry..."
npm install -g $PACKAGE_NAME

# 3. Create User
if ! id -u "$USER" &>/dev/null; then
    log_info "Creating user $USER..."
    useradd --system --no-create-home --shell /bin/false "$USER"
fi

# 4. Setup Directories
log_info "Setting up directories..."
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

# Copy config if provided in playbook
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../config/default.json" ]]; then
    cp "$SCRIPT_DIR/../config/default.json" "$CONFIG_DIR/config.json"
else
    log_info "No default config found, using internal defaults."
fi

# Permissions
chown -R "$USER:$GROUP" "$LOG_DIR"
chown -R "$USER:$GROUP" "$CONFIG_DIR"
chmod 755 "$LOG_DIR"
chmod 755 "$CONFIG_DIR"

# 5. Setup Service
log_info "Configuring systemd service..."
if [[ -f "$SCRIPT_DIR/../deploy/$SERVICE_NAME.service" ]]; then
    cp "$SCRIPT_DIR/../deploy/$SERVICE_NAME.service" "/etc/systemd/system/$SERVICE_NAME.service"
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    systemctl restart "$SERVICE_NAME"
    log_info "Service started."
else
    log_error "Service file not found."
    exit 1
fi

log_info "Installation complete."
log_info "Status: systemctl status $SERVICE_NAME"
