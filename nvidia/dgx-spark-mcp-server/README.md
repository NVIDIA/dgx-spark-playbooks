# DGX Spark MCP Server Playbook

This playbook installs and configures the **DGX Spark MCP Server**, a tool that provides hardware-aware Apache Spark optimization for NVIDIA DGX systems via the Model Context Protocol (MCP).

## Overview

The DGX Spark MCP Server enables MCP clients (like Claude Desktop or Claude Code) to:
*   **Detect Hardware**: Automatically read DGX GPU topology, memory, and CPU specs.
*   **Optimize Spark**: Generate tuned Spark configurations (`spark-submit` args) based on detected hardware and workload type (ETL, ML Training, Inference).
*   **Monitor**: Check real-time GPU availability before submitting jobs.

## Prerequisites

*   **NVIDIA DGX System** (or compatible GPU server)
*   **NVIDIA Drivers** installed (`nvidia-smi` available)
*   **Node.js 18+**
*   **Root access** (for systemd service installation)

## Directory Structure

```
.
├── config/
│   └── default.json       # Default configuration
├── deploy/
│   └── dgx-spark-mcp.service # Systemd service file
└── scripts/
    └── install.sh         # Automated installer
```

## Installation

1.  **Run the installer**:
    ```bash
    sudo ./scripts/install.sh
    ```
    This script will:
    *   Install `dgx-spark-mcp` globally via `npm`.
    *   Create a dedicated system user (`dgx`).
    *   Setup logging directory `/var/log/dgx-spark-mcp`.
    *   Install and start the systemd service.

2.  **Verify Installation**:
    ```bash
    systemctl status dgx-spark-mcp
    ```

## Configuration

The configuration file is located at `/etc/dgx-spark-mcp/config.json`.

### Key Settings

*   **`mcp.transport`**: `stdio` (default) or `sse`.
*   **`hardware.enableGpuMonitoring`**: Set to `true` to enable real-time `nvidia-smi` queries.
*   **`logging.level`**: `info` or `debug`.

## Usage with Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dgx-spark": {
      "command": "dgx-spark-mcp"
    }
  }
}
```

## Troubleshooting

**Service fails to start?**
Check logs:
```bash
journalctl -u dgx-spark-mcp -f
```

**Permission denied?**
Ensure the `dgx` user has permissions to access `nvidia-smi`. You may need to add the user to the `video` group:
```bash
usermod -a -G video dgx
```
