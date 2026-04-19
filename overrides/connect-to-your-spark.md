---
description: Set up SSH access to an NVIDIA DGX Spark from a laptop using NVIDIA Sync (recommended) or manual SSH. Use when a user is new to their Spark and needs to connect remotely, before doing anything else. This is a prerequisite for nearly every other dgx-spark-* skill — if a user hasn't set this up, do this first.
---

## When to use this skill
- User just got their DGX Spark and wants to use it from their laptop
- Any other dgx-spark-* skill needs SSH access and the user hasn't configured it yet
- User reports "can't connect to my Spark" or "SSH hangs / can't resolve spark-abcd.local"

## Two paths — help the user pick
- **NVIDIA Sync (recommended)** — GUI, handles SSH key generation + aliasing + port forwarding for apps. Required if they want one-click app launchers (DGX Dashboard, VS Code, Open WebUI tunnels).
- **Manual SSH** — if they prefer CLI-only workflow, or Sync isn't supported on their platform.

Most users should use NVIDIA Sync unless they have a specific reason not to.

## Key decisions
- **Hostname vs IP** — default is mDNS hostname (`spark-abcd.local`). On corporate networks that block mDNS, they'll need to use the IP address from their router's admin panel. Quick test: `ping spark-abcd.local` — if it hangs, mDNS is blocked.
- **First-boot wait** — after initial system setup, the Spark can take 3–4 minutes to finish updates before SSH becomes available. Don't diagnose connection issues in this window.

## Non-obvious gotchas
- NVIDIA Sync's password prompt happens **once** — it uses the password only to install the SSH key, then discards it. If auth fails, the key install didn't complete; re-run the add-device flow.
- mDNS `.local` resolution is OS + network-stack specific. Works on most home Wi-Fi; often broken on corporate VPNs or guest networks.
- Port-forwarding for web apps is a separate step (SSH `-L` flag or Custom Ports in Sync) — connecting to SSH alone doesn't give laptop browsers access to web UIs running on the Spark.

## Related skills
- **Alternative**: `dgx-spark-tailscale` — use Tailscale VPN for remote access instead of local-network SSH. Works off-network.
- **Follow-ups (what users typically do next)**:
  - `dgx-spark-ollama` — run a local LLM
  - `dgx-spark-open-webui` — web chat UI
  - `dgx-spark-vscode` — remote development
  - `dgx-spark-dgx-dashboard` — system monitoring (already pre-installed, just needs the tunnel)
- **Multi-Spark setups depend on this first**: `dgx-spark-connect-two-sparks`, `dgx-spark-connect-three-sparks`, `dgx-spark-multi-sparks-through-switch`
