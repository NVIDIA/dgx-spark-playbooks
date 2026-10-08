# Secure AI Agents with OpenShell

> Isolate AI coding agents like Pi with kernel-level policies and a locally served vLLM model

## Table of Contents

- [Overview](#overview)
  - [Notice & Disclaimers](#notice-disclaimers)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

AI coding and agent tools run locally with access to your files, credentials, and network — running one directly on your system means it can reach all of that, creating real security exposure.

**NVIDIA OpenShell** solves this problem. It is an open-source sandbox runtime that wraps an agent in kernel-level isolation with declarative YAML policies. OpenShell controls what the agent can read on disk, which network endpoints it can reach, and what privileges it has—without disabling the capabilities that make the agent useful.

This playbook uses [Pi](https://pi.dev), a terminal coding agent, as its reference agent, backed by a model served locally with vLLM on your own hardware. You get the full power of a local AI agent with local model serving, while enforcing explicit controls over filesystem access, network egress, and credential handling.

Get started with additional reference examples using [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw), including express installers for DGX Spark and DGX Station.

### Notice & Disclaimers

#### Quick Start Safety Check

Use a clean environment only. Run this playbook on a fresh device or VM with no personal data, confidential information, or sensitive credentials. Think of it like a sandbox—keep it isolated.

By installing this playbook, you're taking responsibility for all third-party components, including reviewing their licenses, terms, and security posture. Read and accept before you install or use.

---

#### What You're Getting

The playbook showcases experimental AI agent capabilities. Even with cutting-edge open-source tools like OpenShell in your toolkit, you need to layer in proper security measures for your specific threat model.

---

#### Key Risks with AI Agents

Be mindful of these risks with AI agents:

1. **Data leakage** – Any materials the agent accesses could be exposed, leaked, or stolen.
2. **Malicious code execution** – The agent or its connected tools could expose your system to malicious code or cyber-attacks.
3. **Unintended actions** – The agent might modify or delete files, send messages, or access services without explicit approval.
4. **Prompt injection & manipulation** – External inputs or connected content could hijack the agent's behavior in unexpected ways.

---

#### Security Best Practices

No system is perfect, but these practices help keep your information and systems safe:

1. **Isolate your environment** – Run on a clean PC or isolated virtual machine. Only provision the specific data you want the agent to access.
2. **Never use real accounts** – Don't connect personal, confidential, or production accounts. Create dedicated test accounts with minimal permissions.
3. **Vet your skills/plugins** – Only enable skills from trusted sources that have been vetted by the community.
4. **Lock down access** – If your agent exposes a web UI or messaging channel (via `openshell service expose` or similar), ensure it isn't accessible over the network without proper authentication.
5. **Restrict network access** – Where feasible, limit the agent's internet connectivity.
6. **Clean up after yourself** – When you're done, delete the sandbox and provider credentials you created, and revoke any API keys or account access you granted.

---

## What you'll accomplish

You will install the OpenShell CLI (`openshell`), deploy a gateway on your hardware platform, build an agent image containing Pi, and launch it inside a sandbox with a custom provider profile pointing at a model served locally with vLLM. The sandbox enforces filesystem, network, and process isolation by default — you grant the agent exactly the network endpoints and binaries it needs, nothing more, and no external API keys are required.

## Popular use cases

- **Secure agent experimentation**: Test Pi's tool integrations and skills without exposing your main filesystem or credentials to the agent.
- **Private enterprise development**: Route all inference to a local model on your hardware. No data leaves the machine unless you explicitly allow it in the policy.
- **Auditable agent access**: Version-control the policy YAML alongside your project. Review exactly what the agent can reach before granting access.
- **Iterative policy tuning**: Monitor denied connections in real time with `openshell term`, then hot-reload updated policies without recreating the sandbox.

## What to know before starting

**Required:**

- Comfort with the Linux terminal and SSH
- Basic understanding of Docker (OpenShell runs your gateway and sandboxes as containers)
- Familiarity with local LLM serving (this playbook uses vLLM)

**Optional:**

- Awareness of the security model: OpenShell reduces risk through isolation but cannot eliminate all risk. Review the [OpenShell documentation](https://docs.nvidia.com/openshell/latest/).

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. The same OpenShell + Pi workflow applies across supported hardware platforms. Model recommendations differ by platform — see [Agent-ready Models](https://build.nvidia.com/spark/vllm/agent-ready-models).

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |

> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for your chosen agent-ready model (see [Agent-ready Models](https://build.nvidia.com/spark/vllm/agent-ready-models))

**Software requirements**

- Linux (DGX OS / Ubuntu 24.04 or compatible)
- Docker Desktop or Docker Engine running: `docker info`
- Python 3.12 or later: `python3 --version`
- NVIDIA Container Toolkit configured for Docker
- Network access to download packages from PyPI / GitHub, container images, and model weights from Hugging Face
- A Hugging Face token when the chosen model requires authentication

## Time & risk

- **Estimated time:** 30 MIN (plus model download time, which depends on model size and network speed)
- **Risk level:** Medium
  - OpenShell sandboxes enforce kernel-level isolation, significantly reducing the risk compared to running an agent directly on the host.
  - The sandbox default policy denies all outbound traffic not explicitly allowed. Misconfigured policies may block legitimate agent traffic; use `openshell logs` to diagnose.
  - Large model downloads may fail on unstable networks.
- **Rollback:** Delete the sandbox with `openshell sandbox delete <sandbox-name>`, stop the gateway, and remove the vLLM container if you started one. See cleanup in the **Instructions** tab.
- **Last Updated:** 09/30/2026
  - Rewritten for OpenShell 0.1.0: reference agent changed from OpenClaw to Pi, custom provider profiles replace managed inference routing, and the sandbox deploy flow builds a bring-your-own agent image instead of using a prebuilt community sandbox

## Instructions

## Step 1. Confirm your environment

Verify the OS, GPU, Docker, and Python are available on your device.

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
```

Expected output should show Ubuntu 24.04 (or compatible DGX OS), a detected GPU, a Docker server version, and Python 3.12+.

> [!NOTE]
> This playbook requires OpenShell 0.1.0 or later. If an OpenShell installation is already present on this machine, uninstall it first (see the [OpenShell uninstall docs](https://docs.nvidia.com/openshell/latest/about/installation#uninstall-openshell)) before continuing.

## Step 2. Docker configuration

Verify that the local user has Docker permissions:

```bash
docker ps
```

If you get a permission denied error (`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`), add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Reboot (or log out and back in) so the group membership applies to all sessions.

Configure Docker to use the NVIDIA Container Runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access inside a container:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## Step 3. Install the OpenShell CLI

Install OpenShell with the official installer, which installs the CLI, the policy prover, and a local gateway in one step, then starts the gateway automatically:

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

> [!NOTE]
> To pin a specific release, set `OPENSHELL_VERSION` to a release tag (`OPENSHELL_VERSION=v0.1.2 sh`) or `OPENSHELL_VERSION=pre`/`dev` for a prerelease/rolling build. See the [OpenShell installation docs](https://docs.nvidia.com/openshell/latest/about/installation) for macOS, Snap, and Kubernetes install paths. The installer is the only supported install route and is required for the gateway.

The installer picks a package for your platform (a `.deb` on Ubuntu, `.rpm` on Fedora/RHEL) and registers the `openshell-gateway` systemd user service. Open a new shell (or `source ~/.bashrc`) so `openshell` is on your `PATH`, then confirm the CLI can reach the gateway:

```bash
openshell status
```

Expected: `Status: Connected`.

## Step 4. Verify the OpenShell gateway

The installer manages the gateway as a systemd user service, listening at `https://127.0.0.1:17670` and reading `~/.config/openshell/gateway.toml`.

> [!TIP]
> To manage a gateway on remote hardware from a separate workstation, register it as a named gateway instead of connecting to `127.0.0.1`: `openshell gateway add https://<hardware-ip-or-hostname>:17670 --name <name>` (mTLS gateways need the CLI client certificate in place first). See [Manage Gateways](https://docs.nvidia.com/openshell/latest/how-it-works/gateways/overview) for the full registration and authentication flow.

Confirm the service is running, and that the CLI can reach it:

```bash
systemctl --user status --no-pager openshell-gateway
openshell status
```

`openshell status` should report `Status: Connected` and `Authentication: Authenticated`. If the service is not running, start it:

```bash
systemctl --user start openshell-gateway
systemctl --user status --no-pager openshell-gateway
openshell status
```

To keep the gateway available after you log out:

```bash
sudo loginctl enable-linger $USER
```

Follow gateway logs in real time (press `Ctrl+C` to exit) using the following command in a new terminal window:

```bash
journalctl --user -u openshell-gateway -f
```

## Step 5. Serve a model with vLLM

Serve an OpenAI-compatible API for local inference. Use the recommended model and launch recipe for your hardware platform from [Agent-ready Models](https://build.nvidia.com/spark/vllm/agent-ready-models) (covers DGX Spark, DGX Station, and RTX PRO). Keep `--host 0.0.0.0` and port `8000` so sandboxes can reach the server.

> [!IMPORTANT]
> Do not bind the server to `localhost` only. Sandboxes reach host-local services through the `host.openshell.internal` DNS alias (Step 6), which resolves to the gateway host — not to `127.0.0.1` inside the sandbox's own network namespace.

Once the server reports `Application startup complete`, verify the localhost endpoint:

```bash
curl -sf http://localhost:8000/v1/models
```

Expected: a JSON `"data"` array listing your model handle. Export the reported `id` as an environment variable, used throughout the remaining steps:

```bash
export MODEL_HANDLE=your-model-id
```

## Step 6. Create a provider profile for the local vLLM server

Providers attach directly to a sandbox. The agent inside calls the provider's endpoint directly. This playbook runs inference against a self-hosted vLLM server on your own hardware, so it needs a provider profile declaring that specific host and port — OpenShell's example profiles (like `openai`, in the [providers directory](https://github.com/NVIDIA/OpenShell/tree/main/providers)) are scoped to their public vendor endpoints (`api.openai.com`, etc.) and aren't meant to be redirected to a different host.

Save a profile declaring your vLLM server's host, port, and the binaries allowed to call it:

```bash
cat > local-vllm.yaml <<'EOF'
id: local-vllm
display_name: Local vLLM
description: Host-local vLLM OpenAI-compatible API
category: inference
inference_capable: true
credentials: []
endpoints:
  - host: host.openshell.internal
    port: 8000
    protocol: rest
    access: read-write
    enforcement: enforce
binaries:
  - /usr/local/bin/node
  - /usr/bin/curl
  - /usr/local/bin/curl
  - /usr/bin/python3
  - /usr/local/bin/python
  - /sandbox/.uv/python/**
  - /sandbox/.venv/**
EOF
```

> [!NOTE]
> `host.openshell.internal` is the DNS alias sandboxes use to reach services running on the gateway host. The network policy only grants access to binaries named in this list, matched against the kernel-resolved target of the calling process (check with `readlink -f <path>` inside the sandbox if a symlinked interpreter is denied unexpectedly). `/usr/local/bin/node` covers Pi (Step 7); add or remove entries to match the agent image you build.

vLLM does not require an API key, so the profile declares no credentials (`credentials: []`), and the provider is created without any `--credential` flag.

Lint and import the profile, then create a provider from it:

```bash
openshell profile lint -f local-vllm.yaml
openshell profile import -f local-vllm.yaml
openshell provider create --name local-vllm --type local-vllm
```

Verify:

```bash
openshell provider list
```

## Step 7. Build the Pi agent image

This playbook uses [Pi](https://pi.dev), a terminal coding agent, as its reference agent. Build an image containing Pi and pass it to `sandbox create --from`.

Save the Dockerfile:

```bash
cat > Dockerfile.pi <<'EOF'
FROM node:24-bookworm-slim

ARG PI_VERSION=latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates fd-find git ripgrep \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}"

RUN mkdir -p /workspace && chown node:node /workspace
USER node
WORKDIR /workspace

ENV PI_CODING_AGENT_DIR=/tmp/pi-agent
COPY --chown=node:node models.json /tmp/pi-agent/models.json
EOF
```

`fd`/`ripgrep` are baked in because the sandbox policy blocks Pi's fallback download of them on first use. `PI_CODING_AGENT_DIR` points at `/tmp` because that's one of the few paths the default sandbox policy lets Pi write to.

Next to `Dockerfile.pi`, save `models.json`, declaring your local vLLM server as a custom OpenAI-compatible provider. This uses the `$MODEL_HANDLE` you exported in Step 5:

```bash
cat > models.json <<EOF
{
  "providers": {
    "local-vllm": {
      "baseUrl": "http://host.openshell.internal:8000/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        { "id": "$MODEL_HANDLE" }
      ]
    }
  }
}
EOF
```

Build the image:

```bash
docker build -t pi-agent:local -f Dockerfile.pi .
```

> [!NOTE]
> If your gateway uses Podman, build with `podman build -t localhost/pi-agent:local -f Dockerfile.pi .` and use `localhost/pi-agent:local` below. If the gateway runs on different hardware than this shell, push the image to a registry the gateway can pull from instead.

## Step 8. Deploy the sandbox and start Pi

Create the sandbox, attach the `local-vllm` provider from Step 6, and launch Pi as the sandbox's main process with the model preselected (no interactive `/model` picker needed):

```bash
export SANDBOX_NAME=openshell-demo

openshell sandbox create \
  --name "$SANDBOX_NAME" \
  --from pi-agent:local \
  --provider local-vllm \
  -- pi --model local-vllm/"$MODEL_HANDLE"
```

This uses the same `$MODEL_HANDLE` exported in Step 5 and baked into Step 7's `models.json`. OpenShell allocates a TTY automatically when both `stdin` and `stdout` are terminals; add `--tty` explicitly if you run this from a script or wrapper.

By default OpenShell attaches to Pi's session in this terminal and retains the sandbox after Pi exits (pass `--no-keep` for an ephemeral run instead).

Once connected to the sandbox, try a prompt allowed by the OpenShell policy defined in Step 6:

```text
Explain what files are available in this workspace.
```

Then try a prompt that reaches outside the sandbox's granted network access, to confirm isolation is enforced and not just assumed:

```text
Fetch the contents of https://example.com and summarize it.
```

Expect this one to fail with an ` Error: Connection error.`. Step 6's `local-vllm.yaml` declares exactly one network endpoint — `host: host.openshell.internal`, `port: 8000` — and the sandbox denies every destination not explicitly listed in an attached profile's `endpoints`, so `example.com:443` is blocked before the connection leaves the sandbox. Confirm the exact reason with `openshell logs "$SANDBOX_NAME" --tail` (Step 9): look for a `DENIED` line citing `transparent_tcp_policy_denied`.

Detach without stopping Pi: press `Ctrl-P` then `Ctrl-Q`. Later steps reconnect to this same running session.

> [!IMPORTANT]
> Don't type `/quit` yet. Quitting ends Pi as the sandbox's main process — the sandbox is retained for inspection (`sandbox get`, `policy get`, `logs`) afterward, but can no longer be reconnected to or exec'd into. `openshell sandbox delete` (Step 14) tears the sandbox down regardless of whether Pi was quit or just detached, so there's no need to quit manually before cleanup.

## Step 9. Inspect the sandbox

Open a second terminal to check status, effective policy, attached providers, and logs:

```bash
openshell sandbox list
openshell policy get "$SANDBOX_NAME" --full
openshell sandbox provider list "$SANDBOX_NAME"
openshell logs "$SANDBOX_NAME" --tail
```

`openshell logs --tail` streams outbound connections and policy decisions (`allow`/`deny`) in real time — useful for confirming Pi's requests are reaching `host.openshell.internal:8000` and nothing else.

> [!NOTE]
> A successful chat turn with Pi in Step 8 confirms inference connectivity. If Pi can't reach the model, check `openshell sandbox provider list` and the policy/logs commands above before assuming vLLM itself is unhealthy.

## Step 10. Verify sandbox isolation

With Pi running, open `openshell term` for a live dashboard of sandbox status and the log stream:

```bash
openshell term
```

Confirm Pi's traffic to `host.openshell.internal:8000` is allowed and that unrelated outbound traffic is denied.

> [!TIP]
> Press `f` to follow live output, `s` to filter by source, and `q` to quit.

## Step 11 (Optional). Update the local-vllm policy

Grant Pi access to an additional destination — for example `raw.githubusercontent.com:443`, so Pi can fetch the OpenShell README directly from GitHub. This edits the `local-vllm` profile from Step 6, so the change applies everywhere that profile is attached, including the sandbox already running from Step 8.

> [!NOTE]
> The profile scopes access by host and port, not by URL path — granting `raw.githubusercontent.com` permits reaching any repository's raw content on that host, not only the OpenShell repo.

The profile is where policy lives, so it's the only object you edit. Export it first to capture the `resource_version` that `profile update` requires:

```bash
openshell profile export local-vllm -o yaml > local-vllm.yaml
```

Add the new endpoint to the `endpoints:` list. `profile export` appends metadata fields (`resource_version`, `source`, `scope`) after `binaries:`, so appending to the end of the file would land outside `endpoints:` entirely — insert it right before the `binaries:` line instead:

```bash
sed -i '/^binaries:/i\
  - host: raw.githubusercontent.com\
    port: 443\
    protocol: rest\
    access: read-write\
    enforcement: enforce' local-vllm.yaml
```

Update the profile. `profile lint` validates a profile as new-import input and rejects an `id` that already exists, so it isn't used here — `profile update` performs its own validation against the existing profile:

```bash
openshell profile update local-vllm -f local-vllm.yaml
```

The provider from Step 6 references the profile by `id`, so it automatically reflects the update. New sandboxes inherit it immediately; for the sandbox already running from Step 8, confirm it has synced:

```bash
openshell sandbox provider attach "$SANDBOX_NAME" local-vllm --wait
```

Network-policy changes like this hot-reload into a running sandbox, so Pi can retry a previously denied request without restarting. Credential *value* changes need a new process instead — a running process keeps the placeholder it started with, so exit and restart the agent (or use `openshell sandbox exec`) to pick up new credentials. Reconnect and retry the same style of prompt Step 8 showed being denied, now against the newly granted host:

```bash
openshell sandbox connect "$SANDBOX_NAME"
```

```text
Fetch the contents of https://raw.githubusercontent.com/NVIDIA/OpenShell/main/README.md and summarize it.
```

This one succeeds — `raw.githubusercontent.com:443` is now in `local-vllm`'s `endpoints`, while every other destination (`example.com` included) stays denied.

To revoke `local-vllm` access from a sandbox without deleting the sandbox itself:

```bash
openshell sandbox provider detach "$SANDBOX_NAME" local-vllm --wait
```

## Step 12. Work on a local project (optional)

Point Pi at your own code by uploading a project directory at sandbox-creation time. `--upload` copies the directory in before the sandbox starts, skipping anything `.gitignore` excludes. `--upload` cannot be combined with a trailing main command, so create the sandbox first and start Pi inside it as a separate step:

```bash
openshell sandbox create \
  --name pi-project \
  --from pi-agent:local \
  --provider local-vllm \
  --upload .:/workspace
```

```bash
openshell sandbox exec -n pi-project --tty -- pi --model local-vllm/"$MODEL_HANDLE"
```

Pi's changes stay inside the sandbox. Copy them back with `openshell sandbox download` from another terminal while Pi is still running (see Step 13).

## Step 13. Reconnect and transfer files

Reattach to Pi's running session at any time — this replays recent output and hands you back the same interactive process:

```bash
openshell sandbox connect "$SANDBOX_NAME"
```

Press `Ctrl-P` then `Ctrl-Q` to detach without stopping Pi. For a separate shell alongside Pi, or for scripted access, use:

```bash
openshell sandbox exec -n "$SANDBOX_NAME" --tty -- /bin/bash
openshell sandbox ssh-config "$SANDBOX_NAME"
```

Transfer files without attaching:

```bash
openshell sandbox upload "$SANDBOX_NAME" ./local-file /sandbox/destination
openshell sandbox download "$SANDBOX_NAME" /sandbox/file ./local-destination
```

## Step 14. Cleanup

Delete the sandbox(es) and provider while the gateway is still reachable:

```bash
openshell sandbox delete "$SANDBOX_NAME"
openshell provider delete local-vllm
```

To stop the gateway without uninstalling it:

```bash
systemctl --user stop openshell-gateway
```

To fully remove OpenShell (Ubuntu/Debian shown; see the [uninstall docs](https://docs.nvidia.com/openshell/latest/about/installation#uninstall-openshell) for Fedora/RHEL, Homebrew, and Snap):

```bash
systemctl --user disable --now openshell-gateway
sudo apt remove openshell
rm -rf "${XDG_STATE_HOME:-$HOME/.local/state}/openshell"
```

If you enabled linger in Step 4, disable it:

```bash
sudo loginctl disable-linger $USER
```

> [!NOTE]
> `openshell gateway stop` and `openshell gateway destroy` are not real subcommands (valid: `add`/`remove`/`login`/`logout`/`select`/`info`/`list`) — the gateway is a systemd service, not something the CLI starts or stops directly.

Stop and remove the vLLM container if you started one for this playbook (replace the name/image filter to match your launch):

```bash
docker rm -f vllm-server 2>/dev/null || true
```

## Step 15. Next steps

- **Add more providers**: Attach GitHub tokens, GitLab tokens, or cloud API keys the same way as Step 6 — import a profile, `openshell provider create`, then pass `--provider <name>` when creating a sandbox. See the [providers directory](https://github.com/NVIDIA/OpenShell/tree/main/providers) for ready-made profiles (GitHub, PyPI, hosted model APIs).
- **Bring another agent image**: Build a Dockerfile for any agent (OpenCode, Claude Code, etc.) following the same pattern as Step 7, and write it a provider profile following Step 6.
- **Connect VS Code**: Use `openshell sandbox create --editor vscode` or `openshell sandbox connect <name> --editor cursor` to open the sandbox workspace directly in an editor, or `openshell sandbox ssh-config <sandbox-name>` for manual SSH config.
- **Expose a web UI**: If an agent image serves a dashboard or web UI, use `openshell service expose <sandbox> <port>`.
- **Monitor and audit**: Use `openshell logs <sandbox-name> --tail` or `openshell term` to monitor agent activity and policy decisions.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `openshell status` shows "Connection refused" | The `openshell-gateway` systemd user service is not running, or Docker socket is not accessible from the user service | Start it: `systemctl --user start openshell-gateway`. Check logs: `journalctl --user -u openshell-gateway --no-pager -n 50`. If the service cannot reach Docker, fix socket access with `sudo setfacl -m u:$USER:rw /var/run/docker.sock`, then restart the service |
| Gateway service fails to start after installation | Docker is not running | Start Docker: `sudo systemctl start docker`. Then restart the OpenShell gateway service: `systemctl --user restart openshell-gateway` |
| `openshell status` shows gateway as unhealthy | Gateway service or container crashed / failed to initialize | Run `systemctl --user restart openshell-gateway` and inspect `journalctl --user -u openshell-gateway --no-pager -n 50`. Check Docker with `docker ps -a` and `docker logs <container-id>` |
| `openshell sandbox create --from pi-agent:local` fails because the image isn't found | The image was built with a different container engine than the gateway's compute driver, or wasn't built at all | Confirm the build succeeded: `docker images pi-agent:local`. If the gateway uses Podman, build with `podman build -t localhost/pi-agent:local -f Dockerfile.pi .` and reference `localhost/pi-agent:local`. If the gateway runs on different hardware than this shell, push the image to a registry the gateway can pull from instead of using a local tag |
| Sandbox is in `Error` phase after creation | Policy validation failed, a referenced provider profile is missing, or the container failed to start | Run `openshell logs <sandbox-name>` to see error details. Common causes: invalid policy or profile YAML, a provider profile that was never imported, or port conflicts |
| Agent gets `Error: Connection error.` calling the local provider, even though the endpoint and model look right | The calling binary isn't in the attached profile's `binaries` list — a binary being allowed for one endpoint doesn't grant it access to others, and vice versa | Reproduce the request, then check `openshell logs <sandbox-name> --tail` for a line like `DENIED /path/to/binary -> host:port [reason:transparent_tcp_policy_denied]`. Add that exact binary path to the profile's `binaries` (export → edit → `profile lint` → `profile update`, per Step 11), then `openshell sandbox provider attach <sandbox-name> <provider-name> --wait` |
| Agent gets repeated `Error: Connection error.` against the local vLLM endpoint, and `openshell logs --tail` shows `NET:FAIL` (not `DENIED`) for that destination | vLLM itself isn't responding — crashed, OOM, hung on a prior request, or was never started | From the host, confirm it's actually serving: `curl -s http://0.0.0.0:8000/v1/models`. Check the terminal/logs where vLLM is running for errors, and check `nvidia-smi` to see if the GPU is pegged or the process is gone. This is a host-side vLLM problem, not an OpenShell policy issue — a `NET:FAIL` means the connection never completed, unlike a `DENIED` line, which means policy blocked it |
| Agent's outbound connections are all denied for a destination you expect to be allowed | The attached provider's profile doesn't declare that `host`/`port` in `endpoints`, or doesn't declare the calling binary in `binaries` | Follow the profile update workflow in Step 11: export the profile, add the missing `endpoints`/`binaries` entry, lint, then `openshell profile update <profile-id> -f <file>`. For a one-off override scoped to a single sandbox instead of the shared profile, `openshell policy get <sandbox-name>` (without `--full`) and `openshell policy set <sandbox-name> --policy <file> --wait` work directly against that sandbox — see the next row for a common mistake with this path |
| `openshell policy set` fails with `unknown field 'Version'` | `openshell policy get --full` prepends a metadata header (including a `Version` field) that `policy set` does not accept | Use `openshell policy get <sandbox-name>` without `--full` to export only the policy YAML. If you already have output with the metadata header, strip every line before the first `---` (or before the first policy key) before passing it to `policy set` |
| "Permission denied" or Landlock errors inside the sandbox | Agent trying to access a path not in `read_only` or `read_write` filesystem policy | Pull the current policy and add the path to `read_write` (or `read_only` if read access is sufficient). Push the updated policy. Note: filesystem policy is static and requires sandbox recreation |
| vLLM OOM or very slow inference | Model too large for available memory or GPU contention | Free GPU memory (close other GPU workloads), choose a smaller model, or lower `--gpu-memory-utilization` / `--max-model-len`. Monitor with `nvidia-smi` |
| `openshell sandbox connect` or `sandbox exec` refuses with "canonical main process already finished" / "sandbox is not ready (phase: Completed)" | The sandbox's main process (Pi) already exited — a `Completed` sandbox is retained for inspection (`sandbox get`, `policy get`, `logs`) but cannot accept a new attachment or exec'd process | There is no way to resume interactive work in a `Completed` sandbox — delete and recreate it (Step 8). To avoid this, detach instead of quitting next time: press `Ctrl-P` then `Ctrl-Q` rather than typing `/quit`, which keeps Pi running so `sandbox connect` has something to reattach to |
| Policy push returns exit code 1 (validation failed) | Malformed YAML or invalid policy/profile fields | Check the YAML syntax. Common issues: paths not starting with `/`, `..` traversal in paths, `root` as `run_as_user`, endpoints missing required `host`/`port` fields, or a mapping entry landed outside its intended list because it was appended to the end of the file instead of inserted in place (see Step 11's `sed` insertion pattern) |
| Gateway service won't become healthy, and `journalctl` shows it stuck rather than crashing outright | The gateway's internal bootstrap (compute driver, database, or container runtime) is taking longer than expected, or Docker doesn't have enough resources | Check journal logs: `journalctl --user -u openshell-gateway --no-pager -n 50`. Check whether the container is still progressing: `docker ps --filter name=openshell`. Ensure Docker has enough memory and disk. If it does not recover, `systemctl --user stop openshell-gateway`, `docker rm -f <container>`, then `systemctl --user start openshell-gateway` |
| `openshell status` says "No gateway configured" | Gateway service never started or was disabled | Start the gateway: `systemctl --user start openshell-gateway`, then verify with `openshell status` (optionally `systemctl --user enable openshell-gateway` to auto-start on login). If the Docker container is unhealthy, run `systemctl --user stop openshell-gateway`, `docker rm -f <container>`, then `systemctl --user start openshell-gateway` |
| TLS / certificate errors when registering a remote gateway by raw LAN IP | The gateway's certificate may not validate against a bare IP address | Map a hostname to the hardware IP in `/etc/hosts` on the workstation you're connecting from, then register by that hostname instead of the raw IP: `openshell gateway add https://<hostname>:17670 --name <name>` (see Step 4) |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory issues even when within capacity. If that happens, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```
