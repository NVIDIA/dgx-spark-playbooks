# Secure AI Agents with OpenShell

> Isolate OpenClaw with kernel-level policies and route inference to a local model

## Table of Contents

- [Overview](#overview)
  - [Notice & Disclaimers](#notice-disclaimers)
- [Instructions](#instructions)
  - [Access the dashboard](#access-the-dashboard)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendations by hardware platform](#recommendations-by-hardware-platform)
  - [Before you serve](#before-you-serve)
  - [DGX Spark](#dgx-spark)
  - [DGX Station](#dgx-station)
  - [OpenShell-specific requirements](#openshell-specific-requirements)
  - [Next steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

OpenClaw is a local-first AI agent that runs on your machine, combining memory, file access, tool use, and community skills into a persistent assistant. Running it directly on your system means the agent can access your files, credentials, and network—creating real security risks.

**NVIDIA OpenShell** solves this problem. It is an open-source sandbox runtime that wraps the agent in kernel-level isolation with declarative YAML policies. OpenShell controls what the agent can read on disk, which network endpoints it can reach, and what privileges it has—without disabling the capabilities that make the agent useful.

By combining OpenClaw with OpenShell on your hardware platform, you get the full power of a local AI agent backed by local model serving, while enforcing explicit controls over filesystem access, network egress, and credential handling.

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
4. **Lock down access** – Ensure your OpenClaw UI or messaging channels aren't accessible over the network without proper authentication.
5. **Restrict network access** – Where feasible, limit the agent's internet connectivity.
6. **Clean up after yourself** – When you're done, remove OpenClaw and revoke all credentials, API keys, and account access you granted.

---

## What you'll accomplish

You will install the OpenShell CLI (`openshell`), deploy a gateway on your hardware platform, and launch OpenClaw inside a sandboxed environment using the pre-built OpenClaw community sandbox. The sandbox enforces filesystem, network, and process isolation by default. You will also configure local inference routing so OpenClaw uses a model running on your hardware without needing external API keys.

## Popular use cases

- **Secure agent experimentation**: Test OpenClaw skills and integrations without exposing your main filesystem or credentials to the agent.
- **Private enterprise development**: Route all inference to a local model on your hardware. No data leaves the machine unless you explicitly allow it in the policy.
- **Auditable agent access**: Version-control the policy YAML alongside your project. Review exactly what the agent can reach before granting access.
- **Iterative policy tuning**: Monitor denied connections in real time with `openshell term`, then hot-reload updated policies without recreating the sandbox.

## What to know before starting

**Required:**

- Comfort with the Linux terminal and SSH
- Basic understanding of Docker (OpenShell runs a k3s cluster inside Docker)
- Familiarity with local LLM serving (this playbook uses vLLM)

**Optional:**

- Awareness of the security model: OpenShell reduces risk through isolation but cannot eliminate all risk. Review the [OpenShell documentation](https://docs.nvidia.com/openshell/latest/) and [OpenClaw security guidance](https://docs.openclaw.ai/gateway/security).

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. The same OpenShell + OpenClaw workflow applies across supported hardware platforms. Model recommendations differ by platform — see the **Agent-ready Models** tab.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |

> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for your chosen agent-ready model (see the **Agent-ready Models** tab)

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
  - OpenShell sandboxes enforce kernel-level isolation, significantly reducing the risk compared to running OpenClaw directly on the host.
  - The sandbox default policy denies all outbound traffic not explicitly allowed. Misconfigured policies may block legitimate agent traffic; use `openshell logs` to diagnose.
  - Large model downloads may fail on unstable networks.
- **Rollback:** Delete the sandbox with `openshell sandbox delete <sandbox-name>`, stop the gateway, and remove the vLLM container if you started one. See cleanup in the **Instructions** tab.
- **Last Updated:** 07/27/2026
  - Gateway install path updated for the systemd user service; troubleshooting expanded for gateway connectivity and sandbox onboarding

## Instructions

## Step 1. Confirm your environment

Verify the OS, GPU, Docker, and Python are available before installing anything.

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
```

Expected output should show Ubuntu 24.04 (or compatible DGX OS), a detected GPU, a Docker server version, and Python 3.12+. If you access the hardware remotely, ensure port `18789` is available for the OpenClaw dashboard.

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

Install OpenShell with the official installer, which installs the `openshell` CLI and registers the `openshell-gateway` systemd user service:

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

Open a new shell (or `source ~/.bashrc`) so `openshell` is on your `PATH`, then verify:

```bash
openshell --help
```

Expected output should show the `openshell` command tree with subcommands like `gateway`, `sandbox`, `provider`, and `inference`.

> [!NOTE]
> Alternative install: `uv venv openshell-env && source openshell-env/bin/activate && uv pip install openshell`. If you use this path, activate the virtual environment in every new terminal before running `openshell` commands. The systemd unit is only provided by the official installer — see the [OpenShell gateway docs](https://docs.nvidia.com/openshell/sandboxes/manage-gateways) for gateway startup on the uv path.

## Step 4. Verify the OpenShell gateway

The official installer manages the gateway as a systemd user service. Confirm the service is running and the CLI can reach it:

```bash
systemctl --user status --no-pager openshell-gateway
openshell status
```

`openshell status` should report the gateway as **Connected**. If the service is not running, start it:

```bash
systemctl --user start openshell-gateway
systemctl --user status --no-pager openshell-gateway
openshell status
```

To keep the gateway available after you log out:

```bash
sudo loginctl enable-linger $USER
```

Follow gateway logs in real time (press `Ctrl+C` to exit):

```bash
journalctl --user -u openshell-gateway -f
```

The first run may take a few minutes while Docker pulls images and the internal k3s cluster bootstraps.

> [!TIP]
> To manage a gateway on remote hardware from a separate workstation, ensure passwordless SSH works first, then use `openshell gateway start --remote <username>@<hostname>` (or register an existing gateway per the [OpenShell gateway docs](https://docs.nvidia.com/openshell/latest/sandboxes/manage-gateways.html)).

## Step 5. Serve a model with vLLM

Serve an OpenAI-compatible API for local inference. Use the recommended model for your hardware platform from the **Agent-ready Models** tab.

Launch the matching recipe in a **separate terminal**, keeping `--host 0.0.0.0` and port `8000` so the OpenShell gateway (inside Docker) can reach the server.

Once the server reports `Application startup complete`, verify the localhost endpoint:

```bash
curl -sf http://localhost:8000/v1/models
```

Expected: a JSON `"data"` array listing your model handle. Note the exact `id` — you will reuse it in Steps 6–7 and the OpenClaw wizard.

Then verify the hardware-IP endpoint. This check is required because the OpenShell gateway reaches the server from a container rather than the host network namespace:

```bash
export HARDWARE_IP="$(hostname -I | awk '{print $1}')"
test -n "$HARDWARE_IP"
curl -sf "http://${HARDWARE_IP}:8000/v1/models"
```

> [!IMPORTANT]
> Do not bind the server to `localhost` only. The OpenShell gateway cannot reach host services via `127.0.0.1` from inside its container network.

## Step 6. Create an inference provider

Create an OpenShell provider that points to your local vLLM server.

Set the IP address of your hardware:

```bash
export HARDWARE_IP="$(hostname -I | awk '{print $1}')"
test -n "$HARDWARE_IP"
```

Create the provider with that address. vLLM does not require an API key, so any non-empty placeholder works:

```bash
openshell provider create \
    --name local-vllm \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL="http://${HARDWARE_IP}:8000/v1"
```

> [!IMPORTANT]
> Do **not** use `localhost` or `127.0.0.1` here. The OpenShell gateway runs inside Docker and cannot reach host services via those addresses. Use the machine's actual IP from `hostname -I`.

Verify:

```bash
openshell provider list
```

## Step 7. Configure inference routing

Point the `inference.local` endpoint (available inside every sandbox) at your model. The model name must match the handle served in Step 5:

```bash
openshell inference set \
    --provider local-vllm \
    --model <MODEL_HANDLE>
```

Replace `<MODEL_HANDLE>` with the served model name from the **Agent-ready Models** tab:
- **DGX Spark:** `nvidia/Qwen3.6-35B-A3B-NVFP4`
- **DGX Station:** `nemotron-ultra` (the `--served-model-name` set in the launch command, not the HuggingFace handle)

> [!NOTE]
> If you see `failed to verify inference endpoint` or `failed to connect`, confirm the server is healthy and warm up with one chat completion request. You can add `--no-verify` to skip endpoint verification after confirming reachability from the host.

Verify:

```bash
openshell inference get
```

Expected output should show `provider: local-vllm` and your chosen `model`.

## Step 8. Deploy the OpenShell sandbox

Create a sandbox using the pre-built OpenClaw community sandbox:

```bash
export SANDBOX_NAME=openshell-demo

openshell sandbox create \
  --keep \
  --tty \
  --forward 18789 \
  --name "$SANDBOX_NAME" \
  --from openclaw \
  -- openclaw-start
```

> [!NOTE]
> Do not pass `--policy` with a local file path when using `--from openclaw`. The policy is bundled with the community sandbox; a local file path can cause "file not found."

The `--keep` flag keeps the sandbox running after the initial process exits. To terminate when the initial process exits, use `--no-keep` instead.

The CLI will:

1. Resolve `openclaw` against the community catalog
2. Pull and build the container image
3. Apply the bundled sandbox policy
4. Launch OpenClaw inside the sandbox

> [!IMPORTANT]
> Once the container is ready, the OpenClaw onboarding wizard will launch automatically in this terminal. Proceed to Step 9 to complete it before continuing.

## Step 9. Configure OpenClaw within the sandbox

> [!IMPORTANT]
> The onboarding wizard is **fully interactive** — it requires arrow-key navigation and Enter to select options. It cannot be completed from a non-interactive session. You must run `openshell sandbox create` from a terminal with full TTY support.
>
> If the wizard did not complete during sandbox creation, reconnect:
> ```bash
> openshell sandbox connect "$SANDBOX_NAME"
> ```

> [!NOTE]
> If `openshell sandbox get` shows `Phase: Unspecified`, that is expected until the interactive wizard finishes. The sandbox container can still be healthy while the phase shows `Unspecified`. Confirm with supervisor logs if needed:
> ```bash
> docker logs $(docker ps --filter name=openshell-"$SANDBOX_NAME" --format '{{.Names}}') --tail 20
> ```
> Look for `OpenShell Sandbox Supervisor success` and `Applying Landlock filesystem sandbox`.

Use the arrow keys and Enter to complete onboarding:

- If you understand and agree, select **Yes** and press Enter.
- Quickstart vs Manual: select **Quickstart**.
- Model/auth Provider: select **Custom Provider**.
- API Base URL: `https://inference.local/v1`
- How do you want to provide this API key?: **Paste API key for now**.
- API key: enter any non-empty placeholder (for example `vllm` or `not-needed`).
- Endpoint compatibility: select **OpenAI-compatible**.
- Model ID: enter the same handle you set in Step 7.
- Endpoint ID: leave the default.
- Alias: optional; you can reuse the model name.
- Channel: **Skip for now**.
- Search provider: **Skip for now**.
- Skills: **No** for now.
- Enable hooks: **Skip for now** / **No**, then press Enter.

After 1–2 minutes you should see a URL with a token:

```bash
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=<unique-token>
```

Verify the sandbox:

```bash
openshell sandbox get "$SANDBOX_NAME"
```

### Access the dashboard

**On the hardware itself:** open the UI URL from the wizard output in a local browser (right-click → Open Link when available).

**From a remote workstation:** activate port forwarding:

```bash
openshell forward start --background 18789 "$SANDBOX_NAME"
openshell forward list
```

You should see your sandbox name with port `18789`. Then open:

`http://127.0.0.1:18789/#token=<your-token>`

If you manage the gateway from a remote machine, register it with hostname `openshell` (not the raw LAN IP) so TLS certificate validation succeeds — see the [OpenShell gateway docs](https://docs.nvidia.com/openshell/latest/sandboxes/manage-gateways.html). Map `openshell` to the hardware IP in `/etc/hosts` on the workstation, then:

```bash
openshell gateway add https://openshell:8080 --remote <user>@<hardware-ip>
openshell forward start --background 18789 "$SANDBOX_NAME"
```

If the dashboard URL is only reachable inside the sandbox and the host forward is not active, you can also tunnel with the OpenShell SSH proxy (replace sandbox id, token, and gateway URL from your environment):

```bash
ssh -o ProxyCommand='openshell ssh-proxy --gateway https://127.0.0.1:17670/connect/ssh --sandbox-id <sandbox-id> --token <token> --gateway-name openshell' \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  -N -L 18789:127.0.0.1:18789 sandbox
```

Then open `http://127.0.0.1:18789/?token=<your-token>` in your local browser.

From the dashboard you can **Chat** with your OpenClaw agent inside the OpenShell sandbox.

## Step 10. Test inference inside the sandbox

Connect to the sandbox terminal:

```bash
openshell sandbox connect "$SANDBOX_NAME"
```

Test connectivity to the local model:

```bash
curl https://inference.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_HANDLE>",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Replace `<MODEL_HANDLE>` with the same handle from Step 7.

## Step 11. Verify sandbox isolation

Open a second terminal and check live status and logs:

```bash
openshell term
```

The terminal dashboard shows:

- **Sandbox status** — name, phase, image, providers, and port forwards
- **Live log stream** — outbound connections, policy decisions (`allow`, `deny`, `inspect_for_inference`), and inference interceptions

Verify that the agent can reach `inference.local` and that unauthorized outbound traffic is denied.

> [!TIP]
> Press `f` to follow live output, `s` to filter by source, and `q` to quit.

## Step 12. Reconnect and transfer files

Reconnect at any time:

```bash
openshell sandbox connect "$SANDBOX_NAME"
```

> [!NOTE]
> `openshell sandbox connect` is interactive-only. Use upload/download for file transfers, or `openshell sandbox ssh-config` for scripted SSH.

```bash
openshell sandbox upload "$SANDBOX_NAME" ./local-file /sandbox/destination
openshell sandbox download "$SANDBOX_NAME" /sandbox/file ./local-destination
```

## Step 13. Cleanup

Run gateway-dependent cleanup first, while the gateway is still reachable:

```bash
openshell sandbox delete "$SANDBOX_NAME"
openshell provider delete local-vllm
```

If you use the systemd user service (official installer):

```bash
systemctl --user stop openshell-gateway
systemctl --user disable openshell-gateway
sudo loginctl disable-linger $USER
```

If you started the gateway with the CLI instead:

```bash
openshell gateway stop
```

> [!WARNING]
> The following command permanently removes the gateway cluster and all its data.

```bash
openshell gateway destroy
```

Stop and remove the vLLM container if you started one for this playbook (replace the name/image filter to match your launch):

```bash
docker rm -f vllm-server 2>/dev/null || true
```

## Step 14. Next steps

- **Add more providers**: Attach GitHub tokens, GitLab tokens, or cloud API keys with `openshell provider create`, then pass `--provider <name>` when creating a sandbox.
- **Try other community sandboxes**: `openshell sandbox create --from base` or `--from sdg`.
- **Connect VS Code**: Use `openshell sandbox ssh-config <sandbox-name>` and append the output to `~/.ssh/config`.
- **Monitor and audit**: Use `openshell logs <sandbox-name> --tail` or `openshell term` to monitor agent activity and policy decisions.

## Agent-ready Models

## Agent-ready models

Agent-ready models are tuned for **agentic workloads** — tool calling, reasoning traces, and long multi-turn sessions. Pick the recommended model for your hardware platform, serve it with an OpenAI-compatible API (this playbook uses vLLM), then continue with the **Instructions** tab to wire OpenShell inference routing and launch the OpenClaw sandbox.

### Recommendations by hardware platform

| Hardware platform | Recommended agent-ready model | HuggingFace handle |
| ----------------- | ----------------------------- | ------------------ |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (NVFP4) | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| **DGX Station** | NVIDIA Nemotron 3 Ultra (NVFP4) | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` |

### Before you serve

Complete Docker setup in the **Instructions** tab (Steps 1–2), then export a Hugging Face token if your model requires it:

```bash
export HF_TOKEN=your_actual_token_here
```

### DGX Spark

Launch via the vLLM playbook's [Agent-ready Models](https://build.nvidia.com/playbooks/vllm/agent-ready-models) tab, keeping `--host 0.0.0.0` and port `8000`.

### DGX Station

Follow the [Nemotron 3 Ultra Station Deployment Guide](https://github.com/NVIDIA-NeMo/nemotron/tree/main/usage-cookbook/Nemotron-3-Ultra/StationDeploymentGuide), which covers GB300 device selection, CPU offloading, and all required vLLM flags.

> [!IMPORTANT]
> The deployment guide uses `--served-model-name nemotron-ultra`. Use `nemotron-ultra` (not the HuggingFace handle) when running `openshell inference set` in Step 7 and in the OpenClaw onboarding wizard.

Once the server reports `Application startup complete`, confirm the API is up:

```bash
curl -s http://0.0.0.0:8000/v1/models
```

### OpenShell-specific requirements

| Requirement | Why it matters |
| ----------- | -------------- |
| `--host 0.0.0.0` | The OpenShell gateway runs inside Docker and cannot reach a server bound only to `localhost`. |
| Host IP in the provider URL | Create the OpenShell provider with `http://<Machine_IP>:8000/v1`, not `localhost` or `127.0.0.1`. |
| Matching model id | The OpenClaw wizard Model ID must match the handle served by vLLM and configured in `openshell inference set`. |

### Next steps

Once the server is up and `curl http://0.0.0.0:8000/v1/models` returns your model handle, return to the **Instructions** tab at **Step 5** to create the inference provider and deploy the sandbox.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `openshell status` shows "Connection refused" | The `openshell-gateway` systemd user service is not running, or Docker socket is not accessible from the user service | Start it: `systemctl --user start openshell-gateway`. Check logs: `journalctl --user -u openshell-gateway --no-pager -n 50`. If the service cannot reach Docker, fix socket access with `sudo setfacl -m u:$USER:rw /var/run/docker.sock`, then restart the service |
| Gateway service fails to start after installation | Docker is not running | Start Docker: `sudo systemctl start docker`. Then restart the OpenShell gateway service: `systemctl --user restart openshell-gateway` |
| `openshell status` shows gateway as unhealthy | Gateway service or container crashed / failed to initialize | Run `systemctl --user restart openshell-gateway` and inspect `journalctl --user -u openshell-gateway --no-pager -n 50`. Check Docker with `docker ps -a` and `docker logs <container-id>` |
| `openshell sandbox create --from openclaw` fails to build | Network issue pulling the community sandbox or Dockerfile build failure | Check internet connectivity. Retry the command. If the build fails on a specific package, check if the base image is compatible with your Docker version |
| Sandbox is in `Error` phase after creation | Policy validation failed or container startup crashed | Run `openshell logs <sandbox-name>` to see error details. Common causes: invalid policy YAML, missing provider credentials, or port conflicts |
| Agent cannot reach `inference.local` inside the sandbox | Inference routing not configured or provider unreachable | Run `openshell inference get` to verify the provider and model are set. From the host, test the server: `curl -s http://localhost:8000/v1/models`. Ensure the provider `OPENAI_BASE_URL` uses the hardware IP address (not `localhost`), since the gateway runs inside Docker |
| 503 verification failed or timeout when gateway/sandbox accesses vLLM on the host | Provider URL points at `localhost`, firewall blocking port 8000, model still loading, or first-request compile | Confirm the server was started with `--host 0.0.0.0`. Confirm the provider URL uses the hardware IP from `hostname -I`. Warm up with a chat completion before `openshell inference set`. Allow port 8000 through the host firewall if needed: `sudo ufw allow 8000/tcp` (then `sudo ufw reload`). Use `--no-verify` only after confirming the host API works |
| Agent's outbound connections are all denied | Default policy does not include the required endpoints | Monitor denials with `openshell logs <sandbox-name> --tail --source sandbox`. Pull the current policy with `openshell policy get <sandbox-name>` (without `--full`), add the needed host/port under `network_policies`, and push with `openshell policy set <sandbox-name> --policy <file> --wait`. See the `unknown field 'Version'` row below if you used `--full`. |
| `openshell policy set` fails with `unknown field 'Version'` | `openshell policy get --full` prepends a metadata header (including a `Version` field) that `policy set` does not accept | Use `openshell policy get <sandbox-name>` without `--full` to export only the policy YAML. If you already have output with the metadata header, strip every line before the first `---` (or before the first policy key) before passing it to `policy set`. |
| "Permission denied" or Landlock errors inside the sandbox | Agent trying to access a path not in `read_only` or `read_write` filesystem policy | Pull the current policy and add the path to `read_write` (or `read_only` if read access is sufficient). Push the updated policy. Note: filesystem policy is static and requires sandbox recreation |
| vLLM OOM or very slow inference | Model too large for available memory or GPU contention | Free GPU memory (close other GPU workloads), choose a smaller model, or lower `--gpu-memory-utilization` / `--max-model-len`. Monitor with `nvidia-smi` |
| `openshell sandbox connect` hangs or times out | Sandbox not in `Ready` phase | Run `openshell sandbox get <sandbox-name>` to check the phase. If stuck in `Provisioning`, wait or check logs. If in `Error`, delete and recreate the sandbox. If phase is `Unspecified` during onboarding, complete the interactive wizard with `openshell sandbox connect <sandbox-name>` |
| Policy push returns exit code 1 (validation failed) | Malformed YAML or invalid policy fields | Check the YAML syntax. Common issues: paths not starting with `/`, `..` traversal in paths, `root` as `run_as_user`, or endpoints missing required `host`/`port` fields. Fix and re-push |
| Gateway service starts but fails to become healthy — logs show "K8s namespace not ready" or namespace timeout | The k3s cluster inside the Docker container takes longer to bootstrap than expected | Check journal logs: `journalctl --user -u openshell-gateway --no-pager -n 50`. Check whether the container is still progressing: `docker ps --filter name=openshell`. Inspect k3s state: `docker exec <container> sh -c "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get ns"` and `kubectl get pods -A`. If pods are still creating, wait and retry `openshell status`. If it does not recover, stop the service with `systemctl --user stop openshell-gateway`, run `openshell gateway destroy`, then restart: `systemctl --user start openshell-gateway`. Ensure Docker has enough memory and disk |
| `openshell status` says "No gateway configured" | Gateway service never started or was disabled | Start the gateway: `systemctl --user start openshell-gateway`, then verify with `openshell status` (optionally `systemctl --user enable openshell-gateway` to auto-start on login). If the Docker container is unhealthy, run `systemctl --user stop openshell-gateway`, `docker rm -f <container>`, then `systemctl --user start openshell-gateway` |
| TLS / certificate errors when adding a remote gateway by LAN IP | Gateway certificate is valid for `openshell`, `localhost`, and `127.0.0.1` — not the LAN IP | Map `openshell` to the hardware IP in `/etc/hosts`, then register with `openshell gateway add https://openshell:8080 --remote <user>@<hardware-ip>` |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory issues even when within capacity. If that happens, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```
