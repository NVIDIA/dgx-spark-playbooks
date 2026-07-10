# Secure Long Running AI Agents with OpenShell on DGX Spark

> Run OpenClaw with local models in an NVIDIA OpenShell sandbox on DGX Spark

## Table of Contents

- [Overview](#overview)
  - [Notice & Disclaimers](#notice-disclaimers)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

OpenClaw is a local-first AI agent that runs on your machine, combining memory, file access, tool use, and community skills into a persistent assistant. Running it directly on your system means the agent can access your files, credentials, and network—creating real security risks.

**NVIDIA OpenShell** solves this problem. It is an open-source sandbox runtime that wraps the agent in kernel-level isolation with declarative YAML policies. OpenShell controls what the agent can read on disk, which network endpoints it can reach, and what privileges it has—without disabling the capabilities that make the agent useful.

By combining OpenClaw with OpenShell on DGX Spark, you get the full power of a local AI agent backed by 128GB of unified memory for large models, while enforcing explicit controls over filesystem access, network egress, and credential handling.

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

1. Data leakage – Any materials the agent accesses could be exposed, leaked, or stolen.

2. Malicious code execution – The agent or its connected tools could expose your system to malicious code or cyber-attacks.

3. Unintended actions – The agent might modify or delete files, send messages, or access services without explicit approval.

4. Prompt injection & manipulation – External inputs or connected content could hijack the agent's behavior in unexpected ways.

---

#### Security Best Practices

 No system is perfect, but these practices help keep your information and systems safe.

1. Isolate your environment – Run on a clean PC or isolated virtual machine. Only provision the specific data you want the agent to access.

2. Never use real accounts – Don't connect personal, confidential, or production accounts. Create dedicated test accounts with minimal permissions.

3. Vet your skills/plugins – Only enable skills from trusted sources that have been vetted by the community.

4. Lock down access – Ensure your OpenClaw UI or messaging channels aren't accessible over the network without proper authentication.

5. Restrict network access – Where feasible, limit the agent's internet connectivity.

6. Clean up after yourself – When you're done, remove OpenClaw and revoke all credentials, API keys, and account access you granted.

---

## What you'll accomplish

You will install the OpenShell CLI (`openshell`), deploy a gateway on your DGX Spark, and launch OpenClaw inside a sandboxed environment using the pre-built OpenClaw community sandbox. The sandbox enforces filesystem, network, and process isolation by default. You will also configure local inference routing so OpenClaw uses a model running on your Spark without needing external API keys.

## Popular use cases

- **Secure agent experimentation**: Test OpenClaw skills and integrations without exposing your main filesystem or credentials to the agent.
- **Private enterprise development**: Route all inference to a local model on DGX Spark. No data leaves the machine unless you explicitly allow it in the policy.
- **Auditable agent access**: Version-control the policy YAML alongside your project. Review exactly what the agent can reach before granting access.
- **Iterative policy tuning**: Monitor denied connections in real time with `openshell term`, then hot-reload updated policies without recreating the sandbox.

## What to know before starting

- Comfort with the Linux terminal and SSH
- Basic understanding of Docker (OpenShell runs a k3s cluster inside Docker)
- Familiarity with Docker and vLLM for local model serving
- Awareness of the security model: OpenShell reduces risk through isolation but cannot eliminate all risk. Review the [OpenShell documentation](https://pypi.org/project/openshell/) and [OpenClaw security guidance](https://docs.openclaw.ai/gateway/security).

## Prerequisites

**Hardware Requirements:**
- NVIDIA DGX Spark with 128GB unified memory
- Enough unified memory for the served model plus KV cache (the playbook serves `nvidia/Qwen3.6-35B-A3B-NVFP4` with vLLM at `--gpu-memory-utilization 0.4`)

**Software Requirements:**
- NVIDIA DGX OS (Ubuntu 24.04 base)
- Docker Desktop or Docker Engine running: `docker info`
- Python 3.12 or later: `python3 --version`
- `uv` package manager: `uv --version` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA Container Toolkit configured for Docker, plus a HuggingFace token to download the model
- Network access to download Python packages from PyPI and model weights from HuggingFace
- Have [NVIDIA Sync](https://build.nvidia.com/spark/connect-to-your-spark) installed and configured for your DGX Spark

## Time & risk

* **Estimated time:** 20–30 minutes (plus model download time, which depends on model size and network speed).

> [!CAUTION] **Risk level:** Medium
  * OpenShell sandboxes enforce kernel-level isolation, significantly reducing the risk compared to running OpenClaw directly on the host.
  * The sandbox default policy denies all outbound traffic not explicitly allowed. Misconfigured policies may block legitimate agent traffic; use `openshell logs` to diagnose.
  * Large model downloads may fail on unstable networks.
* **Rollback:** Delete the sandbox with `openshell sandbox delete <sandbox-name>`, stop the gateway with `openshell gateway stop`, and optionally destroy it with `openshell gateway destroy`. The vLLM container can be removed with `docker rm`/`docker rmi`.
* **Last Updated:** 06/12/2026
  * Switch local inference backend to vLLM (agent-ready Qwen3.6 35B recipe)

## Instructions

## Step 1. Confirm your environment

Verify the OS, GPU, Docker, and Python are available before installing anything.

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
```
Ensure [NVIDIA Sync](https://build.nvidia.com/spark/connect-to-your-spark/sync) is configured with a custom port: use "OpenClaw" as the Name and "18789" as the port.

Expected output should show Ubuntu 24.04 (DGX OS), a detected GPU, a Docker server version, and Python 3.12+.

## Step 2. Docker Configuration

First, verify that the local user has Docker permissions using the following command.
``` bash
docker ps
```
If you get a permission denied error (`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`), add your user to the system's Docker group. This will enable you to run Docker commands without requiring `sudo`. The command to do so is as follows:

``` bash
sudo usermod -aG docker $USER
newgrp docker
```
Note that you should reboot the Spark after adding the user to the group for this to take persistent effect across all terminal sessions.

Now that we have verified the user's Docker permission, we must configure Docker so that it can use the NVIDIA Container Runtime.
``` bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Run a sample workload to verify the setup:

``` bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## Step 3. Install OpenShell

Run the official installer, which installs both the `openshell` CLI and the `openshell-gateway` daemon as a systemd user service.

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

After installation, open a new shell (or `source ~/.bashrc`) so the `openshell` binary is on your `PATH`, then verify:

```bash
openshell --help
```

Expected output should show the `openshell` command tree with subcommands like `sandbox`, `provider`, and `inference`.

## Step 4. Verify the OpenShell gateway

As of OpenShell v0.0.37, the gateway is managed as a systemd user service installed automatically in Step 3. Confirm the service is running and the CLI can reach it:

```bash
systemctl --user status --no-pager openshell-gateway
openshell status
```

`openshell status` should report the gateway as **Connected**. If the service is not running, start it manually:

```bash
systemctl --user start openshell-gateway
```

To ensure the gateway persists after you log out:

```bash
sudo loginctl enable-linger $USER
```

To follow gateway logs in real time (streams continuously — press `Ctrl+C` to exit):

```bash
journalctl --user -u openshell-gateway -f
```

## Step 5. Serve a model with vLLM

Serve a model with **vLLM** for local inference. This playbook uses the agent-ready `nvidia/Qwen3.6-35B-A3B-NVFP4` recipe — the same one documented in the vLLM playbook's [Run Agent Ready Qwen3.6 35B Model with vLLM](https://build.nvidia.com/spark/vllm/agent-ready-qwen35b) tab.

Follow that tab to launch the server in a **separate terminal**. It serves `nvidia/Qwen3.6-35B-A3B-NVFP4` on an OpenAI-compatible API at port `8000`.

> [!IMPORTANT]
> The recipe binds `--host 0.0.0.0`, which is required here: the OpenShell gateway runs inside Docker and reaches the server over the Spark's IP address, not `localhost`. Keep the `--host 0.0.0.0` flag when you launch it.

Once the server reports `Application startup complete`, verify it is reachable on all interfaces:

```bash
curl http://0.0.0.0:8000/v1/models
```

Expected: a JSON `"data"` array listing `nvidia/Qwen3.6-35B-A3B-NVFP4`. If the request hangs, the model is likely still loading — wait for the startup line and retry.

## Step 6. Create an inference provider

We are going to create an OpenShell provider that points to your local vLLM server. This lets OpenShell route inference requests to your Spark-hosted model.

First, find the IP address of your DGX Spark:

```bash
hostname -I | awk '{print $1}'
```

Then create the provider, substituting your actual IP for `MACHINE_IP` in the command below. vLLM does not require an API key, so any non-empty placeholder works:

```bash
openshell provider create \
    --name local-vllm \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL=http://MACHINE_IP:8000/v1
```

> [!IMPORTANT]
> Do **not** use `localhost` or `127.0.0.1` here. The OpenShell gateway runs inside a Docker container, so it cannot reach the host via `localhost`. Replace MACHINE_IP with the machine's actual IP address.

Verify the provider was created:

```bash
openshell provider list
```

## Step 7. Configure inference routing

Point the `inference.local` endpoint (available inside every sandbox) at your vLLM model. The model name must match the handle served in Step 5:

```bash
openshell inference set \
    --provider local-vllm \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4
```

The output should confirm the route and show a validated endpoint URL, for example: `http://10.110.106.169:8000/v1/chat/completions (openai_chat_completions)`.

> [!NOTE]
> If you see `failed to verify inference endpoint` or `failed to connect` (for example because the gateway cannot reach the host IP from inside its container), add `--no-verify` to skip endpoint verification: `openshell inference set --provider local-vllm --model nvidia/Qwen3.6-35B-A3B-NVFP4 --no-verify`. Ensure the vLLM server is running and reachable on the Spark's IP (see Step 5).

Verify the configuration:

```bash
openshell inference get
```

Expected output should show `provider: local-vllm` and `model: nvidia/Qwen3.6-35B-A3B-NVFP4`.

## Step 8. Deploy OpenShell Sandbox

Create a sandbox using the pre-built OpenClaw community sandbox. This pulls the OpenClaw Dockerfile, the default policy, and startup scripts from the OpenShell Community catalog:

``` bash
openshell sandbox create \
  --keep \
  --tty \
  --forward 18789 \
  --name dgx-demo \
  --from openclaw \
  -- openclaw-start
```

> [!NOTE]
> Do not pass `--policy` with a local file path (e.g. `openclaw-policy.yaml`) when using `--from openclaw`. The policy is bundled with the community sandbox; a local file path can cause "file not found."

The `--keep` flag keeps the sandbox running after the initial process exits, so you can reconnect later. This is the default behavior. To terminate the sandbox when the initial process exits, use the `--no-keep` flag instead.

The CLI will:
1. Resolve `openclaw` against the community catalog
2. Pull and build the container image
3. Apply the bundled sandbox policy
4. Launch OpenClaw inside the sandbox

Set the sandbox name as an environment variable now so subsequent commands can reference it:

```bash
export SANDBOX_NAME=dgx-demo
```

## Step 9. Configure OpenClaw within OpenShell Sandbox

The sandbox container will spin up and the OpenClaw onboarding wizard will launch automatically in your terminal.

> [!IMPORTANT]
> The onboarding wizard is **fully interactive** — it requires arrow-key navigation and Enter to select options. It cannot be completed from a non-interactive session (e.g. a script or automation tool). You must run `openshell sandbox create` from a terminal with full TTY support.
>
> If the wizard did not complete during sandbox creation, reconnect to the sandbox to re-run it:
> ```bash
> openshell sandbox connect dgx-demo
> ```

The wizard prompt structure varies across openclaw versions — the community sandbox image and the latest npm release may differ. Use this version-agnostic walkthrough rather than following prompts by position:

1. **Accept terms** — select **Yes**.
2. **Quickstart vs Manual** — select **Quickstart**.
3. **Model/auth Provider** — find and select **Custom Provider**:
   - In openclaw 2026.6.5+: the short list shows `OpenAI / Anthropic / xAI / Google / More… / Skip`. Select **More…** to open the full provider submenu, then search for or scroll to **Custom Provider**.
   - In older openclaw (community image): **Custom Provider** appears directly in the list.
4. **API Base URL** — enter `https://inference.local/v1`.
5. **API key** — enter any non-empty value (e.g. `not-needed`; vLLM does not validate it).
6. **Model ID** — enter the model handle from Step 5: `nvidia/Qwen3.6-35B-A3B-NVFP4`.
7. **Remaining prompts** (Channel, Search provider, Skills, Hooks) — select **Skip** or **No** for each.

> [!NOTE]
> The community sandbox image (`ghcr.io/nvidia/openshell-community/sandboxes/openclaw:latest`) may be several versions behind the latest openclaw npm release. If the wizard behaves unexpectedly, check the version baked into the image: `docker exec $(docker ps --filter name=openshell-${SANDBOX_NAME} --format '{{.Names}}') openclaw --version`.

It might take 1-2 minutes to get through the final stages. Afterwards, you should see a URL with a token you can use to connect to the gateway. 

The expected output will be similar, but the token will be unique.
``` bash
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=9b4c9a9c9f6905131327ce55b6d044bd53e0ec423dd6189e
```

In order to verify the default policy enabled for your sandbox, please run the following command:

```bash
openshell sandbox get $SANDBOX_NAME
```
> [!NOTE]
> Step 8’s `--forward 18789` already sets up port forwarding from the OpenShell gateway to the sandbox. You do **not** need a manual `ssh` command with `openshell ssh-proxy` for the usual case.

To verify the forward is active, use the following command:

```bash
openshell forward list
```

You should see your sandbox name (e.g. `dgx-demo`) with port `18789`. If it is missing or `dead`, start it:

```bash
openshell forward start --background 18789 $SANDBOX_NAME
```

Path A: If you are using the Spark as the primary device, right-click on the URL in the UI section and select Open Link.

Path B: If you are using a laptop or workstation that is *not* on the Spark (e.g. you SSH into the Spark only): Install the OpenShell CLI on **that** machine.

> [!IMPORTANT]
> **SSH must work from this machine to the Spark before `gateway add`.** Run `ssh nvidia@<spark-ip>` (or your user/host) and confirm you get a shell without `Permission denied (publickey)`. If that fails, add your public key to the Spark: `ssh-copy-id nvidia@<spark-ip>` (from the same machine), or paste your `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`) into `~/.ssh/authorized_keys` on the Spark. OpenShell uses this SSH session to reach the remote Docker API and extract gateway TLS certificates. If you use a non-default key, pass `--ssh-key ~/.ssh/your_key` to `gateway add` (same as Step 4’s remote gateway note).

Register the Spark’s **already-running** gateway. Do **not** use `openshell gateway add user@ip` alone—that is parsed as a cloud URL and will not write `mtls/ca.crt`.

Per the [OpenShell gateway docs](https://docs.nvidia.com/openshell/latest/sandboxes/manage-gateways.html), register using **hostname `openshell`**, not the raw Spark IP, for HTTPS.

> [!WARNING]
> The gateway TLS certificate is valid for `openshell`, `localhost`, and `127.0.0.1` — **not** for your Spark’s LAN IP. If you use `https://10.x.x.x:8080` or `ssh://user@10.x.x.x:8080`, `openshell status` may fail with **certificate not valid for name "10.x.x.x"**.

**On your laptop/WSL**, map `openshell` to the Spark (once per machine):

```bash
## Replace with your Spark’s IP. Requires sudo on Linux/WSL.
echo "<spark-ip> openshell" | sudo tee -a /etc/hosts
## Example: echo "10.110.17.10 openshell" | sudo tee -a /etc/hosts
```

Then add the gateway (SSH target stays the real IP or hostname; HTTPS URL uses `openshell`):

```bash
openshell gateway add https://openshell:8080 --remote <user>@<spark-ip>
```

Example:

```bash
openshell gateway add https://openshell:8080 --remote nvidia@10.110.17.10
```

If you already registered with the IP and see the cert error, remove that entry and re-add:

```bash
openshell gateway destroy 
openshell gateway add https://openshell:8080 --remote nvidia@10.110.17.10
```

(Use `openshell gateway select` if the destroy name differs.)

Complete any browser or CLI prompts until the command finishes (do not Ctrl+C early). Then:

```bash
openshell status   # should show Connected, not TLS CA errors
openshell forward start --background 18789 dgx-demo
```

Then on the **laptop** browser open (use `#token=` so the UI receives the gateway token):

`http://127.0.0.1:18789/#token=<your-token>`

Use the token value from the OpenClaw wizard output on the Spark. Path B requires SSH from the laptop to the Spark so the CLI can reach the gateway on `:8080`.

**NVIDIA Sync:** Right-click the URL in the UI and select Copy Link. Connect to your Spark in Sync, open the OpenClaw entry, and paste the URL in the browser address bar.

From this page, you can now **Chat** with your OpenClaw agent within the protected confines of the runtime OpenShell provides.
## Step 10. Conduct Inference within Sandbox

#### Connecting to the Sandbox (Terminal)

Now that OpenClaw has been configured within the OpenShell protected runtime, you can connect directly into the sandbox environment via:

```bash
openshell sandbox connect $SANDBOX_NAME
```

Once loaded into the sandbox terminal, you can test connectivity to the vLLM model with this command:
``` bash
curl https://inference.local/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
```

## Step 11. Verify sandbox isolation

Open a second terminal and check the sandbox status and live logs:

```bash
openshell term
```

The terminal dashboard shows:
- **Sandbox status** — name, phase, image, providers, and port forwards
- **Live log stream** — outbound connections, policy decisions (`allow`, `deny`, `inspect_for_inference`), and inference interceptions

Verify that the OpenClaw agent can reach `inference.local` for model requests and that unauthorized outbound traffic is denied.

> [!TIP]
> Press `f` to follow live output, `s` to filter by source, and `q` to quit the terminal dashboard.

## Step 12. Reconnect to the sandbox

If you exit the sandbox session, reconnect at any time:

```bash
openshell sandbox connect $SANDBOX_NAME
```

> [!NOTE]
> `openshell sandbox connect` is interactive-only — it opens a terminal session inside the sandbox. There is no way to pass a command for non-interactive execution. Use `openshell sandbox upload`/`download` for file transfers, or `openshell sandbox ssh-config` for scripted SSH (see Step 14).

To transfer files in or out out of the sandbox, please use the following:

```bash
openshell sandbox upload $SANDBOX_NAME ./local-file /sandbox/destination
openshell sandbox download $SANDBOX_NAME /sandbox/file ./local-destination
```

## Step 13. Cleanup

Stop and remove the sandbox:

```bash
openshell sandbox delete $SANDBOX_NAME
```

Remove the inference provider you created in Step 6:

```bash
openshell provider delete local-vllm
```

To stop the gateway service (it will restart automatically on next login unless you disable it):

```bash
systemctl --user stop openshell-gateway
```

To disable the gateway service entirely and remove linger:

```bash
systemctl --user disable openshell-gateway
sudo loginctl disable-linger $USER
```

To also stop and remove the vLLM container and image:

```bash
docker rm $(docker ps -aq --filter ancestor=vllm/vllm-openai:nightly-aarch64)
docker rmi vllm/vllm-openai:nightly-aarch64
```

## Step 14. Next steps

- **Add more providers**: Attach GitHub tokens, GitLab tokens, or cloud API keys as providers with `openshell provider create`. When creating the sandbox, pass the provider name(s) with `--provider <name>` (e.g. `--provider my-github`) to inject those credentials into the sandbox securely.
- **Try other community sandboxes**: Run `openshell sandbox create --from base` or `--from sdg` for other pre-built environments.
- **Connect VS Code**: Use `openshell sandbox ssh-config <sandbox-name>` and append the output to `~/.ssh/config` to connect VS Code Remote-SSH directly into the sandbox.
- **Monitor and audit**: Use `openshell logs <sandbox-name> --tail` or `openshell term` to continuously monitor agent activity and policy decisions.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `openshell status` shows "Connection refused" after install | The `openshell-gateway` systemd user service failed to start, usually because the user's systemd session predates the docker group add | Run `systemctl --user status --no-pager openshell-gateway` to confirm the failure. Then run `systemctl --user start openshell-gateway`. If it still fails with a Docker socket auth error, apply a temporary ACL: `sudo setfacl -m u:$USER:rw /var/run/docker.sock` and restart: `systemctl --user restart openshell-gateway`. For a permanent fix, reboot the Spark (Step 2 recommends this after `usermod`) so the user session picks up the docker group. |
| `openshell status` shows gateway as unhealthy | The gateway service crashed | Run `journalctl --user -u openshell-gateway --no-pager -n 50` to see the error. Restart with `systemctl --user restart openshell-gateway`. If Docker socket access is denied, see the row above. |
| `openshell sandbox create --from openclaw` fails to build | Network issue pulling the community sandbox or Dockerfile build failure | Check internet connectivity. Retry the command. If the build fails on a specific package, check if the base image is compatible with your Docker version |
| Sandbox is in `Error` phase after creation | Policy validation failed or container startup crashed | Run `openshell logs <sandbox-name>` to see error details. Common causes: invalid policy YAML, missing provider credentials, or port conflicts |
| Agent cannot reach `inference.local` inside the sandbox | Inference routing not configured or provider unreachable | Run `openshell inference get` to verify the provider and model are set. Test the vLLM server from the host: `curl http://localhost:8000/v1/models`. Ensure the provider `OPENAI_BASE_URL` uses the Spark's IP address (not `localhost`), since the gateway runs inside Docker |
| 503 verification failed or timeout when gateway/sandbox accesses vLLM on the host | Provider URL points at `localhost`, or host firewall blocking port 8000 | The recipe already binds vLLM to all interfaces (`--host 0.0.0.0`). Confirm the provider `OPENAI_BASE_URL` uses the Spark's IP (from `hostname -I`) so the gateway container (e.g. on Docker network 172.17.x.x) can reach it. Allow port 8000 through the host firewall: `sudo ufw allow 8000/tcp comment 'vLLM for OpenShell Gateway'` (then `sudo ufw reload` if needed). |
| Agent's outbound connections are all denied | Default policy does not include the required endpoints | Monitor denials with `openshell logs <sandbox-name> --tail --source sandbox`. Pull the current policy with `openshell policy get <sandbox-name> --full`, add the needed host/port under `network_policies`, and push with `openshell policy set <sandbox-name> --policy <file> --wait` |
| "Permission denied" or Landlock errors inside the sandbox | Agent trying to access a path not in `read_only` or `read_write` filesystem policy | Pull the current policy and add the path to `read_write` (or `read_only` if read access is sufficient). Push the updated policy. Note: filesystem policy is static and requires sandbox recreation |
| vLLM OOM or very slow inference | Model too large for available memory or GPU contention | Free GPU memory (close other GPU workloads), or relaunch vLLM with a lower `--gpu-memory-utilization` / `--max-model-len` (or a smaller model handle). Monitor with `nvidia-smi` |
| `openshell sandbox connect` hangs or times out | Sandbox not in `Ready` phase | Run `openshell sandbox get <sandbox-name>` to check the phase. If stuck in `Provisioning`, wait or check logs. If in `Error`, delete and recreate the sandbox |
| Policy push returns exit code 1 (validation failed) | Malformed YAML or invalid policy fields | Check the YAML syntax. Common issues: paths not starting with `/`, `..` traversal in paths, `root` as `run_as_user`, or endpoints missing required `host`/`port` fields. Fix and re-push |
| `openshell status` says "No gateway configured" | The gateway service is not running or was never started | Run `systemctl --user start openshell-gateway` and then `openshell status`. If the service fails to start, check logs: `journalctl --user -u openshell-gateway --no-pager -n 50`. |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
