# Secure Long Running AI Agents with OpenShell on DGX Station

> Run OpenClaw with local models in an NVIDIA OpenShell sandbox on DGX Station

## Table of Contents

- [Overview](#overview)
  - [Notice & Disclaimers](#notice-disclaimers)
- [Instructions](#instructions)
  - [5a. Start the OpenAI-compatible server on port 8000](#5a-start-the-openai-compatible-server-on-port-8000)
  - [Access the dashboard](#access-the-dashboard)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

OpenClaw is a local-first AI agent that runs on your machine, combining memory, file access, tool use, and community skills into a persistent assistant. Running it directly on your system means the agent can access your files, credentials, and network—creating real security risks.

**NVIDIA OpenShell** solves this problem. It is an open-source sandbox runtime that wraps the agent in kernel-level isolation with declarative YAML policies. OpenShell controls what the agent can read on disk, which network endpoints it can reach, and what privileges it has—without disabling the capabilities that make the agent useful.

By combining OpenClaw with OpenShell on DGX Station (with NVIDIA GB300 GPUs), you get the full power of a local AI agent backed by GPU memory for local models, while enforcing explicit controls over filesystem access, network egress, and credential handling.

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

You will install the OpenShell CLI (`openshell`), deploy a gateway on your DGX Station, and launch OpenClaw inside a sandboxed environment using the pre-built OpenClaw community sandbox. The sandbox enforces filesystem, network, and process isolation by default. You will also configure local inference routing so OpenClaw uses a model running on your DGX Station via **vLLM** (NVIDIA NGC container on the host) without needing external API keys.

## Popular use cases

- **Secure agent experimentation**: Test OpenClaw skills and integrations without exposing your main filesystem or credentials to the agent.
- **Private enterprise development**: Route all inference to a local model on DGX Station. No data leaves the machine unless you explicitly allow it in the policy.
- **Auditable agent access**: Version-control the policy YAML alongside your project. Review exactly what the agent can reach before granting access.
- **Iterative policy tuning**: Monitor denied connections in real time with `openshell term`, then hot-reload updated policies without recreating the sandbox.

## What to know before starting

- Comfort with the Linux terminal and SSH
- Basic understanding of Docker (OpenShell runs a k3s cluster inside Docker)
- Familiarity with Docker and local LLM serving (vLLM in this playbook)
- Awareness of the security model: OpenShell reduces risk through isolation but cannot eliminate all risk. Review the [OpenShell overview](https://github.com/NVIDIA/OpenShell) and [OpenClaw security guidance](https://docs.openclaw.ai/gateway/security).

## Prerequisites

**Hardware Requirements:**
- NVIDIA DGX Station with GB300 GPU(s)
- Sufficient GPU memory for your chosen model: we recommend Nemotron 3 Super in NVFP4 (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) served by vLLM on GB300; smaller GPUs can use other Hugging Face models—check `nvidia-smi` and model requirements

**Software Requirements:**
- DGX OS or Ubuntu 24.04 (or compatible Linux)
- Docker Desktop or Docker Engine running: `docker info`
- Python 3.12 or later: `python3 --version`
- `uv` package manager: `uv --version` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA Container Toolkit (for GPU-enabled Docker): configured per `instructions.md`
- Network access to download Python packages from PyPI, the NGC vLLM image, and model weights from Hugging Face

## Time & risk

* **Estimated time:** 20–30 minutes (plus model download time, which depends on model size and network speed).
* **Risk level:** Low to Medium
  * OpenShell sandboxes enforce kernel-level isolation, significantly reducing the risk compared to running OpenClaw directly on the host.
  * The sandbox default policy denies all outbound traffic not explicitly allowed. Misconfigured policies may block legitimate agent traffic; use `openshell logs` to diagnose.
  * Large model downloads may fail on unstable networks.
* **Rollback:** Delete the sandbox with `openshell sandbox delete dgx-demo` (or your sandbox name), stop the gateway with `openshell gateway stop`, and optionally destroy it with `openshell gateway destroy`. Stop and remove the vLLM container and delete Hugging Face cache directories if you want to reclaim disk space (see `instructions.md` cleanup).
* **Last Updated:** 03/13/2026

## Instructions

## Step 1. Confirm your environment

Verify the OS, GPU, Docker, and Python are available before installing anything.

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
```
Expected output should show Ubuntu 24.04 (DGX OS), a detected GPU (e.g. NVIDIA GB300 on DGX Station), a Docker server version, and Python 3.12+. If you access the DGX Station remotely, ensure port 18789 is available for the OpenClaw dashboard.

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

## Step 5. Run vLLM with Nemotron 3 Super (recommended)

OpenShell's inference routing requires an OpenAI-compatible API endpoint on the host — any inference server compatible with the `/v1/chat/completions` protocol will work. This guide uses vLLM, which is the recommended path for DGX Station. It ships as an NVIDIA-maintained container, supports the NVFP4 quantization format used by Nemotron 3 Super, and exposes an OpenAI-compatible server out of the box.

Pull the vLLM container image first (this may take a few minutes):

```bash
docker pull nvcr.io/nvidia/vllm:26.03-py3
```

### 5a. Start the OpenAI-compatible server on port 8000

The OpenShell gateway must reach this service using the **host’s real IP address** (not `localhost` from inside other containers). Binding `--host 0.0.0.0` and publishing `-p 8000:8000` makes the API available on all interfaces.

The Nemotron weights may require a Hugging Face account and token. Create your own read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), keep it private (do not paste real tokens into shared docs, tickets, or git), then **export it in your shell** before `docker run` so the command below only references the variable:

```bash
export HF_TOKEN=your_actual_token_here
```

Replace `your_actual_token_here` with your real token value. If you do not need Hugging Face authentication for this model, skip the `export` and remove the `-e HF_TOKEN="$HF_TOKEN"` line from the `docker run` command.

We are going to use the `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` model as it fits in DGX Station VRAM with KV headroom at `--max-model-len 32768`.

**If this is a shared DGX Station**, verify port 8000 is free and a GPU has sufficient VRAM (~60 GB) before proceeding:

```bash
## Check port availability (no output = port is free)
ss -tlnp sport = :8000
## Check free VRAM per GPU index
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

If port 8000 is already in use, replace `-p 8000:8000` with an unused port (e.g. `-p 8001:8000`) and update the port in Steps 6 and 7 to match. If a specific GPU has more free VRAM, replace `--gpus all` with `--gpus '"device=<index>"'` (e.g. `--gpus '"device=1"'`).

> [!WARNING]
> The **`--trust-remote-code`** flag in the following `docker run` command allows execution of arbitrary code from the model repository. Only use this with trusted models.

```bash
docker run -d --name vllm-openshell \
  --runtime nvidia --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --restart unless-stopped \
  nvcr.io/nvidia/vllm:26.03-py3 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser nemotron_v3
```

> [!TIP]
> If you get a `docker: Error response from daemon:` error due to `Bind for 0.0.0.0:8000 failed: port is already allocated`, follow the steps below.
>
> Find what is using port 8000:
> ```bash
> sudo ss -tlnp sport = :8000
> ```
>
> If it's a Docker container, find its name:
> ```bash
> docker ps --filter "publish=8000"
> ```
>
> Stop and remove the container (replace `<name>` with the name from above):
> ```bash
> docker stop <name>
> docker rm <name>
> ```
>
> If it's a non-Docker process, kill it by PID (replace `<pid>` with the PID from the `ss` output):
> ```bash
> sudo kill <pid>
> ```

Watch logs until the server is ready (first start can take several minutes while weights load). Then, in a new terminal window, run:

```bash
docker logs -f vllm-openshell
```

Wait for logs to output `Application startup complete.`, then verify the API using:

```bash
curl -s http://localhost:8000/v1/models
```

You should see JSON listing `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.

Warm up with a short completion so CUDA graphs compile before OpenClaw validates the route (first request may take 30–90 seconds):

```bash
curl -s --max-time 120 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

## Step 6. Create an inference provider

Create an OpenShell provider that points at the vLLM OpenAI-compatible API on the host (`/v1` on port **8000**).

First, find the IP address of your DGX Station:

```bash
hostname -I | awk '{print $1}'
```

Then create the provider, substituting your actual IP for `10.110.106.169` in the command below:

```bash
openshell provider create \
    --name local-vllm \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL=http://10.110.106.169:8000/v1
```

> [!IMPORTANT]
> Do **not** use `localhost` or `127.0.0.1` here. The OpenShell gateway runs inside a Docker container, so it cannot reach the host via `localhost`. Use the machine's actual IP address.

Some Linux Docker setups can use `http://host.docker.internal:8000/v1` instead of the host IP; if your gateway resolves that hostname, it is equivalent.

Verify the provider was created:

```bash
openshell provider list
```

## Step 7. Configure inference routing

Point the `inference.local` endpoint (available inside every sandbox) at vLLM. The **model id must exactly match** what vLLM is serving — confirm it first:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Use the `id` value returned (e.g. `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) in the command below. If you used a different model in Step 5, substitute that id here:

```bash
openshell inference set \
    --provider local-vllm \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

The output should confirm the route and show a validated endpoint URL, for example: `http://10.110.106.169:8000/v1/chat/completions (openai_chat_completions)`.

> [!NOTE]
> If you see `failed to verify inference endpoint` or `failed to connect`, ensure vLLM is healthy (`docker logs vllm-openshell`) and you completed at least one chat completion so cold-start compilation has finished. You can add `--no-verify` to skip verification: `openshell inference set --provider local-vllm --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --no-verify`.

Verify the configuration:

```bash
openshell inference get
```

Expected output should show `provider: local-vllm` and `model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (or whichever model you configured in Step 5).

## Step 8. Deploy OpenShell Sandbox

Create a sandbox using the pre-built OpenClaw community sandbox. This pulls the OpenClaw Dockerfile, the default policy, and startup scripts from the OpenShell Community catalog:

``` bash
openshell sandbox create \
  --keep \
  --forward 18789 \
  --name dgx-demo \
  --from openclaw \
  -- openclaw-start
```

> [!NOTE]
> Do not pass `--policy` with a local file path (e.g. `openclaw-policy.yaml`) when using `--from openclaw`. The policy is bundled with the community sandbox; a local file path can cause "file not found."

The `--keep` flag keeps the sandbox running after the initial process exits, so you can reconnect later. This is the default behavior. To terminate the sandbox when the initial process exits, use the `--no-keep` flag instead.

> [!NOTE]
> The sandbox name is displayed in the creation output. You can also set it explicitly with `--name <your-name>`. To find it later, run `openshell sandbox list`.

The CLI will:
1. Resolve `openclaw` against the community catalog
2. Pull and build the container image
3. Apply the bundled sandbox policy
4. Launch OpenClaw inside the sandbox

After the sandbox is created, activate the port forward so the OpenClaw dashboard is reachable on the host. The `--forward 18789` flag registers the intent but does not activate the forward automatically — run this to start it:

```bash
openshell forward start -d 18789 dgx-demo
```

Verify the forward is active:

```bash
openshell forward list
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

> [!NOTE]
> **If `openshell sandbox get dgx-demo` shows `Phase: Unspecified`**, this is expected when the OpenClaw onboarding wizard has not yet been completed interactively. The phase does not advance to `Ready` until the wizard finishes. The sandbox container itself may be healthy even while the phase shows `Unspecified` — confirm by checking the supervisor logs:
> ```bash
> docker logs $(docker ps --filter name=openshell-dgx-demo --format '{{.Names}}') --tail 20
> ```
> A healthy sandbox will show `OpenShell Sandbox Supervisor success` and `Applying Landlock filesystem sandbox` in the output. If you are provisioning over SSH without a TTY, drive the wizard manually after creation:
> ```bash
> openshell sandbox connect dgx-demo
> ```

Use the arrow keys and Enter key to interact with the installation.
- If you understand and agree, use the arrow key of your keyboard to select 'Yes' and press the Enter key.
- Quickstart vs Manual: select Quickstart and press the Enter key.
- Model/auth Provider: Select **Custom Provider**, the second-to-last option.
- API Base URL: update to https://inference.local/v1
- How do you want to provide this API key?: Paste API key for now.
- API key: enter `not-needed` (or any placeholder; vLLM is not checking the key unless you enabled API-key auth in the server).
- Endpoint compatibility: select **OpenAI-compatible** and press Enter.
- Model ID: enter the same id you set in Step 7 (e.g. `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
	- The first request may take up to a minute while vLLM compiles; ensure the container from Step 5 is already serving (`docker logs vllm-openshell`).
- Endpoint ID: leave the default value.
- Alias: enter the same model name (this is optional).
- Channel: Select **Skip for now**.
- Search provider: Select **Skip for now**.
- Skills: Select **No** for now.
- Enable hooks: Select **No** for now (using the space bar) and press Enter.


It might take 1-2 minutes to get through the final stages. Afterwards, you should see a URL with a token you can use to connect to the gateway. 

The expected output will be similar, but the token will be unique.
``` bash
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=9b4c9a9c9f6905131327ce55b6d044bd53e0ec423dd6189e
```

In order to verify the default policy enabled for your sandbox, please run the following command:

```bash
openshell sandbox get dgx-demo
```

### Access the dashboard
**Accessing the dashboard from DGX station as the primary device:** right-click on the URL in the UI section and select Open Link.

**Accessing the dashboard from the host or a remote system:** The dashboard URL (e.g. `http://127.0.0.1:18789/?token=...`) is inside the sandbox, so the host does not forward port 18789 by default. To reach it from your host or another machine, use SSH local port forwarding. From a machine that can reach the OpenShell gateway, run (replace gateway URL, sandbox-id, token, and gateway-name with values from your environment):

```bash
ssh -o ProxyCommand='openshell ssh-proxy --gateway https://127.0.0.1:17670/connect/ssh --sandbox-id <sandbox-id> --token <token> --gateway-name openshell' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -N -L 18789:127.0.0.1:18789 sandbox
```

Then open `http://127.0.0.1:18789/?token=<your-token>` in your local browser.

To access from another machine, use the SSH tunnel described above, or open the dashboard URL in your browser (e.g. after port forwarding or from the DGX Station's local browser).

From this page, you can now **Chat** with your OpenClaw agent within the protected confines of the runtime OpenShell provides.
## Step 10. Conduct Inference within Sandbox

#### Connecting to the Sandbox (Terminal)

Now that OpenClaw has been configured within the OpenShell protected runtime, you can connect directly into the sandbox environment via:

```bash
openshell sandbox connect dgx-demo
```

Once loaded into the sandbox terminal, you can test connectivity to vLLM via `inference.local` with this command:
``` bash
curl https://inference.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
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
openshell sandbox connect dgx-demo
```

> [!NOTE]
> `openshell sandbox connect` is interactive-only — it opens a terminal session inside the sandbox. There is no way to pass a command for non-interactive execution. Use `openshell sandbox upload`/`download` for file transfers, or use the SSH proxy for scripted access (see Step 9).

To transfer files in or out (replace `dgx-demo` with your sandbox name if you used a different one):

```bash
openshell sandbox upload dgx-demo ./local-file /sandbox/destination
openshell sandbox download dgx-demo /sandbox/file ./local-destination
```

## Step 13. Cleanup

Stop and remove the sandbox (use the name you gave it, e.g. `dgx-demo`):

```bash
openshell sandbox delete dgx-demo
```

To stop the gateway service (it will restart automatically on next login unless you disable it):

```bash
systemctl --user stop openshell-gateway
```

To disable the gateway service entirely and remove linger so user services no longer start on boot:

```bash
systemctl --user disable openshell-gateway
sudo loginctl disable-linger $USER
```

Remove the inference provider you created in Step 6:

```bash
openshell provider delete local-vllm
```

Stop and remove the vLLM container started in Step 5:

```bash
docker stop vllm-openshell
docker rm vllm-openshell
```

(Optional) Remove the container image to free disk:

```bash
docker rmi nvcr.io/nvidia/vllm:26.03-py3
```

## Step 14. Next steps

- **Add more providers**: Attach GitHub tokens, GitLab tokens, or cloud API keys as providers with `openshell provider create`. When creating the sandbox, pass the provider name(s) with `--provider <name>` (e.g. `--provider my-github`) to inject those credentials into the sandbox securely.
- **Try other community sandboxes**: Run `openshell sandbox create --from base` or `--from sdg` for other pre-built environments.
- **Connect VS Code**: Use `openshell sandbox ssh-config <sandbox-name>` and append the output to `~/.ssh/config` to connect VS Code Remote-SSH directly into the sandbox.
- **Monitor and audit**: Use `openshell logs <sandbox-name> --tail` or `openshell term` to continuously monitor agent activity and policy decisions.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `openshell gateway start` fails with "connection refused" or Docker errors | Docker is not running | Start Docker with `sudo systemctl start docker` or launch Docker Desktop, then retry `openshell gateway start` |
| `openshell status` shows gateway as unhealthy | Gateway container crashed or failed to initialize | Run `openshell gateway destroy` and then `openshell gateway start` to recreate it. Check Docker logs with `docker ps -a` and `docker logs <container-id>` for details |
| `openshell sandbox create --from openclaw` fails to build | Network issue pulling the community sandbox or Dockerfile build failure | Check internet connectivity. Retry the command. If the build fails on a specific package, check if the base image is compatible with your Docker version |
| Sandbox is in `Error` phase after creation | Policy validation failed or container startup crashed | Run `openshell logs <sandbox-name>` to see error details. Common causes: invalid policy YAML, missing provider credentials, or port conflicts |
| Agent cannot reach `inference.local` inside the sandbox | Inference routing not configured or provider unreachable | Run `openshell inference get` to verify the provider and model are set. From the host, test vLLM: `curl -s http://localhost:8000/v1/models`. The provider base URL must use the host’s real IP (not `127.0.0.1`/`localhost`) so the gateway container can reach vLLM (see `instructions.md` Step 6). |
| 503 verification failed or timeout when the gateway validates vLLM | vLLM not listening on all interfaces, firewall blocking port 8000, model still loading, or first-request CUDA graph compile | Ensure the vLLM server was started with `--host 0.0.0.0` and port `8000` mapped (see Step 5). Warm up with a chat completion request before `openshell inference set`. Allow port 8000 if you use a host firewall: `sudo ufw allow 8000/tcp comment 'vLLM for OpenShell Gateway'` (then `sudo ufw reload` if needed). For very large models, try `openshell inference set ... --no-verify` after confirming vLLM works from the host. |
| Agent's outbound connections are all denied | Default policy does not include the required endpoints | Monitor denials with `openshell logs <sandbox-name> --tail --source sandbox`. Pull the current policy with `openshell policy get <sandbox-name> --full`, add the needed host/port under `network_policies`, and push with `openshell policy set <sandbox-name> --policy <file> --wait` |
| "Permission denied" or Landlock errors inside the sandbox | Agent trying to access a path not in `read_only` or `read_write` filesystem policy | Pull the current policy and add the path to `read_write` (or `read_only` if read access is sufficient). Push the updated policy. Note: filesystem policy is static and requires sandbox recreation |
| vLLM OOM or very slow inference | Model too large for available VRAM, `--max-model-len` too high, or GPU contention | Free GPU memory (close other GPU workloads), use a smaller Hugging Face model or quantized variant, or lower `--max-model-len`. Check `docker logs` for the vLLM container. Monitor with `nvidia-smi` |
| `openshell sandbox connect` hangs or times out | Sandbox not in `Ready` phase | Run `openshell sandbox get <sandbox-name>` to check the phase. If stuck in `Provisioning`, wait or check logs. If in `Error`, delete and recreate the sandbox |
| Policy push returns exit code 1 (validation failed) | Malformed YAML or invalid policy fields | Check the YAML syntax. Common issues: paths not starting with `/`, `..` traversal in paths, `root` as `run_as_user`, or endpoints missing required `host`/`port` fields. Fix and re-push |
| `openshell gateway start` fails with "K8s namespace not ready" / timed out waiting for namespace | The k3s cluster inside the Docker container takes longer to bootstrap than the CLI timeout allows. The internal components (TLS secrets, Helm chart, namespace creation) may need extra time, especially on first run when images are pulled inside the container. | First, check whether the container is still running and progressing: `docker ps --filter name=openshell` (look for `health: starting`). Inspect k3s state inside the container: `docker exec <container> sh -c "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get ns"` and `kubectl get pods -A`. If pods are in `ContainerCreating` and TLS secrets are missing (`navigator-server-tls`, `openshell-server-tls`), the cluster is still bootstrapping — wait a few minutes and run `openshell status` again. If it does not recover, destroy with `openshell gateway destroy` (and `docker rm -f <container>` if needed) and retry `openshell gateway start`. Ensure Docker has enough resources (memory and disk) for the k3s cluster. |
| `openshell status` says "No gateway configured" even though the Docker container is running | The `gateway start` command failed or timed out before it could save the gateway configuration to the local config store | The container may still be healthy — check with `docker ps --filter name=openshell`. If the container is running and healthy, try `openshell gateway start` again (it should detect the existing container). If the container is unhealthy or stuck, remove it with `docker rm -f <container>` and then `openshell gateway destroy` followed by `openshell gateway start`. |
