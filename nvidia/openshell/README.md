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
- Familiarity with Ollama for local model serving
- Awareness of the security model: OpenShell reduces risk through isolation but cannot eliminate all risk. Review the [OpenShell documentation](https://pypi.org/project/openshell/) and [OpenClaw security guidance](https://docs.openclaw.ai/gateway/security).

## Prerequisites

**Hardware Requirements:**
- NVIDIA DGX Spark with 128GB unified memory
- At least 70GB available memory for a large local model (e.g., gpt-oss:120b at ~65GB plus overhead), or 25GB+ for a smaller model (e.g., gpt-oss-20b)

**Software Requirements:**
- NVIDIA DGX OS (Ubuntu 24.04 base)
- Docker Desktop or Docker Engine running: `docker info`
- Python 3.12 or later: `python3 --version`
- `uv` package manager: `uv --version` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Ollama 0.17.0 or newer (latest recommended for gpt-oss MXFP4 support): `ollama --version`
- Network access to download Python packages from PyPI and model weights from Ollama
- Have [NVIDIA Sync](https://build.nvidia.com/spark/connect-to-your-spark) installed and configured for your DGX Spark

## Time & risk

* **Estimated time:** 20–30 minutes (plus model download time, which depends on model size and network speed).

> [!CAUTION] **Risk level:** Medium
  * OpenShell sandboxes enforce kernel-level isolation, significantly reducing the risk compared to running OpenClaw directly on the host.
  * The sandbox default policy denies all outbound traffic not explicitly allowed. Misconfigured policies may block legitimate agent traffic; use `openshell logs` to diagnose.
  * Large model downloads may fail on unstable networks.
* **Rollback:** Delete the sandbox with `openshell sandbox delete <sandbox-name>`, stop the gateway with `openshell gateway stop`, and optionally destroy it with `openshell gateway destroy`. Ollama models can be removed with `ollama rm <model>`.
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

## Step 3. Install the OpenShell CLI

Create a virtual environment and install the `openshell` CLI.

```bash
cd ~
uv venv openshell-env && source openshell-env/bin/activate
uv pip install openshell 
openshell --help
```

If you don't have `uv` installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Expected output should show the `openshell` command tree with subcommands like `gateway`, `sandbox`, `provider`, and `inference`.

## Step 4. Deploy the OpenShell gateway on DGX Spark

The gateway is the control plane that manages sandboxes. Since you are running directly on the Spark, it deploys locally inside Docker.

```bash
openshell gateway start
openshell status
```

`openshell status` should report the gateway as **Connected**. The first run may take a few minutes while Docker pulls the required images and the internal k3s cluster bootstraps.

> [!NOTE]
> Remote gateway deployment requires passwordless SSH access. Ensure your SSH public key is added to `~/.ssh/authorized_keys` on the DGX Spark before using the `--remote` flag.

> [!TIP]
> If you want to manage the Spark gateway from a separate workstation, run `openshell gateway start --remote <username>@<spark-ssid>.local` from that workstation instead. All subsequent commands will route through the SSH tunnel.

## Step 5. Install Ollama and pull a model

Install Ollama (if not already present) and download a model for local inference.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

DGX Spark's 128GB memory can run large models:

| GPU memory available | Suggested model          | Model size | Notes |
|---------------------|---------------------------|-----------|-------|
| 25–48 GB            | nemotron-3-nano           | ~24GB     | Lower latency, good for interactive use |
| 48–80 GB            | gpt-oss:120b              | ~65GB     | Good balance of quality and speed |
| 128 GB              | nemotron-3-super:120b     | ~86GB     | Best quality on DGX Spark |

Verify Ollama is running (it auto-starts as a service after installation). If not, start it manually:

```bash
ollama serve &
```

Configure Ollama to listen on all interfaces so the OpenShell gateway container can reach it. Create a systemd override:

```bash
mkdir -p /etc/systemd/system/ollama.service.d/
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

Add these lines to the file (create the file if it does not exist):

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Save and exit, then reload and restart Ollama:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Verify Ollama is listening on all interfaces:

```bash
ss -tlnp | grep 11434
```

You should see `*:11434` in the output. If it only shows `127.0.0.1:11434`, confirm the override file contents and that you ran `systemctl daemon-reload` before restarting.

Next, run a model from Ollama (adjust the model name to match your choice from [the Ollama model library](https://ollama.com/library)). The `ollama run` command will pull the model automatically if it is not already present. Running the model here ensures it is loaded and ready when you use it with OpenClaw, reducing the chance of timeouts later. Example for nemotron-3-super:

```bash
ollama run nemotron-3-super:120b
```

Verify the model is available:

```bash
ollama list
```

## Step 6. Create an inference provider

We are going to create an OpenShell provider that points to your local Ollama server. This lets OpenShell route inference requests to your Spark-hosted model.

First, find the IP address of your DGX Spark:

```bash
hostname -I | awk '{print $1}'
```

Then create the provider, replacing `{Machine_IP}` with the IP address from the command above (e.g. `10.110.106.169`):

```bash
openshell provider create \
    --name local-ollama \
    --type openai \
    --credential OPENAI_API_KEY=not-needed \
    --config OPENAI_BASE_URL=http://{Machine_IP}:11434/v1
```

> [!IMPORTANT]
> Do **not** use `localhost` or `127.0.0.1` here. The OpenShell gateway runs inside a Docker container, so it cannot reach the host via `localhost`. Use the machine's actual IP address.

Verify the provider was created:

```bash
openshell provider list
```

## Step 7. Configure inference routing

Point the `inference.local` endpoint (available inside every sandbox) at your Ollama model. Replace the model name with your choice from Step 5:

```bash
openshell inference set \
    --provider local-ollama \
    --model nemotron-3-super:120b
```

The output should confirm the route and show a validated endpoint URL, for example: `http://10.110.106.169:11434/v1/chat/completions (openai_chat_completions)`.

> [!NOTE]
> If you see `failed to verify inference endpoint` or `failed to connect` (for example because the gateway cannot reach the host IP from inside its container), add `--no-verify` to skip endpoint verification: `openshell inference set --provider local-ollama --model nemotron-3-super:120b --no-verify`. Ensure Ollama is running and listening on all interfaces (see Step 5).

Verify the configuration:

```bash
openshell inference get
```

Expected output should show `provider: local-ollama` and `model: nemotron-3-super:120b` (or whichever model you chose).

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

The CLI will:
1. Resolve `openclaw` against the community catalog
2. Pull and build the container image
3. Apply the bundled sandbox policy
4. Launch OpenClaw inside the sandbox

## Step 9. Configure OpenClaw within OpenShell Sandbox

The sandbox container will spin up and the OpenClaw onboarding wizard will launch automatically in your terminal.

> [!IMPORTANT]
> The onboarding wizard is **fully interactive** — it requires arrow-key navigation and Enter to select options. It cannot be completed from a non-interactive session (e.g. a script or automation tool). You must run `openshell sandbox create` from a terminal with full TTY support.
>
> If the wizard did not complete during sandbox creation, reconnect to the sandbox to re-run it:
> ```bash
> openshell sandbox connect dgx-demo
> ```

Use the arrow keys and Enter key to interact with the installation.
- If you understand and agree, use the arrow key of your keyboard to select 'Yes' and press the Enter key.
- Quickstart vs Manual: select Quickstart and press the Enter key.
- Model/auth Provider: Select **Custom Provider**, the second-to-last option.
- API Base URL: update to https://inference.local/v1
- How do you want to provide this API key?: Paste API key for now.
- API key: please enter "ollama".
- Endpoint compatibility: select **OpenAI-compatible** and press Enter.
- Model ID: enter the model name you chose in Step 5 (e.g. `nemotron-3-super:120b`).
	- This may take 1-2 minutes as the Ollama model is spun up in the background.
- Endpoint ID: leave the default value.
- Alias: enter the same model name (this is optional).
- Channel: Select **Skip for now**.
- Skills: Select **No** for now.
- Enable hooks: Select **No** for now and press Enter.

It might take 1-2 minutes to get through the final stages. Afterwards, you should see a URL with a token you can use to connect to the gateway. 

The expected output will be similar, but the token will be unique.
``` bash
OpenClaw gateway starting in background.
  Logs: /tmp/gateway.log
  UI:   http://127.0.0.1:18789/?token=9b4c9a9c9f6905131327ce55b6d044bd53e0ec423dd6189e
```

Now that we have configured OpenClaw within the OpenShell sandbox, let's set the name of our openshell sandbox as an environment variable. This will make future commands easier to run. Note that the name of the sandbox was set in the `openshell sandbox create` command in Step 8 using the `--name` flag.

```bash
export SANDBOX_NAME=dgx-demo
```

In order to verify the default policy enabled for your sandbox, please run the following command:

```bash
openshell sandbox get $SANDBOX_NAME
```

If you are using the Spark as the primary device, right-click on the URL in the UI section and select Open Link.

**Accessing the dashboard from the host or a remote system:** The dashboard URL (e.g. `http://127.0.0.1:18789/?token=...`) is inside the sandbox, so the host does not forward port 18789 by default. To reach it from your host or another machine, use SSH local port forwarding. From a machine that can reach the OpenShell gateway, run (replace gateway URL, sandbox-id, token, and gateway-name with values from your environment):

```bash
ssh -o ProxyCommand='/usr/local/bin/openshell ssh-proxy --gateway https://127.0.0.1:8080/connect/ssh --sandbox-id <sandbox-id> --token <token> --gateway-name openshell' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -N -L 18789:127.0.0.1:18789 sandbox
```

Then open `http://127.0.0.1:18789/?token=<your-token>` in your local browser.

Otherwise, if you are using NVIDIA Sync, right-click on the URL listed in the UI and select Copy Link. Next, connect to your Spark and select the OpenClaw entry. When your web browser opens the tab for OpenClaw, paste the URL in the navigation bar and press the Enter key.

From this page, you can now **Chat** with your OpenClaw agent within the protected confines of the runtime OpenShell provides.
## Step 10. Conduct Inference within Sandbox

#### Connecting to the Sandbox (Terminal)

Now that OpenClaw has been configured within the OpenShell protected runtime, you can connect directly into the sandbox environment via:

```bash
openshell sandbox connect $SANDBOX_NAME
```

Once loaded into the sandbox terminal, you can test connectivity to the Ollama model with this command:
``` bash
curl https://inference.local/v1/responses \
          -H "Content-Type: application/json" \
          -d '{
        "instructions": "You are a helpful assistant.",
        "input": "Hello!"
      }'
```

## Step 11. Verify sandbox isolation

Open a second terminal and check the sandbox status and live logs:

```bash
source ~/openshell-env/bin/activate
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
> `openshell sandbox connect` is interactive-only — it opens a terminal session inside the sandbox. There is no way to pass a command for non-interactive execution. Use `openshell sandbox upload`/`download` for file transfers, or use the SSH proxy for scripted access (see Step 9).

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
openshell provider delete local-ollama
```

Stop the gateway (preserves state for later):

```bash
openshell gateway stop
```

> [!WARNING]
> The following command permanently removes the gateway cluster and all its data.

```bash
openshell gateway destroy
```

To also remove the Ollama model:

```bash
ollama rm nemotron-3-super:120b
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
| Agent cannot reach `inference.local` inside the sandbox | Inference routing not configured or provider unreachable | Run `openshell inference get` to verify the provider and model are set. Test Ollama is accessible from the host: `curl http://localhost:11434/api/tags`. Ensure the provider URL uses `host.docker.internal` instead of `localhost` |
| 503 verification failed or timeout when gateway/sandbox accesses Ollama on the host | Ollama bound only to localhost, or host firewall blocking port 11434 | Make Ollama listen on all interfaces so the gateway container (e.g. on Docker network 172.17.x.x) can reach it: `OLLAMA_HOST=0.0.0.0 ollama serve &`. Allow port 11434 through the host firewall: `sudo ufw allow 11434/tcp comment 'Ollama for OpenShell Gateway'` (then `sudo ufw reload` if needed). |
| Agent's outbound connections are all denied | Default policy does not include the required endpoints | Monitor denials with `openshell logs <sandbox-name> --tail --source sandbox`. Pull the current policy with `openshell policy get <sandbox-name> --full`, add the needed host/port under `network_policies`, and push with `openshell policy set <sandbox-name> --policy <file> --wait` |
| "Permission denied" or Landlock errors inside the sandbox | Agent trying to access a path not in `read_only` or `read_write` filesystem policy | Pull the current policy and add the path to `read_write` (or `read_only` if read access is sufficient). Push the updated policy. Note: filesystem policy is static and requires sandbox recreation |
| Ollama OOM or very slow inference | Model too large for available memory or GPU contention | Free GPU memory (close other GPU workloads), try a smaller model (e.g., `gpt-oss:20b`), or reduce context length. Monitor with `nvidia-smi` |
| `openshell sandbox connect` hangs or times out | Sandbox not in `Ready` phase | Run `openshell sandbox get <sandbox-name>` to check the phase. If stuck in `Provisioning`, wait or check logs. If in `Error`, delete and recreate the sandbox |
| Policy push returns exit code 1 (validation failed) | Malformed YAML or invalid policy fields | Check the YAML syntax. Common issues: paths not starting with `/`, `..` traversal in paths, `root` as `run_as_user`, or endpoints missing required `host`/`port` fields. Fix and re-push |
| `openshell gateway start` fails with "K8s namespace not ready" / timed out waiting for namespace | The k3s cluster inside the Docker container takes longer to bootstrap than the CLI timeout allows. The internal components (TLS secrets, Helm chart, namespace creation) may need extra time, especially on first run when images are pulled inside the container. | First, check whether the container is still running and progressing: `docker ps --filter name=openshell` (look for `health: starting`). Inspect k3s state inside the container: `docker exec <container> sh -c "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get ns"` and `kubectl get pods -A`. If pods are in `ContainerCreating` and TLS secrets are missing (`navigator-server-tls`, `openshell-server-tls`), the cluster is still bootstrapping — wait a few minutes and run `openshell status` again. If it does not recover, destroy with `openshell gateway destroy` (and `docker rm -f <container>` if needed) and retry `openshell gateway start`. Ensure Docker has enough resources (memory and disk) for the k3s cluster. |
| `openshell status` says "No gateway configured" even though the Docker container is running | The `gateway start` command failed or timed out before it could save the gateway configuration to the local config store | The container may still be healthy — check with `docker ps --filter name=openshell`. If the container is running and healthy, try `openshell gateway start` again (it should detect the existing container). If the container is unhealthy or stuck, remove it with `docker rm -f <container>` and then `openshell gateway destroy` followed by `openshell gateway start`. |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
