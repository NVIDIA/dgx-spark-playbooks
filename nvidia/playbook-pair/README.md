# Install and Use NVIDIA PAIR

> Run local AI requests through PAIR and route independent Ollama or LM Studio requests across compatible systems.


## Table of Contents

- [Overview](#overview)
- [Set Up the PAIR App](#set-up-the-pair-app)
  - [Windows](#windows)
  - [Debian or Ubuntu](#debian-or-ubuntu)
  - [macOS](#macos)
  - [Choose a request style](#choose-a-request-style)
  - [OpenAI-style request](#openai-style-request)
  - [Ollama-style request](#ollama-style-request)
  - [Port reference](#port-reference)
  - [Change a port](#change-a-port)
  - [Use an engine's command line](#use-an-engines-command-line)
- [Set Up PAIR with Terminal](#set-up-pair-with-terminal)
  - [Keep PAIR running after an SSH disconnect](#keep-pair-running-after-an-ssh-disconnect)
  - [Move around the terminal interface](#move-around-the-terminal-interface)
  - [Use the terminal interface tabs](#use-the-terminal-interface-tabs)
  - [Inspect errors and logs](#inspect-errors-and-logs)
  - [Change settings](#change-settings)
  - [Command locations and flags](#command-locations-and-flags)
  - [Terminal-interface limits](#terminal-interface-limits)
- [Troubleshooting](#troubleshooting)
  - [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA Personal AI Router (PAIR) connects computers on your local network and
presents Ollama-compatible and OpenAI-compatible proxy endpoints to applications
and agents. It automatically routes each independent AI inference request to an
eligible computer according to engine availability, model availability, and
current workload.
Run PAIR on one computer for local inference, or run it on several computers
to handle more requests.

PAIR is useful for workloads such as multi-agent applications that make several
requests at once. PAIR is designed to keep prompts and responses on your local
network when the application, model source, engine, and paired computers are
all local.

> [!IMPORTANT]
> PAIR sends each request to one system. It does not combine GPU memory, join
> GPUs into one larger GPU, or split a model or request across systems.

## What you'll accomplish

You'll install PAIR, prepare an engine and model, and send a test request
through PAIR's local endpoint.

You can also pair trusted systems and route independent requests to systems
that have the requested model.

## What to know before starting

**Required:**

- One compatible system on which to install PAIR.
- Ollama or LM Studio, plus a model, on at least one system that will serve
  requests.
- A trusted local network when you pair systems. The six-digit PIN is a
  short-lived setup code, not a long-term credential.

**Optional:**

- Two or more systems to route requests across a local cluster.
- A graphical desktop. Use the terminal interface on a headless system or over
  SSH.

Use one PAIR interface at a time. Do not run the desktop application and the
terminal interface on the same system. They start competing services and can
conflict over ports, engines, and settings.

PAIR accepts requests only from the local system. Install PAIR on the system
where your application runs. That system can use an engine on another cluster
node, so it does not need its own GPU or inference engine.

Read the [PAIR security policy](https://github.com/NVIDIA/Personal-AI-Router/blob/main/SECURITY.md)
before you use PAIR on a shared or untrusted network.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, operating system,
memory, and whether you can use it in a PAIR cluster.

| Hardware platform | OS | Memory | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **RTX Spark** | Windows 11 | Up to 64 GB Unified Memory  | ✅ (PAIR cluster) |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | ✅ (PAIR cluster) |
| **GeForce RTX** | Windows 11 or Linux | Depends on the GPU and model | ✅ (PAIR cluster) |
| **RTX PRO** | Windows 11 or Linux | Depends on the GPU and model | ✅ (PAIR cluster) |

NVIDIA RTX Spark, DGX Spark, GeForce RTX, and RTX PRO are the hardware platforms
covered by this playbook. GeForce RTX requires a 20 Series or newer GPU, and
RTX PRO requires a Turing or newer GPU.
PAIR can also pair compatible systems that run the supported operating systems
below.

| Support | Details |
| :---- | :---- |
| Operating systems | Windows 11, Linux, and macOS |
| Architectures | x64 on Windows 11, ARM 64 on Linux, Windows and macOS. |
| Installers | Windows `.exe`, Linux `.deb`, and macOS `.dmg`. Build from source for other Linux distributions. |
| Mixing systems | Windows, Linux, and macOS systems can pair with each other. |
| Inference engines | Ollama and LM Studio |

Check the [PAIR releases page](https://github.com/NVIDIA/Personal-AI-Router/releases)
for a package for your operating system and architecture.

## Prerequisites

**Hardware requirements**

- One compatible system for local inference.
- Two or more compatible systems on the same trusted local network for cluster
  routing.

**Software requirements**

- The appropriate PAIR package on each participating system.
- Ollama, LM Studio, or both, running on each system that will serve requests.
- A model downloaded on at least one system that will serve requests.

PAIR can run on a supported system even when an engine cannot. Each engine has
its own requirements for the operating system, GPU, and drivers. Each model
also needs enough memory to load. Check the engine documentation before you
expect a system to serve a model.

## Ancillary files

No extra files are required.

## Time & risk

- **Estimated time:** About 10 minutes, plus engine and model download time.
- **Risk level:** Low to medium. PAIR installs software, downloads models, and
  enables communication between trusted local systems.
- **Rollback:** Remove paired systems, uninstall PAIR, and remove engines or
  models you no longer need.
- **Last updated:** 08/17/2026.

For a graphical setup, open **Set Up the PAIR App**. For a headless or SSH
setup, open **Set up PAIR with Terminal**.

## Set Up the PAIR App

## Step 1. Install and open PAIR

Install PAIR on each desktop system that will join the cluster. Download the
appropriate package from the
[PAIR releases page](https://github.com/NVIDIA/Personal-AI-Router/releases).

### Windows

1. Download the Windows installer that matches the system's architecture.
2. Run the installer and approve the operating-system and firewall prompts.
3. Open **NVIDIA Personal AI Router** from the Start menu.

### Debian or Ubuntu

Open a terminal in the directory that contains the downloaded package, then
run:

```bash
sudo apt install "./NVPAIR-Setup-VERSION-ARCH.deb"
```

Replace `VERSION` and `ARCH` with the values in the downloaded filename. Then
open **NVIDIA Personal AI Router** from the desktop application menu.

### macOS

1. Open the downloaded `.dmg`.
2. Drag **NVIDIA Personal AI Router** to **Applications**.
3. Open PAIR from **Applications**.

PAIR opens when the installation is complete.

## Step 2. Complete first-run setup

When you first open PAIR, it shows the engines that it can install. Ollama is
selected by default when it is available for the platform.

1. Review the available engines.
2. Select the engines to install, or skip engines that you manage separately.
3. Finish setup and wait for the selected engine to report that it is running.

The first start can take longer while PAIR starts its background services. If
**Overview** still shows **Loading...** after one or two minutes, open
**Settings → Service**, then use the **Troubleshooting** tab.

You can return to engine settings by selecting a node in **Overview**. Use
**Settings → Cluster** to pair systems at any time.

## Step 3. Pair systems

Start PAIR on each system. Confirm that the systems are on the same trusted
local network.

1. On a system already in the cluster, select **Add node** in the top-right
   toolbar. You can also open **Settings → Cluster** and use **Available nodes
   to add**.
2. Select a discovered system. If PAIR does not find it, add the system by IP
   address.
3. PAIR shows a six-digit PIN and sends an invitation.
4. On the invited system, accept the **Cluster invitation** and enter the PIN.
5. Open **Settings → Cluster** and check that the peer appears under
   **Connected nodes**. You can also check the node list in **Overview**.
6. Repeat these steps from any cluster member to add more systems.

The peer appears as a connected node when pairing is complete.

## Step 4. Start an engine and add a model

Repeat these steps on every system that should serve the model:

1. Select the system in **Overview** to open its engine settings.
2. Install the engine if needed, then use its switch to start it.
3. Expand the engine and select **Add model**.
4. Download a model and wait for the download to finish.
5. Load the model if the engine requires a separate load step.

A system can serve a request when it is online, the engine is running, and the
requested model is available there. To route requests across several systems,
add the same model to each system that should serve it.

## Step 5. Copy the local endpoint

Applications connect to PAIR on the system where they run. They do not connect
directly to an engine or to the system that will run the request.

1. Select **Endpoints** in PAIR's top toolbar.
2. In the **API endpoints** window, find the engine that you prepared.
3. Copy its `http://127.0.0.1:<port>` URL.

If the window says **No engines are running**, return to step 4 and start an
engine. The endpoint appears when that engine is running on any system in the
cluster, including a remote system.

> [!IMPORTANT]
> Copy the URL from **Endpoints**. PAIR puts a proxy on the engine's usual port
> and moves the engine to the next available port.

## Step 6. Send a test request

Replace `<PAIR_BASE_URL>` with the endpoint from step 5. Replace
`<MODEL_NAME>` with the exact model name from step 4.

### Choose a request style

PAIR passes each request to the selected engine without rewriting it.

| Endpoint | OpenAI `/v1/chat/completions` | Ollama `/api/chat` |
| --- | --- | --- |
| Ollama | Works | Works |
| LM Studio | Works | Not available |

If you are unsure, use the OpenAI-style request. It works with either engine.

### OpenAI-style request

```bash
curl <PAIR_BASE_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_NAME>",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a short story about a dog who learns to skateboard."
      }
    ]
  }'
```

### Ollama-style request

Use this request only with an Ollama endpoint. The `-N` option shows the
response as it streams.

```bash
curl -N <PAIR_BASE_URL>/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_NAME>",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a short story about a dog who learns to skateboard."
      }
    ]
  }'
```

Open **Overview**, select the **Jobs** filter, and read **Ran on** or
**Running on** on the job card. A response and a job card show that PAIR routed
the request.

## Step 7. Connect an application

Set the application's base URL to the endpoint from step 5. Select a model
that you added in step 4.

PAIR accepts requests only from the system where it is running. The proxy uses
plaintext HTTP on loopback. A network request to an address such as
`http://some-node:11434` returns `403`.

Install PAIR on the system where you use the application. Join that system to
the cluster, then use its local endpoint. The system does not need a GPU or an
engine when another cluster system can serve the request.

PAIR does not provide a network-reachable inference endpoint. Configuring an
engine to listen on the network is outside PAIR and creates security exposure
that you must manage separately.

### Port reference

PAIR's proxy uses the port that an engine normally uses. PAIR moves the engine
to the next available port.

| Service | Default port |
| --- | --- |
| Ollama-compatible proxy | `11434` |
| Ollama engine behind PAIR | `11435` and upward |
| LM Studio / OpenAI-compatible proxy | `1234` |
| LM Studio engine behind PAIR | `1235` and upward |

If `OLLAMA_HOST` names a different local loopback address, PAIR also serves
that address when the port is free. PAIR does not use a remote or HTTPS
`OLLAMA_HOST` value.

PAIR uses these ports to communicate between cluster systems:

| Port | Purpose |
| --- | --- |
| `5353/udp` | Local-network discovery through mDNS |
| `14318` | Node hardware and model inventory |
| `14319` | Service-error synchronization |
| `14320` | Workload propagation |
| `14321` | Pairing and cluster membership |
| `14322` | Model list served to cluster peers |
| `14323` | Cluster-scoped remote engine control |

The Windows installer adds the firewall rules. If a Linux firewall is
restrictive, allow these ports between trusted cluster systems.

## Next steps

### Change a port

1. Open **Overview** and expand **Engine settings** on the local system's card.
2. Expand **Ports** for the engine.
3. Edit **Proxy**, **Server**, or both, then select **Apply ports**.

PAIR applies the change as one operation and restores the new values when it
next starts. You can change ports only on the local system. Remote system cards
show them as read-only. If you change a proxy port, update the application's
base URL. **Endpoints** always shows the current URL.

If another application uses a port that PAIR does not manage, choose a
different port in PAIR or stop the other application. Then restart the service
from **Settings → Service**.

### Use an engine's command line

PAIR-installed engines are normal installations, but their binaries are not in
`PATH` and their server ports differ from the usual defaults.

Ollama locations:

| Platform | Path |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Nvidia Corporation\Personal AI Router\engine-bin\ollama\ollama.exe` |
| Linux | `~/.config/Nvidia Corporation/Personal AI Router/engine-bin/ollama/bin/ollama` |
| macOS | `~/Library/Application Support/Nvidia Corporation/Personal AI Router/engine-bin/ollama/Ollama.app/Contents/Resources/ollama` |

Linux example:

```bash
ENGINE="$HOME/.config/Nvidia Corporation/Personal AI Router/engine-bin/ollama"
LD_LIBRARY_PATH="$ENGINE/lib/ollama" OLLAMA_HOST=127.0.0.1:11435 "$ENGINE/bin/ollama" list
```

Windows PowerShell example:

```powershell
$ollama = "$env:LOCALAPPDATA\Nvidia Corporation\Personal AI Router\engine-bin\ollama\ollama.exe"
$env:OLLAMA_HOST = "127.0.0.1:11435"
& $ollama list
```

LM Studio installs to its standard location:

```bash
~/.lmstudio/bin/lms status
```

On Windows, the executable is `%USERPROFILE%\.lmstudio\bin\lms.exe`.

Set `OLLAMA_HOST` to the engine's **Server** value when you want to inspect the
local engine. Otherwise, the CLI uses proxy port `11434` and returns the
cluster-wide view. The proxy shows what the cluster can serve. The server port
shows what is on that system.

## Set Up PAIR with Terminal

## Step 1. Start the terminal interface

Use the PAIR terminal interface on a system without a desktop environment or
over SSH. If the desktop application is running, quit it first. Do not run the
desktop application and terminal interface on the same system. They start
competing services and can conflict over ports, engines, and settings.

Install PAIR with the appropriate platform package, then run:

```bash
nvpair
```

`nvpair` starts the bundled terminal interface and its PAIR service process
tree. No other PAIR process needs to be running first.

The header changes to `broker ready v<version>` when the terminal interface
connects to the service. If it remains on `connecting to broker...`, open the
**Logs** tab and inspect the service output.

## Step 2. Pair this system

PAIR uses the same six-digit PIN exchange as the desktop application. Pair only
systems on a trusted network. The PIN is a short-lived setup code, not a
long-term credential.

To invite a discovered system:

1. Open **Nodes** (tab 3).
2. Select the system with `j` or `k`.
3. Press `i`.
4. Give the displayed PIN to the person operating the other system.

To invite a system by address:

1. Open **Cluster** (tab 7).
2. Press `i`.
3. Enter the other system's hostname or IP address. Use `host:port` when it
   does not use the default pairing port.
4. Press `enter`, then give the displayed PIN to the other operator.

Use address-based pairing when network discovery is unavailable, such as when
a network filters multicast.

To accept an invitation, open **Cluster**, wait for
`invite received from <name>`, press `a`, enter the PIN, and press `enter`.
Press `d` to decline an invitation.

The peer appears under **Members** when pairing is complete.

## Step 3. Start an engine and add a model

Open **Engines** (tab 6). Select an engine with `j` or `k`, then use these
keys:

| Key | Action |
| --- | --- |
| `i` | Install the selected engine. |
| `s` | Start it. |
| `x` | Stop it. |
| `r` | Restart it. |
| `u` | Uninstall it. |
| `p` | Download a model. |

After you press `p`, enter a model name such as `llama3.2`, then press
`enter`. Download progress appears on the status line.

A system can serve a request when it is online, the engine is running, and the
requested model is available there. Add the same model to several systems when
you want any of them to serve it.

## Step 4. Check the service and endpoint

Open **Overview** (tab 1). Confirm that the header shows
`broker ready v<version>`. An `ok` worker state means no crash was reported; it
does not prove that the worker is responding. `DOWN` means the broker reported
a crash.

Open **Proxies** (tab 4), press `g` until the engine that you prepared is
selected, and read its listening port. Use `http://127.0.0.1:<port>` as
`<PAIR_BASE_URL>`. The default port is `11434` for the Ollama-compatible proxy
and `1234` for the LM Studio / OpenAI-compatible proxy.

Ask the endpoint what the cluster can serve:

```bash
curl <PAIR_BASE_URL>/v1/models
```

The response lists the cluster's model inventory, not only the local system's
models. The terminal interface does not send inference requests. Configure a
compatible client with the local PAIR endpoint to send a request.

## Step 5. Check routing activity

Open **Workloads** (tab 5) to see live inference activity, including the
workload ID, model, engine, state, and age. The terminal interface does not
show which system served a workload.

Open **Proxies** (tab 4) to check each proxy's listening port and selected
system. `selected=auto` means that routing is automatic. Press `g` to switch
between engines, `enter` to pin the highlighted upstream, and `a` to restore
automatic routing.

Leave automatic routing enabled unless you are testing one system.

## Next steps

### Keep PAIR running after an SSH disconnect

If an SSH session closes, the terminal interface exits and the system stops
serving requests. Use a terminal multiplexer when PAIR must stay running:

```bash
tmux new -s pair
nvpair
```

Detach from tmux with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t pair
```

GNU Screen also works. Start it with `screen -S pair`, detach with `Ctrl-a d`,
and reattach with `screen -r pair`. tmux and Screen are not included with PAIR.

### Move around the terminal interface

The second line shows the numbered tabs. The footer shows the keys for the
current tab. If the screen shows `starting...`, enlarge the terminal window.

| Key | Action |
| --- | --- |
| `tab`, `l`, or `→` | Move to the next tab. |
| `shift+tab`, `h`, or `←` | Move to the previous tab. |
| `?` | Show or hide full help. |
| `q` or `ctrl+c` | Quit. |
| `j` / `k` or `↓` / `↑` | Move within a table. |
| `f` / `b` | Move forward or backward by one page. |
| `g` / `G` | Jump to the first or last row. |

When you enter a PIN, address, port, or model name, all keys go to that field.
Press `enter` to submit or `esc` to cancel.

### Use the terminal interface tabs

| # | Tab | What it shows |
| --- | --- | --- |
| 1 | **Overview** | Service uptime and version, plus an `ok` / `DOWN` worker table. |
| 2 | **Errors** | Active service errors by severity, age, system, and message. |
| 3 | **Nodes** | Discovered systems and their connection or cluster status. |
| 4 | **Proxies** | Compatible proxy ports, discovered upstreams, and selected systems. |
| 5 | **Workloads** | Live inference workload ID, model, engine, state, and age. |
| 6 | **Engines** | Local engine installation, running, health, and port state. |
| 7 | **Cluster** | Local-system identity, cluster membership, and pairing controls. |
| 8 | **Manual** | Systems added by address and their reachability. |
| 9 | **Settings** | Force ports, cluster auto-sync, cluster ID, and cluster name. |
| 10 | **Logs** | Service output and live log-level controls. |

### Inspect errors and logs

In **Errors** (tab 2), select an error and press `c` to clear it.

In **Logs** (tab 10), scroll with `j`, `k`, and the page keys. Change the
service log level with `d` for debug, `i` for info, `w` for warn, or `e` for
error. Check this tab first when a service does not start.

### Change settings

In **Settings** (tab 9), select a row with `j` or `k` and press `enter`.
Boolean settings change immediately. Text fields open for editing; press
`enter` to save or `esc` to cancel.

In **Proxies**, press `p` to change proxy ports. Engine ports are read-only in
the terminal interface. Use the desktop application to change them.

### Command locations and flags

The installer adds `nvpair` to `PATH`. It places the command in these
locations:

| Platform | Installed command location |
| --- | --- |
| Linux | `~/.local/bin/nvpair` |
| macOS | `/usr/local/bin/nvpair` when that directory is writable |
| Windows | A per-user `bin` directory added to `PATH` |

Open a new terminal after installation if the command is not found.

| Flag | Effect |
| --- | --- |
| `--broker-path <path>` | Use a PAIR service binary that is not beside the terminal-interface binary. |
| `--log-level <level>` | Set terminal-interface logging to `debug`, `info`, `warn`, or `error`. `NVPAIR_LOG_LEVEL` provides the same setting. |
| `--version` | Print the version and exit. |

The terminal interface writes its logs to stderr. Service logs appear on the
**Logs** tab. Press `q` to quit. Quitting also shuts down the PAIR services
cleanly.

### Terminal-interface limits

The terminal interface is an operations tool. It cannot:

- List or delete models. It can download a model but does not show a model
  inventory.
- Change an engine's port.
- Update an engine.
- Control engines on other cluster systems.
- Show which system served a workload.
- Send an inference request. Use a compatible client with the local endpoint.

## Troubleshooting

### Troubleshooting

Start with the symptom and then inspect **Settings → Service**, the desktop
error surface, or the terminal interface's **Errors** tab for more detail.

| Symptom | Usual meaning | What to do |
| --- | --- | --- |
| PAIR remains on **Loading...** | A background service did not start correctly. | Wait one or two minutes, then inspect **Settings → Service** and restart the affected service. |
| A node is not discovered | mDNS is blocked or unavailable. | Confirm both nodes are on the same trusted local network, allow `5353/udp`, or add the node by IP address. |
| Pairing stalls or fails | The invitation expired, the PIN is incorrect, or port `14321` is blocked. | Start a new invitation, enter the new PIN, and confirm cluster ports are reachable. |
| Connection refused | Nothing is listening at that address. | Copy the current URL from **Endpoints** and confirm the PAIR service is running. |
| `403` from another machine | PAIR endpoints accept loopback traffic only. | Run PAIR on the machine hosting the client and use its local endpoint. |
| `502` with `no active node` | No node is eligible for the request. | Start a compatible engine and make the requested model available on at least one online node. |
| Persistent `404` on inference | No eligible node has the requested model. | Verify the exact model name and prepare it on at least one node. |
| `400` or `422` | The request is malformed. | Correct its JSON, route, model name, or required fields; malformed requests are not retried. |
| A response arrives but **Jobs** is empty | Another process owns the expected proxy port. | Check **Endpoints** and **Settings → Service**, then change the PAIR port or stop the conflicting process. |
| Requests do not use every GPU | PAIR routes each request to one eligible node. | Send independent requests and verify **Ran on** for each job. PAIR does not split one request across GPUs. |
| Desktop and terminal behavior conflicts | Both PAIR interfaces are running on one system. | Stop one interface and use only the desktop application or terminal interface. |
