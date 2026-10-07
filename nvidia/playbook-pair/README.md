# Install and Use NVIDIA PAIR with llama.cpp

> Route independent local AI requests across paired systems using llama.cpp, Ollama, or LM Studio

## Table of Contents

- [Overview](#overview)
  - [Why memory matters](#why-memory-matters)
  - [Choose an engine and model](#choose-an-engine-and-model)
- [Set Up the PAIR App](#set-up-the-pair-app)
  - [Windows](#windows)
  - [Debian or Ubuntu](#debian-or-ubuntu)
  - [macOS](#macos)
  - [llama.cpp example](#llamacpp-example)
  - [Choose a request style](#choose-a-request-style)
  - [Windows PowerShell: llama.cpp example](#windows-powershell-llamacpp-example)
  - [Linux or macOS: OpenAI-style request](#linux-or-macos-openai-style-request)
  - [Linux or macOS: Ollama-style request](#linux-or-macos-ollama-style-request)
  - [Port reference](#port-reference)
  - [Change ports or engine launch options](#change-ports-or-engine-launch-options)
  - [Use an engine's command line](#use-an-engines-command-line)
- [Set Up PAIR with Terminal](#set-up-pair-with-terminal)
  - [Keep PAIR running after an SSH disconnect](#keep-pair-running-after-an-ssh-disconnect)
  - [Move around the terminal interface](#move-around-the-terminal-interface)
  - [Use the terminal interface tabs](#use-the-terminal-interface-tabs)
  - [Inspect errors and logs](#inspect-errors-and-logs)
  - [Change ports and startup arguments](#change-ports-and-startup-arguments)
  - [Command flags and current limits](#command-flags-and-current-limits)
- [Troubleshooting](#troubleshooting)
  - [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA PAIR connects computers on your local network and
presents Ollama-compatible and OpenAI-compatible proxy endpoints to applications
and agents. It automatically routes each independent AI inference request to an
eligible computer according to engine availability, model availability, and
current workload.
Run PAIR on one computer for local inference, or run it on several computers
to handle more requests.

This playbook covers PAIR **1.0.0**, including managed **llama.cpp** support and
the redesigned terminal interface. Ollama and LM Studio remain available.

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
- llama.cpp, Ollama, or LM Studio, plus a model, on at least one system that
  will serve requests.
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
| **RTX Spark** | Windows 11 on ARM | Unified memory; capacity depends on the system | ✅ (PAIR cluster) |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | ✅ (PAIR cluster) |
| **GeForce RTX** | Windows 11 or Linux | Depends on the GPU and model | ✅ (PAIR cluster) |
| **RTX PRO** | Windows 11 or Linux | Depends on the GPU and model | ✅ (PAIR cluster) |
| **Mac (Apple silicon or Intel)** | macOS 13 or later | Depends on the Mac and model | ✅ (PAIR cluster) |

NVIDIA RTX Spark, DGX Spark, GeForce RTX, RTX PRO, and compatible Macs are the
hardware platforms covered by this playbook. GeForce RTX requires a 20 Series
or newer GPU, and RTX PRO requires a Turing or newer GPU.
PAIR can also pair compatible systems that run the supported operating systems
below.

| Support | Details |
| :---- | :---- |
| Operating systems | Windows 11, Linux, and macOS 13 or later |
| Architectures | x64 and ARM64 packages for Windows, Linux, and macOS. RTX Spark uses Windows ARM64; DGX Spark uses Linux ARM64. |
| Installers | Windows `.exe`, Linux `.deb`, and macOS `.dmg`. Build from source for other Linux distributions. |
| Mixing systems | Windows, Linux, and macOS systems can pair with each other. |
| Inference engines | llama.cpp, Ollama, and LM Studio |

Open the [NVIDIA PAIR download page](https://www.nvidia.com/en-us/ai-on-rtx/personal-ai-router/)
and select your operating system and architecture. Check that the downloaded
package is for **1.0.0**; use the
[PAIR releases page](https://github.com/NVIDIA/Personal-AI-Router/releases) to
check the version and release notes. The app setup tab lists the installer
choices. These instructions use native Windows PowerShell on Windows, with no
WSL requirement.

### Why memory matters

Each system serving a request needs enough memory for the model weights,
the model's context, and the engine's runtime overhead. A model's download
size is not its total memory requirement. On RTX Spark, DGX Spark, and Apple
silicon Macs, the CPU and GPU share unified memory; on systems with a discrete
GPU, GPU VRAM and system RAM are separate resources. Leave room for the
operating system and other applications.

Use your system's memory information and the model's requirements when choosing
a model. PAIR's GPU-memory display can understate available memory on Windows
systems with unified memory. Pairing systems does not add their memory together
to fit one larger model.

## Prerequisites

**Hardware requirements**

- One compatible system for local inference.
- Two or more compatible systems on the same trusted local network for cluster
  routing.

**Software requirements**

- The PAIR 1.0.0 package for each participating system's OS and architecture.
- llama.cpp, Ollama, or LM Studio running on each system that will serve requests.
- A model downloaded on at least one system that will serve requests.
- Internet access and free disk space for installer, engine, and model downloads.

PAIR can run on a supported system even when an engine cannot. Each engine has
its own requirements for the operating system, GPU, and drivers. Each model
also needs enough memory to load. Check the engine documentation before you
expect a system to serve a model.

### Choose an engine and model

For the RTX Spark example in this playbook, select **llama.cpp** in PAIR and
let PAIR install it. The 1.0.0 release adds llama.cpp with CUDA support for
RTX Spark's Windows ARM64 platform. Choose Ollama or LM Studio when you need
their model format or client workflow and they support your system.

| Engine | Model selection | Request style |
| --- | --- | --- |
| llama.cpp | Compatible GGUF models from Hugging Face, such as `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | OpenAI-compatible |
| Ollama | Exact Ollama model names, such as `llama3.2` | OpenAI-compatible or Ollama-native |
| LM Studio | A model available through LM Studio; copy its advertised ID | OpenAI-compatible |

For llama.cpp, the example ID consists of a Hugging Face owner (`ggml-org`),
repository (`gemma-3-1b-it-GGUF`), and quantization (`Q4_K_M`). Use the exact
model ID shown by PAIR or returned by `<PAIR_BASE_URL>/v1/models` in requests;
do not substitute a local GGUF filename or a display name.

## Ancillary files

No extra files are required.

## Time & risk

- **Estimated time:** About 10 minutes, plus engine and model download time.
- **Risk level:** Low to medium. PAIR installs software, downloads models, and
  enables communication between trusted local systems.
- **Rollback:** Remove paired systems, uninstall PAIR, and remove engines or
  models you no longer need.
- **Last updated:** 10/06/2026.

For a graphical setup, open **Set Up the PAIR App**. For a headless or SSH
setup, open **Set up PAIR with Terminal**.

## Set Up the PAIR App

## Step 1. Install and open PAIR

These instructions target **PAIR 1.0.0**, including llama.cpp support. Install
that version on each desktop system that will join the cluster.

Open the [NVIDIA PAIR download page](https://www.nvidia.com/en-us/ai-on-rtx/personal-ai-router/),
select **Download PAIR**, then choose the operating system and architecture.
Check that the downloaded filename contains `1.0.0`. If the download page
provides an older version, use the matching package from the
[PAIR releases page](https://github.com/NVIDIA/Personal-AI-Router/releases).

| System | Package filename |
| --- | --- |
| RTX Spark or another Windows ARM64 system | `NVPAIR-Setup-1.0.0-arm64.exe` |
| Windows on an Intel or AMD processor | `NVPAIR-Setup-1.0.0-x64.exe` |
| DGX Spark or another Debian/Ubuntu ARM64 system | `NVPAIR-Setup-1.0.0-arm64.deb` |
| Debian/Ubuntu on an Intel or AMD processor | `NVPAIR-Setup-1.0.0-amd64.deb` |
| Mac with Apple silicon | `NVPAIR-Setup-1.0.0-arm64.dmg` |
| Mac with an Intel processor | `NVPAIR-Setup-1.0.0-x64.dmg` |

### Windows

1. Download the Windows installer that matches the system's architecture.
2. Run the installer and approve the operating-system and firewall prompts.
3. Open **NVIDIA PAIR** from the Start menu.

RTX Spark uses the native Windows ARM64 installer and the PowerShell examples
below.

### Debian or Ubuntu

Open a terminal in the directory that contains the downloaded package, then
run:

```bash
sudo apt install "./NVPAIR-Setup-VERSION-ARCH.deb"
```

Replace `VERSION` and `ARCH` with the values in the downloaded filename. Then
open **NVIDIA PAIR** from the desktop application menu.

### macOS

Use macOS 13 or later. Choose ARM64 for Apple silicon and x64 for an Intel Mac.

1. Open the downloaded `.dmg`.
2. Drag **NVIDIA PAIR** to **Applications**.
3. Open **NVIDIA PAIR** from **Applications**.

Upgrading the macOS app preserves downloaded models.

PAIR opens to its setup window or **Overview** when started.

## Step 2. Complete first-run setup

When you first open PAIR, it shows the engines that it can install: Ollama,
LM Studio, and llama.cpp. Available engines can already be selected.

1. Review the available engines.
2. For the RTX Spark example, select **llama.cpp** and clear the other selections
   unless you also want those engines. On other systems, choose any supported
   engine, or skip engines that you manage separately.
3. Finish setup and wait for the selected engine to report that it is running.

The first start can take longer while PAIR starts its background services. If
**Overview** still shows **Loading...** after one or two minutes, open
**Settings → Service**, then use the **Troubleshooting** tab.

You can return to engine settings by selecting a node in **Overview**. Use
**Settings → Cluster** to pair systems at any time.

## Step 3. Pair systems

Start PAIR on each system. Confirm that the systems are on the same trusted
local network.

1. On the first system, select **Add node** in the top-right
   toolbar. You can also open **Settings → Cluster** and use **Available nodes
   to add**.
2. Select a discovered system. If PAIR does not find it, add the system by IP
   address.
3. PAIR shows a six-digit PIN and sends an invitation.
4. On the invited system, accept the **Cluster invitation** and enter the PIN.
5. Open **Settings → Cluster** and check that the peer appears under
   **Connected nodes**. You can also check the node list in **Overview**.
6. Repeat these steps from any cluster member to add more systems.

The peer appears as a connected node when pairing is complete. With one system,
continue to step 4 without pairing.

## Step 4. Start an engine and add a model

Repeat these steps on every system that should serve the model:

1. Select the system in **Overview** to open its engine settings.
2. Install **llama.cpp**, **Ollama**, or **LM Studio** if needed. Installation
   starts the engine; if it is stopped, use its switch to start it.
3. Expand the engine and select **Add model**.
4. Select a model, select **Download**, and wait for the download to finish.
5. Load the model if the engine requires a separate load step.

A system can serve a request when it is online, the engine is running, and the
requested model is available there. To route requests across several systems,
add the same model to the same engine on each system that should serve it.
The downloaded model appears in that engine's model list.

### llama.cpp example

On RTX Spark, use PAIR's managed **llama.cpp** installation. PAIR downloads the
engine and CUDA runtime separately from the PAIR installer.

1. Under **llama.cpp**, select **Add model**.
2. Search for `ggml-org/gemma-3-1b-it-GGUF`. Typing filters the initial list;
   press **Enter** or select the search button to search public Hugging Face
   repositories if it is not listed.
3. Select the `Q4_K_M` entry and select **Download**.
4. Wait until `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` appears in the engine's model
   list. Use **Load** to load it before testing, or allow the first request to
   load it.

This small model is an example; you can choose another compatible GGUF model
that fits the system's memory. PAIR's llama.cpp catalog offers verified
`Q4_K_M` entries from public repositories. Model IDs include both the repository
and quantization: `owner/repository:quantization`. Use the exact ID when sending
a request; an Ollama tag such as `gemma3:1b` is a different identifier.

Use **Eject** to release a loaded model's memory while keeping its download.
Managed llama.cpp models also sleep after five minutes without inference work;
the next request wakes the model. Some GPU memory can remain allocated by the
engine.

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
> and uses a separate port for the engine. llama.cpp defaults to proxy port
> `8080` and server port `8081`.

## Step 6. Send a test request

Use the endpoint from step 5 and the exact downloaded model ID from step 4.
The examples below use the OpenAI-compatible API to work with all three engines.

### Choose a request style

PAIR passes each request to the selected engine without rewriting it.

| Endpoint | OpenAI `/v1/chat/completions` | Ollama `/api/chat` |
| --- | --- | --- |
| Ollama | Works | Works |
| LM Studio | Works | Not available |
| llama.cpp | Works | Not available |

If you are unsure, use the OpenAI-style request. It works with every supported
engine.

### Windows PowerShell: llama.cpp example

Open PowerShell on the system running PAIR. Set `$pairBaseUrl` to the endpoint
you copied. With llama.cpp's default proxy port, list the available model IDs:

```powershell
$pairBaseUrl = "http://127.0.0.1:8080"
(Invoke-RestMethod -Uri "$pairBaseUrl/v1/models").data | Select-Object id
```

Confirm that the output includes the model you downloaded. Set `$pairModel` to
its exact `id`, then send the request:

```powershell
$pairModel = "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M"
$pairRequest = @{
    model = $pairModel
    messages = @(@{
        role = "user"
        content = "Tell me a short story about a dog who learns to skateboard."
    })
    stream = $false
} | ConvertTo-Json -Depth 5
$pairResponse = Invoke-RestMethod -Uri "$pairBaseUrl/v1/chat/completions" -Method Post -ContentType "application/json" -Body $pairRequest
$pairResponse.choices[0].message.content
```

The response prints the model's answer. To use Ollama or LM Studio in
PowerShell, use that engine's endpoint and an exact model ID returned by its
`/v1/models` endpoint.

### Linux or macOS: OpenAI-style request

In a terminal, replace `<PAIR_BASE_URL>` with the endpoint from step 5 and list
its models:

```bash
curl <PAIR_BASE_URL>/v1/models
```

Copy the exact `id` from the response. Replace `<MODEL_NAME>` with that ID and
`<PAIR_BASE_URL>` with the same endpoint in the request below:

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

### Linux or macOS: Ollama-style request

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
the request. Send several independent requests to observe routing across systems
that have the same engine and exact model ID. One request runs on one system.

## Step 7. Connect an application

Set the application's base URL to the endpoint from step 5. Select a model
that you added in step 4.

For an OpenAI-compatible SDK or application that appends the API path, include
`/v1` in the base URL. For example, llama.cpp uses
`http://127.0.0.1:8080/v1` with the default proxy port. A direct HTTP request uses
the complete path, such as `http://127.0.0.1:8080/v1/chat/completions`. An
Ollama-native client uses the host URL, such as `http://127.0.0.1:11434`.

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

PAIR's proxy uses the port that an engine normally uses. Engines run behind the
proxy on separate server ports.

| Service | Default port |
| --- | --- |
| Ollama-compatible proxy | `11434` |
| Ollama engine behind PAIR | `11435` and upward |
| LM Studio / OpenAI-compatible proxy | `1234` |
| LM Studio engine behind PAIR | `1235` and upward |
| llama.cpp / OpenAI-compatible proxy | `8080` |
| Managed llama.cpp server behind PAIR | `8081` |

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

### Change ports or engine launch options

1. Open **Overview** and expand **Engine settings** on the local system's card.
2. Expand the engine's **Settings**.
3. Edit **Proxy port**, **Server port**, or the engine arguments, then select
   **Apply**. Confirm the restart if PAIR requests it.

PAIR applies the change as one operation and restores the new values when it
next starts. Supported paired systems can also be edited from another cluster
member. If you change a proxy port, update the application's base URL.
**Endpoints** always shows the current URL.

Keep managed llama.cpp in router mode. Do not add `-m` or `-hf` model-selection
arguments; download and load models with PAIR's model controls instead. See
[PAIR engine settings](https://github.com/NVIDIA/Personal-AI-Router/blob/feature/tui-llamacpp/docs/engine-settings.mdx)
for supported argument notation and settings.

For browser applications, configure only the required CORS origins on the system
running the engine. Managed llama.cpp starts without cross-origin browser
permission, and PAIR follows the engine's policy. Native clients such as
PowerShell do not need a CORS setting.

If another application uses a port that PAIR does not manage, choose a
different port in PAIR or stop the other application. Then restart the service
from **Settings → Service**.

### Use an engine's command line

PAIR-installed engines are normal installations, but their binaries are not in
`PATH` and their server ports differ from the usual defaults.

PAIR-managed Ollama locations:

| Platform | Path |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Nvidia Corporation\Personal AI Router\engine-bin\ollama\ollama.exe` |
| Linux | `~/.config/Nvidia Corporation/Personal AI Router/engine-bin/ollama/bin/ollama` |
| macOS | `~/Library/Application Support/Nvidia Corporation/Personal AI Router/engine-bin/ollama/Ollama.app/Contents/Resources/ollama` |

On Linux, the table shows the default location. If `XDG_CONFIG_HOME` is set,
PAIR uses it instead of `~/.config`. The command below handles either location:

```bash
pair_ollama_dir="${XDG_CONFIG_HOME:-$HOME/.config}/Nvidia Corporation/Personal AI Router/engine-bin/ollama"
LD_LIBRARY_PATH="$pair_ollama_dir/lib/ollama" OLLAMA_HOST=127.0.0.1:11435 "$pair_ollama_dir/bin/ollama" list
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
over SSH. Quit the desktop application first, including its tray or menu-bar
instance. Run one PAIR interface per system: the desktop application and
terminal interface each start services that use the same ports, engines, and
settings.

These instructions target **PAIR 1.0.0** and its five-tab terminal interface.
Install the matching 1.0.0 platform package from the
[NVIDIA PAIR download page](https://www.nvidia.com/en-us/ai-on-rtx/personal-ai-router/)
or [PAIR releases page](https://github.com/NVIDIA/Personal-AI-Router/releases).
The app setup tab lists installer choices by OS and architecture.
On Windows, launching the installed desktop application creates the `nvpair`
command wrapper. On macOS it does so when `/usr/local/bin` is writable. Quit
the application before using that command and open a new terminal after the
wrapper is created. On Windows, use native PowerShell. The Debian package
creates the wrapper during installation.

```shell
nvpair --version
nvpair
```

For a system where you will not launch the desktop application, download and
extract the **1.0.0** services and terminal-interface archive for your platform
and architecture instead.
Keep all its binaries together. From the directory containing them, run:

**Windows PowerShell:**

```powershell
.\nvpair-tui.exe --version
.\nvpair-tui.exe
```

**Linux or macOS:**

```bash
./nvpair-tui --version
./nvpair-tui
```

Confirm the downloaded package or archive belongs to PAIR `1.0.0`.
The `--version` command prints the terminal-interface component version,
which can differ from the PAIR release version.
The terminal interface starts its PAIR service tree; no other PAIR process
needs to be running first. The header shows `service ready` and a version when
it connects. If it stays on `starting service...`, inspect **Logs** (tab 5).

## Step 2. Pair this system

One system can serve local inference. To route requests across systems, start
PAIR on each system and pair them on a trusted local network. The six-digit
PIN is a temporary setup code. Inviting the first system forms the cluster
automatically.

To pair with a discovered system:

1. Open **Nodes** (tab 1).
2. Select the other system with `j` / `k` or the arrow keys.
3. Press `p`.
4. Give the displayed PIN to the person operating the other system.

If discovery has not found the system, press `n` on **Nodes**, enter its
hostname or IP address, and press `enter`. Use `host:port` when the system
does not use the default pairing port.

On the invited system, open **Nodes**, press `a` when the pairing request
appears, enter the PIN, and press `enter`. Press `d` to decline. When pairing
succeeds, the peer's **CLUSTER** column reads `Member`.

Press `c` to cancel an invitation you sent. A wrong or expired PIN requires a
new invitation. Repeat pairing to add more systems; each system can belong to
one cluster at a time.

## Step 3. Start an engine and add a model

In **Nodes**, select your own system and press `enter` to open its hardware,
**Engines**, and **Models**. Move between the two panes with `h` / `l` or
`←` / `→`. Within a pane, select a row with `j` / `k` or `↓` / `↑`.

For a llama.cpp example:

1. In **Engines**, select **llama.cpp** and press `i` to install it.
2. Press `s` to start it. Wait for **RUNNING** and **HEALTHY** to read `yes`.
3. Keep llama.cpp selected and move to **Models**.
4. Press `n`, enter `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M`, and press `enter`.
5. Wait for the download to finish and the model to appear in the model list.
6. Select that model and press `enter` to load it. Confirm **LOADED** reads `yes`.

Select the intended running engine before downloading: the download goes to
the highlighted running engine. llama.cpp uses a Hugging Face GGUF repository
and quantization as its model ID. Keep the exact ID, including `:Q4_K_M`, for
model operations and inference requests.

You can browse instead of typing a name. In **Models**, press `p` to open the
selected engine's download catalog. Select a model and press `enter` to
download it. For llama.cpp, `/` opens a Hugging Face search; enter a query and
press `enter`. Press `c` to return to the initial popular-model list, `o` to
change the sort order, and `esc` to leave the catalog. Other engines use `/`
to filter their catalog by name, family, or parameter size.

| Pane | Key | Action |
| --- | --- | --- |
| Engines | `i` / `s` / `x` | Install, start, or stop the selected engine. |
| Engines | `r` / `u` | Restart or uninstall a local engine. |
| Engines | `e` / `p` / `a` | Edit engine port, proxy port, or startup arguments. |
| Models | `p` / `n` | Browse downloads or download by exact name. |
| Models | `enter` / `e` | Load the selected model or eject it from memory. |
| Models | `d` | Delete the selected model after confirmation. |

The same node-detail screen can install, start, and stop engines and manage
models on paired peers. Restart and uninstall are available on the local
system only. Destructive operations require `y` to confirm; any other key
cancels.

A system can serve a request when it is online, its compatible engine is
running, and the requested model is available there. Prepare the same exact
model on multiple systems when you want any of them to serve it. Press `esc`
to return to the **Nodes** list and check each system's model inventory.

## Step 4. Check the service and endpoint

Open **Service** (tab 3) and confirm the service is connected. Worker health
is best-effort: `ok` means no crash was reported, `DOWN` means a worker
reported a crash, and `?` means PAIR is still checking or the service is not
answering. The error-reporting worker cannot report its own crash. Use
**Errors** and **Logs** to investigate a failure.

Open **Jobs** (tab 2). Its top line shows each engine's current local proxy
port. Use the port for the engine you prepared to form
`http://127.0.0.1:<port>`. This is the client endpoint; the engine's own port
is a separate value.

| Engine | Default client endpoint |
| --- | --- |
| Ollama | `http://127.0.0.1:11434` |
| LM Studio | `http://127.0.0.1:1234` |
| llama.cpp | `http://127.0.0.1:8080` |

Read the actual port in **Jobs**, because it can differ from the default.
Keep the terminal interface running and open another terminal on the same
system to check the endpoint. The examples below use llama.cpp's default;
replace the URL if **Jobs** shows another port.

**Windows PowerShell:**

```powershell
$pairBaseUrl = 'http://127.0.0.1:8080'
Invoke-RestMethod -Uri "$pairBaseUrl/v1/models" | Select-Object -ExpandProperty data
```

**Bash:**

```bash
PAIR_BASE_URL=http://127.0.0.1:8080
curl "$PAIR_BASE_URL/v1/models"
```

The response lists the cluster's available model IDs, including the model you
prepared. Use an exact returned ID in your client. An OpenAI-compatible
client normally uses `<PAIR_BASE_URL>/v1` as its base URL and appends API
paths itself. All three engines accept OpenAI-style chat requests; Ollama's
`/api/chat` is available only on the Ollama endpoint.

The proxy endpoint accepts requests from its own system only. Run PAIR on
the system hosting your client and pair it with the systems serving models.
That client system does not need its own engine or GPU.

## Step 5. Check routing activity

Open **Jobs** and press `t` to generate up to sixty seconds of synthetic
inference traffic through the local endpoints. It needs a running engine
with a text-generation model available. The progress line shows requests
sent and time remaining. Press `t` again to stop new requests early; requests
already sent can finish.

Jobs appear in the table. **FROM** identifies the system where a request
arrived; **RAN ON** identifies the system that served it. Press `a` to include
finished jobs. Different systems in those columns demonstrate routing to a
peer. PAIR selects a system for each request automatically; one request runs
on one system.

The synthetic test shows routing activity without displaying prompts or
responses. To verify an answer, send a request from your configured client
and confirm both its reply and the serving system in **Jobs**. To prove
remote routing, prepare the requested model on a peer only, send from your
local endpoint, and check that **RAN ON** names that peer.

## Next steps

### Keep PAIR running after an SSH disconnect

Quitting the terminal interface stops its PAIR services. An SSH disconnect
also ends it. On a Linux system reached over SSH, use a terminal multiplexer
when PAIR must stay available:

```bash
tmux new -s pair
nvpair
```

If you extracted a services archive, start `./nvpair-tui` from its binary
directory inside the tmux session instead. Detach with `Ctrl-b d` and reattach
with `tmux attach -t pair`. GNU Screen also works: start with `screen -S pair`,
detach with `Ctrl-a d`, and reattach with `screen -r pair`. Neither tool is
included with PAIR.

### Move around the terminal interface

Use a terminal at least 40 columns wide and 12 rows high. The tab bar and
footer show the current controls. When a text field is open, press `enter`
to submit or `esc` to cancel before changing tabs.

| Key | Action |
| --- | --- |
| `1` – `5` | Go directly to a tab. |
| `tab` / `shift+tab` | Move to the next or previous tab. |
| `?` | Show or hide full help. |
| `q` or `ctrl+c` | Quit and shut down the PAIR service tree. |
| `j` / `k` or `↓` / `↑` | Move within a table or settings list. |
| `h` / `l` or `←` / `→` | Move between node-detail panes. |
| `home` / `end` | Jump to the first or last table row. |

### Use the terminal interface tabs

| # | Tab | What it shows |
| --- | --- | --- |
| 1 | **Nodes** | Systems, reachability, cluster membership, pairing, and node details with engines, models, and hardware. |
| 2 | **Jobs** | Local endpoint ports, inference activity, originating and serving systems, and the synthetic test. |
| 3 | **Service** | Service version, uptime, workers, settings, and data reset. |
| 4 | **Errors** | Active errors by severity, age, system, and message. |
| 5 | **Logs** | Service output, filtering, tailing, and save-to-file. |

### Inspect errors and logs

**Errors** shows its active-error count in the tab label. Select an entry to
read its operation and suggested action. Press `c` to clear an error on the
system that reported it; a peer's error must be cleared from that peer.

In **Logs**, scroll with `j` / `k`, press `/` to filter, `c` to clear the
filter, and `t` to toggle tailing. Press `s` to save the full log buffer to a
timestamped file in your home directory. Change the service log level from
**Service**.

### Change ports and startup arguments

On **Nodes**, open a system's detail screen and select an engine in
**Engines**. Press `e` to edit its engine port, `p` to edit its proxy port,
or `a` to edit startup arguments and environment assignments. Clients use
the proxy port. When changing it, update your client's base URL and check
the resulting port in **Jobs** on that system.

These settings can also be edited on paired peers when the engine supports
configuration; changes to CORS settings must be made on the engine's own
system. An engine PAIR found already running can report that its settings
are not editable. Startup arguments use literal shell-style
quoting, with environment assignments first; variables such as `$HOME` and
`%USERPROFILE%` are not expanded. Do not include the engine executable or
startup subcommand.

Press `enter` to validate and save. PAIR validates syntax and its managed
network settings; other options must be supported by the engine. If applying
the change requires an engine restart, confirm with `y`. Read the reported
result rather than assuming the requested port was available.

### Command flags and current limits

| Flag | Effect |
| --- | --- |
| `--broker-path <path>` | Use a PAIR service binary that is not beside the terminal-interface binary. |
| `--log-level <level>` | Set terminal-interface logging to `debug`, `info`, `warn`, or `error`. `NVPAIR_LOG_LEVEL` provides the same setting. |
| `--appearance <mode>` | Choose `auto`, `light`, or `dark`; use an explicit value if your terminal theme is hard to read. |
| `--version` | Print the terminal-interface component version and exit. |

The terminal interface can notify you of PAIR updates but does not install
them. It cannot update an inference engine; use the desktop application for
that operation. The **Jobs** test generates synthetic traffic, while a
compatible external client sends your own prompts and displays the replies.

## Troubleshooting

### Troubleshooting

Start with the symptom and then inspect **Settings → Service**, the desktop
error surface, or the terminal interface's **Errors** tab for more detail.
In the terminal interface, also check **Service** for worker state and **Logs**
for engine installation or startup output.

| Symptom | Usual meaning | What to do |
| --- | --- | --- |
| PAIR remains on **Loading...** | A background service did not start correctly. | Wait one or two minutes, then inspect **Settings → Service** and restart the affected service. |
| llama.cpp is missing, or the terminal shows the old ten-tab layout | The installed PAIR version predates this playbook's 1.0.0 workflow. | Check the PAIR version and install the matching 1.0.0 package for your OS and architecture. |
| An installer does not match the system | The architecture choice is incorrect. | Use Windows ARM64 for RTX Spark, Linux ARM64 for DGX Spark, and macOS ARM64 for Apple silicon. Intel/AMD systems use x64 packages, named `amd64` for Debian/Ubuntu. |
| LM Studio remains after uninstalling PAIR 1.0.0 | LM Studio installations created by older PAIR versions are not removed by the newer PAIR uninstaller. | If you also want to remove LM Studio, uninstall that older installation manually from `~/.lmstudio` on Linux or macOS, or `%USERPROFILE%\.lmstudio` on Windows. |
| A node is not discovered | mDNS is blocked or unavailable. | Confirm both nodes are on the same trusted local network, allow `5353/udp`, or add the node by IP address. |
| Pairing stalls or fails | The invitation expired, the PIN is incorrect, or port `14321` is blocked. | Start a new invitation, enter the new PIN, and confirm cluster ports are reachable. |
| Connection refused | Nothing is listening at that address. | Copy the current URL from **Endpoints** and confirm the PAIR service is running. |
| `403` from another machine | PAIR endpoints accept loopback traffic only. | Run PAIR on the machine hosting the client and use its local endpoint. |
| `502` with `no active node` | No node is eligible for the request. | Start a compatible engine and make the requested model available on at least one online node. |
| A model is listed but the inference request fails | The request's model ID, API route, or engine does not match the prepared model. | Copy the exact ID from `<PAIR_BASE_URL>/v1/models`. For llama.cpp, preserve `owner/repository:quantization` and use `/v1/chat/completions`, not Ollama's `/api/chat`. |
| `400` or `422` | The request is malformed. | Correct its JSON, route, model name, or required fields; malformed requests are not retried. |
| A response arrives but **Jobs** is empty | Another process owns the expected proxy port. | Check **Endpoints** and **Settings → Service**, then change the PAIR port or stop the conflicting process. |
| Requests do not use every GPU | PAIR routes each request to one eligible node. | Send independent requests and verify **Ran on** for each job. PAIR does not split one request across GPUs. |
| Desktop and terminal behavior conflicts | Both PAIR interfaces are running on one system. | Stop one interface and use only the desktop application or terminal interface. |
| `nvpair` is not found in native Windows PowerShell | The packaged launcher has not been created or the shell has not picked up its PATH entry. | Open the installed PAIR app once, quit it, and open a new PowerShell window before running `nvpair`. |
| A llama.cpp model download or load fails | The GGUF selection or available resources need checking. | Confirm the selected engine is llama.cpp, choose a compatible catalog entry, and inspect **Errors** and **Logs**. Check available disk space and memory, including context overhead. |
| llama.cpp fails after custom launch arguments | A fixed-model argument can conflict with PAIR's managed router mode. | Remove custom `-m` or `-hf` arguments and use PAIR's model download and load controls. |
| A loaded llama.cpp model becomes idle and the next request takes longer | Managed llama.cpp puts idle models to sleep after five minutes and wakes them for a new request. | Allow the model to wake; inspect **Jobs** to confirm that the request completes. |
| A browser client reports a CORS error | PAIR 1.0.0 follows the selected engine's origin policy. | Configure the allowed browser origin on the system running that engine. For Ollama, set `OLLAMA_ORIGINS` in the engine's **Settings** launch arguments. |
