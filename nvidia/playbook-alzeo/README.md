# Run a Private AI Server with Alzeo

> One command installs a served model, OpenAI and Anthropic compatible endpoints, and a browser cockpit

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Alzeo is an open-source (MIT) installer for DGX Spark and other GB10 machines. One command installs a served model, a
proxy that speaks the OpenAI and the Anthropic APIs, and a browser cockpit. Other models (text, vision, images, video)
are installed on demand, each loaded with one button from the cockpit, and they take turns on the GPU. Speech runs on
the CPU beside them. Every image is pinned by its digest and every checkpoint by its revision. It is a community
project, not an NVIDIA product.

## What you'll accomplish

- Serve Qwen3.8 27B with a 1M-token context behind OpenAI-compatible and Anthropic-compatible endpoints.
- Open a cockpit in your browser that shows what serves and what it holds in memory, and loads another model.
- Point a client you already use (an OpenAI SDK, Claude Code, opencode) at the box.

## What to know before starting

- Basic use of a terminal on the Spark.
- The engine listens on the box itself. Clients reach it through the proxy, which asks for an API key.

## Prerequisites

- A DGX Spark or another GB10 machine with 128 GB of unified memory (a 64 GB DGX Spark gets the models that fit it).
- Stock DGX OS, with its own Docker CE and NVIDIA Container Toolkit (not Docker from the Snap package).
- Free disk: about 58 GB for the default model (a new box downloads about 39 GB). Each other model takes its
  room when you first load it.
- An account that can use `sudo`. Do not put `sudo` in front of the command: the installer asks for it when needed.

## Time & risk

- **Time:** a new box downloads about 39 GB (6 to 7 minutes at 100 MB/s), then the first boot of the model takes 7 to
  9 minutes.
- **Risk:** low. The installer pulls a container image and writes systemd units, a configuration folder
  (`~/.config/qwen38`) and a clone (`~/dgx-spark-qwen38`). It says each step as it runs, and `./uninstall.sh --list`
  shows everything it left.
- **Rollback:** `./uninstall.sh` removes the services and the configuration and prints the commands that reclaim the
  model files.

---

## Instructions

## Step 1. Install

On the Spark, as your own user:

```bash
curl -fsSL https://get.alzeo.ai | bash
```

The run ends by printing the cockpit's address. The model proves that it serves before the run ends.

## Step 2. Open the cockpit

Open the address the installer printed (port 30090). The Now view shows the model that serves, the address a client
sends its requests to, and where the API key is.

## Step 3. Send a request

On the Spark (the key is read by the shell and never appears in a process list):

```bash
curl http://127.0.0.1:30001/v1/chat/completions \
  -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.config/qwen38/api-key)") \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

From another machine, the address is `http://<spark>:30001`, with the same key. The Anthropic protocol is at
`/v1/messages` (Bearer authorization).

## Step 4. Load another model

In the cockpit's Lanes view, press **Load** on another model. Its first Load downloads and prepares it, with no sudo.
The models take turns: loading one stops the one that served.

## Step 5. Update

The same command updates the box and keeps your choices:

```bash
cd ~/dgx-spark-qwen38 && curl -fsSL https://get.alzeo.ai | bash
```

## Step 6. Cleanup and rollback

```bash
cd ~/dgx-spark-qwen38
./uninstall.sh --list     # everything any version left here, with sizes
./uninstall.sh            # removes the services and the configuration
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A client gets 401 | It does not send the key | Send `Authorization: Bearer` with the key of `~/.config/qwen38/api-key` |
| The installer stops on Docker | Docker comes from the Snap package | Use the Docker CE that DGX OS ships |
| The box stops answering while a model loads | Unified memory was exhausted | Load one model at a time from the cockpit, which refuses two engines at once |
| The first request is slow after a restart | The model is still booting (7 to 9 minutes at first) | Watch the Lanes view until it says ready |

More: https://github.com/hasso5703/alzeo and https://alzeo.ai/guides/
