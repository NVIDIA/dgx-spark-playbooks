# Keeping vLLM Up

> Run vLLM as a long-lived service on DGX Spark: sizing the memory reservation, probing liveness without causing outages, and sandboxing the service without losing the GPU

## Table of Contents

- [Overview](#overview)
- [Sizing the reservation](#sizing-the-reservation)
- [A liveness check that does not cause outages](#a-liveness-check-that-does-not-cause-outages)
- [Sandboxing without losing the GPU](#sandboxing-without-losing-the-gpu)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Basic idea

[vLLM for Inference](../vllm/README.md) covers getting vLLM running. This
playbook covers keeping it running — the questions that only appear once the
server has been up for weeks and something else on the box depends on it.

Three of them, each with a failure mode specific to this hardware:

- `--gpu-memory-utilization` is spending your **system RAM**, and the default
  reservation is likely far larger than a single-user box will ever use.
- `/health` returns 200 on an engine that has stopped serving, so a liveness
  check has to ask the model to generate. Building that probe carelessly makes
  it *cause* the outages it reports.
- `PrivateDevices=true` takes the GPU away from your service, and the obvious
  fix does not work.

### What you'll accomplish

You'll pick a memory reservation deliberately rather than by copying a flag,
install a watchdog that restarts a genuinely wedged engine and refuses to
restart a healthy one, and harden the surrounding service without breaking
CUDA inside it.

### What to know before starting

- vLLM already serving on the box, with an API key set
- Comfort with `systemd` units and timers
- Measurements here are from a DGX Spark (GB10, 121 GB unified, DGX OS,
  aarch64) serving Qwen3.6-35B-A3B in NVFP4

---

## Sizing the reservation

On a discrete GPU, `--gpu-memory-utilization` carves up VRAM. The GB10 has
none — CPU and GPU share one pool — so **it carves up the same RAM your other
services run in**. A value copied from a discrete-GPU example is a decision
about your whole machine, not just the model server.

vLLM tells you the split at startup. Read it rather than guessing:

```
gpu_worker.py:538]      Available KV cache memory: 53.84 GiB
kv_cache_utils.py:2146] GPU KV cache size: 4,073,445 tokens
kv_cache_utils.py:2147] Maximum concurrency for 131,072 tokens per request: 31.08x
```

At `--gpu-memory-utilization 0.75` on a 121.7 GB machine that is roughly:

| | |
|---|---|
| Model weights (35B, NVFP4) | ~22 GB |
| KV cache | 53.8 GB |
| Activations, CUDA graphs, overhead | ~15 GB |
| **Reserved at startup** | **~91 GB** |

It is a reservation, not consumption — taken at load, held whether or not
anything is inferencing, and never returned.

That matters because the third line of that log says the cache is sized for 31
concurrent requests at a full 131k context. On a single-user box, measured at
idle and during a request:

```
GPU KV cache usage: 0.0%
GPU KV cache usage: 0.5%
```

Under 1% of a 54 GB reservation. If you need RAM back for other services,
this is the dial: `0.75` → `0.55` frees roughly 24 GB and still leaves about
17 concurrent full-length requests. `--max-model-len` is the other lever if
you never send very long prompts.

> [!NOTE]
> `nvidia-smi --query-gpu=memory.used` reports `N/A` here for the same reason.
> Track `MemAvailable` in `/proc/meminfo`, which accounts for kernel
> reservations and page-cache pressure that `MemTotal` does not.

---

## A liveness check that does not cause outages

### `/health` will lie to you

A vLLM engine can wedge while every signal looks healthy: `/health` returns
200, the process is alive, the GPU shows busy, and the scheduler quietly stops
accepting work — `Running: 0 reqs`, KV cache 0.0%, nothing in the journal.

The only check that catches this is asking the model to generate and seeing
whether it answers.

### Two ways to build that probe wrong

Both of these were live in production before being found, and both make the
watchdog restart a server that was working.

**Sending no credentials.** vLLM requires an API key. A probe without one gets
`{"error":"Unauthorized"}` back, which contains no `"choices"` — so a probe
looking for `"choices"` concludes the engine is wedged. Ours restarted a
healthy vLLM on its 45-minute floor, each restart costing four minutes of
weight reload and the entire KV cache, while logging `WEDGED ... investigate`
and the server answered completions in three seconds throughout.

**Reading the model name with a greedy match.** `/v1/models` returns a
permission object with its own `id` alongside the model's. A greedy
`sed 's/.*"id":"\([^"]*\)".*/\1/'` returns `modelperm-...`, the probe asks for
a model that does not exist, and the rejection again looks like a wedge.

The shared lesson: **a monitor that cannot ask the question does not detect
outages, it manufactures them.** Every failure branch must distinguish *"I
could not ask"* from *"the answer was wrong"*, because no restart has ever
repaired a credential.

### The watchdog

The full script is in [`assets/vllm-watchdog.sh`](assets/vllm-watchdog.sh):

```bash
#!/usr/bin/env bash
# Probe vLLM with a real completion; restart it only if it is genuinely wedged.
set -uo pipefail

URL=http://127.0.0.1:8000/v1/chat/completions
ENV_FILE=/etc/vllm/vllm.env          # holds VLLM_API_KEY=...
STATE=/var/lib/vllm-watchdog/state
PROBE_TIMEOUT=30
MIN_UPTIME=600                       # a loading model is not a wedged one
RESTART_COOLDOWN=2700                # 45 min hard floor between restarts
FAILS_BEFORE_RESTART=2               # one slow moment is not a wedge

say() { logger -t vllm-watchdog "$1"; echo "$1"; }
FAILS=0; LAST_RESTART=0
[ -f "$STATE" ] && . "$STATE" 2>/dev/null || true
save() { printf 'FAILS=%s\nLAST_RESTART=%s\nLAST_RESULT=%s\n' \
         "$FAILS" "$LAST_RESTART" "$1" > "$STATE"; }

# Derive the target. A hardcoded unit name silently disarms the watchdog the
# first time you swap models: the rail below sees an inactive unit, exits 0,
# and records "skipped" forever while nothing is guarded.
UNIT=$(systemctl list-units --state=active --no-legend 'vllm-*.service' \
       | awk '{print $1}' | head -1)
[ -n "${UNIT:-}" ] || { save "skipped: no vllm unit active"; exit 0; }

START=$(date -d "$(systemctl show "$UNIT" -p ActiveEnterTimestamp --value)" +%s)
NOW=$(date +%s)
[ $((NOW - START)) -lt "$MIN_UPTIME" ] && { save "skipped: still loading"; exit 0; }

# The key comes first: everything below, including asking which model is
# served, is an authenticated call.
KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$ENV_FILE" | tr -d "\"'" | head -1)
if [ -z "${KEY:-}" ]; then
    say "NO KEY in $ENV_FILE - refusing to probe; a restart cannot fix this"
    save "error: no api key"; exit 1
fi

# Take the FIRST "id" and nothing else — /v1/models also carries a permission
# object whose id a greedy match will return instead.
MODELS=$(curl -s -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models)
if echo "$MODELS" | grep -qi unauthorized; then
    say "the key in $ENV_FILE is not accepted; NOT restarting"
    save "error: unauthorized (not a wedge)"; exit 1
fi
MODEL=$(echo "$MODELS" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[ -n "${MODEL:-}" ] || { say "could not read the served model name"; save "error: no model"; exit 1; }

BODY=$(curl -s -m "$PROBE_TIMEOUT" "$URL" \
       -H 'Content-Type: application/json' \
       -H "Authorization: Bearer $KEY" \
       -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8,\"temperature\":0}")

if echo "$BODY" | grep -q '"choices"'; then
    [ "$FAILS" -ne 0 ] && say "recovered after $FAILS failed probe(s)"
    FAILS=0; save "ok ($MODEL)"; exit 0
fi

# A rejected key is a configuration fault and must never count toward the
# restart threshold. This is the branch whose absence caused the outages.
if echo "$BODY" | grep -qi unauthorized; then
    say "PROBE UNAUTHORIZED on $UNIT - the key is not accepted; NOT restarting"
    save "error: unauthorized (not a wedge)"; exit 1
fi

FAILS=$((FAILS + 1))
say "PROBE FAILED ($FAILS/$FAILS_BEFORE_RESTART) on $UNIT: no completion in ${PROBE_TIMEOUT}s"
if [ "$FAILS" -ge "$FAILS_BEFORE_RESTART" ]; then
    SINCE=$((NOW - LAST_RESTART))
    if [ "$SINCE" -lt "$RESTART_COOLDOWN" ]; then
        say "WEDGED but restarted ${SINCE}s ago - NOT restarting again; investigate"
        save "wedged: in cooldown"; exit 1
    fi
    say "restarting $UNIT"
    systemctl restart "$UNIT"
    FAILS=0; LAST_RESTART=$NOW; save "restarted $UNIT"
fi
save "failed ($FAILS)"
exit 1
```

Install it at `/usr/local/bin/vllm-watchdog` and drive it from a timer every
five minutes. A false positive here costs minutes of downtime, so the rails
matter as much as the probe: skip while the model is loading, require two
consecutive failures, never restart twice inside the cooldown, and never
restart on an authentication failure at all.

Confirm both paths before trusting it — a correct key should report
`LAST_RESULT=ok (<model>)` and exit 0, and a deliberately wrong one should
print `NOT restarting` and leave the unit's restart count untouched.

---

## Sandboxing without losing the GPU

If an agent or gateway runs beside vLLM, `systemd` hardening is worth adding.
`systemd-analyze security` scores it, and a standard set — `NoNewPrivileges`,
`ProtectSystem=strict`, `PrivateDevices`, `CapabilityBoundingSet=`,
`ProtectProc=invisible` — moved one service here from **7.8 EXPOSED** to
**4.0 OK**.

`PrivateDevices=true` will also take the GPU away from that service, and
`DeviceAllow` does not give it back. After adding `DeviceAllow=/dev/nvidia0 rw`
and restarting, the running service's `/dev` contained:

```console
$ sudo ls -1 /proc/$(systemctl show my-agent -p MainPID --value)/root/dev/
char core fd full hugepages log mqueue null ptmx pts random shm
stderr stdin stdout tty urandom zero
```

No nvidia nodes at all. `PrivateDevices` does not forbid the devices — it
mounts a fresh minimal `/dev` in which they were never created, and no cgroup
permission conjures a node that is not there. `BindPaths` is what works:

```ini
PrivateDevices=true
BindPaths=/dev/nvidia0:/dev/nvidia0
BindPaths=/dev/nvidiactl:/dev/nvidiactl
BindPaths=/dev/nvidia-uvm:/dev/nvidia-uvm
BindPaths=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools
DeviceAllow=/dev/nvidia0 rw
DeviceAllow=/dev/nvidiactl rw
DeviceAllow=/dev/nvidia-uvm rw
DeviceAllow=/dev/nvidia-uvm-tools rw
```

Then verify in the namespace rather than in the file:

```console
$ sudo ls -1 /proc/$(systemctl show my-agent -p MainPID --value)/root/dev/ | grep nvidia
nvidia0
nvidiactl
nvidia-uvm
nvidia-uvm-tools
```

Exposure stays at 4.0 OK with the binds in place.

---

## Troubleshooting

**The watchdog reports the engine is wedged, but it answers when you curl it.**
The probe cannot ask the question. Check that it reads an API key, and that the
model name it sends matches `/v1/models` — a greedy match returns the
permission object's id instead of the model's.

**The watchdog reports `skipped` forever and never probes.** It is looking for
a unit name that is no longer active, usually after a model swap. Derive the
active `vllm-*.service` instead of hardcoding it.

**CUDA fails only inside a systemd service.** Check the sandbox rather than the
driver: `ls /proc/$(systemctl show <unit> -p MainPID --value)/root/dev/`. If
`PrivateDevices=true` is set, the nvidia nodes need `BindPaths`.

**Out-of-memory errors while apparently within capacity.** Unified memory is
shared with the page cache; see the UMA note in
[vLLM for Inference](../vllm/README.md) for flushing the buffer cache.

> [!TIP]
> Test the effect, not the configuration. A unit test asserting the service
> file mentioned `DeviceAllow=/dev/nvidia0` passed for the entire time GPU work
> was broken. Both failures on this page were configuration that read correctly
> and did not work, found only by asking the running system what it was doing.
