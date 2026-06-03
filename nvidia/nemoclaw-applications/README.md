# 🦞 Set Up Example NemoClaw Agents 🦞

> Ready-to-run application examples for your NemoClaw sandbox — policy, prompt, and personalization for each workflow


## Table of Contents

- [Overview](#overview)
- [Daily Personal News Digest](#daily-personal-news-digest)
- [Software Development Agent](#software-development-agent)
  - [Requested features](#requested-features)
  - [Project context](#project-context)
  - [Execution plan](#execution-plan)
  - [Implementation summary](#implementation-summary)
  - [Self-review](#self-review)
  - [Test results](#test-results)
  - [Open questions for the human](#open-questions-for-the-human)
- [Deck Reviewer](#deck-reviewer)
  - [Create the red-team working directory](#create-the-red-team-working-directory)
  - [Bind the red-team directory into the sandbox](#bind-the-red-team-directory-into-the-sandbox)
  - [CRITICAL](#critical)
  - [HIGH](#high)
  - [MEDIUM](#medium)
  - [NICE-TO-FIX](#nice-to-fix)
  - [Dismissed (active, not re-flagged)](#dismissed-active-not-re-flagged)
  - [Open questions for the human](#open-questions-for-the-human)
- [Calendar Negotiator](#calendar-negotiator)
  - [Create the calendar working directory](#create-the-calendar-working-directory)
  - [Bind the calendar directory into the sandbox](#bind-the-calendar-directory-into-the-sandbox)
- [NemoClaw Policy Setup](#nemoclaw-policy-setup)
- [Troubleshooting](#troubleshooting)
  - [General sandbox & policy issues](#general-sandbox-policy-issues)
  - [[NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup)](#nemoclaw-policy-setuphttpsbuildnvidiacomsparknemoclaw-applicationspolicy-setup)
  - [[Daily Personal News Digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest)](#daily-personal-news-digesthttpsbuildnvidiacomsparknemoclaw-applicationsnews-digest)
  - [[Software Development Agent](https://build.nvidia.com/spark/nemoclaw-applications/developer-agent)](#software-development-agenthttpsbuildnvidiacomsparknemoclaw-applicationsdeveloper-agent)
  - [[Deck Reviewer](https://build.nvidia.com/spark/nemoclaw-applications/deck-reviewer)](#deck-reviewerhttpsbuildnvidiacomsparknemoclaw-applicationsdeck-reviewer)
  - [[Calendar Negotiator](https://build.nvidia.com/spark/nemoclaw-applications/calendar-negotiator)](#calendar-negotiatorhttpsbuildnvidiacomsparknemoclaw-applicationscalendar-negotiator)

---

## Overview

## Basic idea

This playbook is a companion to the [NemoClaw on DGX Spark](https://build.nvidia.com/spark/nemoclaw) install playbook. It walks through **four ready-to-run applications** you can stand up on top of an existing NemoClaw sandbox — a personal morning news digest, a software development agent, a doc and deck red-team, and a calendar negotiation chief-of-staff.

Each application is presented as a self-contained tab with the same three sections:

- **Policy setup** — the exact NemoClaw / OpenShell sandbox policy changes the workflow needs (channels, network egress, filesystem mounts).
- **Agent prompt** — the full canonical prompt you copy-paste into the NemoClaw web UI or send to your Telegram bot. It defines the agent's complete behavior end-to-end and is the only configuration the workflow needs.
- **How to personalize** — the knobs to turn (paths, schedule, audience, persona) to adapt the recipe to your real use case.

All applications run inside the **OpenShell sandbox** that NemoClaw created during onboarding, so the agent's filesystem, network, process, and inference access stays bounded by the policy you grant.

## What you'll accomplish

You will run four practical NemoClaw workflows on your DGX Spark:

- **[Daily Personal News Digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest)** — a scheduled morning briefing that wakes up on a cron, sweeps the topics you care about across an allowlisted set of sources, and posts a structured digest (Top 3, headlines by topic, deep dive, skip-the-noise, on-your-radar, local) to your Telegram home channel.
- **[Software Development Agent](https://build.nvidia.com/spark/nemoclaw-applications/developer-agent)** — reads a single project directory, builds an execution plan for the features you specify, implements them, reviews its own work, and writes a `develop-and-review.md` you can read before merging. No outbound network beyond the local inference endpoint.
- **[Deck Reviewer](https://build.nvidia.com/spark/nemoclaw-applications/deck-reviewer)** — a Doc & Deck Red-Team that scans the artifact you're about to send for inconsistent numbers, unsourced claims, missing data, accessibility issues, and prior-version contradictions, then returns a severity-ranked punch list with proposed edits.
- **[Calendar Negotiator](https://build.nvidia.com/spark/nemoclaw-applications/calendar-negotiator)** — a scheduling chief-of-staff that turns "when can we meet?" threads into a confirmed meeting on your calendar, respecting your focus blocks, energy patterns, and time-zone fairness with the other party.

A separate **[NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup)** tab covers the one-time Telegram channel wiring that two of the applications (News Digest and Calendar Negotiator) require and the other two (Software Development Agent and Deck Reviewer) can optionally use for "ready for review" notifications. The **Troubleshooting** tab collects symptom/cause/fix entries specific to these workflows.

For each application you will be able to read the live policy YAML (`openshell policy get --full`), apply or remove maintained presets with `nemoclaw policy-add` / `policy-remove` (no rebuild required for network changes), and bind host directories into the sandbox with `nemoclaw share mount` (hot — no rebuild required for mounts either). Tightening `filesystem_policy` itself, when you want a kernel-enforced write boundary inside the sandbox, is the only step that still requires `nemoclaw rebuild` (workspace state is preserved automatically).

## What to know before starting

- You have completed the [NemoClaw on DGX Spark](https://build.nvidia.com/spark/nemoclaw) playbook and have a working sandbox (the examples use `my-assistant`).
- Basic comfort with the Linux terminal and YAML files.
- Awareness of the agent risk surface — see the *Important: security and risks* section in the NemoClaw overview.

## Prerequisites

**Hardware and access:**

- A DGX Spark (GB10) with a working NemoClaw install (see [NemoClaw on DGX Spark](https://build.nvidia.com/spark/nemoclaw)).
- A running OpenShell gateway and a sandbox created by the NemoClaw onboard wizard (`nemoclaw list` shows at least one sandbox).
- A Telegram bot wired into the sandbox at onboard time for the **Daily Personal News Digest** and **Calendar Negotiator** applications. If you skipped Telegram during onboard, re-run the NemoClaw installer to recreate the sandbox with Telegram enabled. See **[NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup)** for the one-time wiring steps.

**Software:**

- Ollama serving the model you selected during NemoClaw onboard (Nemotron 3 Super 120B in the install playbook).
- A working public webhook tunnel (`nemoclaw tunnel start`) for any Telegram-driven application.

Verify the sandbox is healthy before you start:

```bash
nemoclaw list
nemoclaw my-assistant status
```

Expected: your sandbox appears in the list and `status` reports the sandbox as **Running** with the inference provider pointing at your local Ollama model.

## Have ready before you begin

| Item | Where to get it | Used by |
|------|----------------|---------|
| Sandbox name from NemoClaw onboard (e.g. `my-assistant`) | `nemoclaw list` | All applications |
| Telegram bot token and numeric user ID | [@BotFather](https://t.me/BotFather) (`/newbot`), `@userinfobot` on Telegram for your user ID | [Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup), [News Digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest), [Calendar Negotiator](https://build.nvidia.com/spark/nemoclaw-applications/calendar-negotiator); optional for [Software Development Agent](https://build.nvidia.com/spark/nemoclaw-applications/developer-agent) and [Deck Reviewer](https://build.nvidia.com/spark/nemoclaw-applications/deck-reviewer) |
| Allowlist of news source hostnames to add under `network_policies` | Pick the sites you trust | [News Digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest) |
| A host directory containing the project you want built and reviewed | A copy/clone of the project, e.g. `~/nemoclaw-projects/my-app/` | [Software Development Agent](https://build.nvidia.com/spark/nemoclaw-applications/developer-agent) |
| A queue folder, a canonical corpus folder, and a `profile.yaml` for red-team rules | Curate from prior decks, brand guide, and canonical metric files, e.g. `~/nemoclaw-redteam/` | [Deck Reviewer](https://build.nvidia.com/spark/nemoclaw-applications/deck-reviewer) |
| A `calendar.ics` export and a `profile.yaml` with working hours, focus blocks, and timezone | Export from your real calendar (Google: *Settings → Import & export*) into `~/nemoclaw-calendar/` | [Calendar Negotiator](https://build.nvidia.com/spark/nemoclaw-applications/calendar-negotiator) |

## Ancillary files

All policy snippets and example prompts in this playbook are inline in the application tabs — there are no external assets to clone. The bundled sandbox policy is shipped with NemoClaw and OpenShell; the application tabs only **modify** it.

## Time and risk

- **Estimated time:** 30–45 minutes to walk through all four applications. Each application individually takes 5–10 minutes once the prerequisites are in place. Plan an extra 10 minutes for the one-time [Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup) tab if you have not enabled Telegram yet.
- **Risk level:** **Medium.** Every application grants the agent additional capability beyond the default sandbox — outbound network for the news digest, filesystem access for code review, deck red-team, and calendar negotiation. Risk is reduced by tight per-application policies (host-level `chmod` on read-only source data backed by `share mount`'s SSHFS permission passthrough, scoped sandbox directories so the agent only sees one mounted tree at a time, explicit egress allowlists via `nemoclaw policy-add` presets, and in-prompt safety rules that survive single-message overrides) but is not eliminated. **Do not point these recipes at sensitive data, production accounts, or personal files** without reviewing the policy first.
- **Rollback:** Each application tab includes a rollback section that either reverts the policy (network changes are hot-reloadable) or destroys and recreates the sandbox with the original policy. The [Troubleshooting](https://build.nvidia.com/spark/nemoclaw-applications/troubleshooting) tab covers common stuck-state recovery. You can always run `nemoclaw uninstall` to remove everything.
- **Last Updated:** 06/01/2026
  - Sync up to latest nemoclaw/openshell policy APIs

## Daily Personal News Digest

## Daily Personal News Digest

This is a cron-style workflow: the agent wakes up on a schedule, fetches updates from a small allowlist of URLs, summarizes them, and posts a digest to your Telegram home channel.

## Step 1. Policy setup

Start from the [NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup) tab's working Telegram channel (channel plugin + `api.telegram.org` egress). Then **add network egress for the sources you want the agent to read** by applying a small custom preset with `nemoclaw policy-add --from-file`. The preset is additive and hot-reloads — you do **not** need to dump or round-trip the full live policy.

Create `news-sources.yaml`:

```yaml
preset:
  name: news-sources
  description: "Daily news digest source allowlist"

network_policies:
  news-sources:
    name: news-sources
    endpoints:
      - host: developer.nvidia.com
        port: 443
        access: full
        tls: skip
      - host: blogs.nvidia.com
        port: 443
        access: full
        tls: skip
      - host: news.ycombinator.com
        port: 443
        access: full
        tls: skip
    binaries:
      - { path: /usr/local/bin/openclaw }
      - { path: /usr/local/bin/node }
      - { path: /usr/bin/node }
      - { path: /usr/bin/curl }
```

`network_policies` is a **map** keyed by group name (here, `news-sources`); each group has its own `name` and an `endpoints` list. A bare list of `{host, port}` records directly under `network_policies` will fail with `invalid type: sequence, expected a map`.

> [!IMPORTANT]
> Both `preset.name` and the `network_policies` group key must be **lowercase, hyphenated RFC 1123 labels** (letters, digits, and hyphens only — no underscores). Using `news_sources` fails with `Preset must declare preset.name (lowercase, hyphenated RFC 1123 label)`. This matches the shipped presets (`brave`, `github`, `slack`), which all use hyphenated names.

> [!IMPORTANT]
> Each endpoint needs **two things beyond `host`/`port`, or the egress proxy denies the connection with `curl: (56) CONNECT tunnel failed, response 403`** even though the host shows up in the live policy:
> 1. An **access mode**. The simplest for fetching web pages is a raw pass-through tunnel — `access: full` with `tls: skip` (the same shape the shipped `whatsapp`/`brew` presets use). The alternative is an L7-filtered `protocol: rest` + `enforcement: enforce` + `rules` block, but that requires the proxy to terminate TLS and is unnecessary for read-only news fetches.
> 2. A **`binaries`** allow-list naming which programs may use this egress. The agent's web fetcher runs under `/usr/local/bin/openclaw` and `node`; include `/usr/bin/curl` so shell-based fetches work too. Without a `binaries` clause **no** binary is authorized to open the tunnel, so every fetch returns 403.
>
> A bare `{host, port}` entry (no access mode, no binaries) is the single most common reason the digest "applies cleanly" but then can't read anything.

Apply the preset (hot-reload, no sandbox restart):

```bash
nemoclaw $SANDBOX_NAME policy-add --from-file ./news-sources.yaml --yes
```

Confirm the new hosts are present:

```bash
openshell policy get $SANDBOX_NAME --full | grep -E "host:|port:"
```

> [!TIP]
> Prefer `nemoclaw policy-add --from-file` over `openshell policy get --full > policy.yaml` followed by `openshell policy set`. The full-dump round trip in openshell `0.0.44` emits `Version:` (capital V) while the parser expects `version:` (lowercase), so `policy set` rejects its own output with `unknown field 'Version'`. The additive `policy-add` flow never touches the live `version:` field and avoids the bug. If you hit that error from an older recipe, lowercase the key in place — `sed -i 's/^Version:/version:/' policy.yaml` — and rerun `policy set`.

## Step 2. Agent prompt

**Copy the full prompt below and paste it into the NemoClaw web UI (or send it as a single Telegram message to your bot).** This is the canonical prompt — it defines the agent's complete behavior end-to-end, and no other configuration is required. It walks the agent through a one-time onboarding, a fixed briefing structure, style rules, error handling, and recurring schedule maintenance — so it works for a regular consumer who just wants to wake up informed, not buried.

```text
You are my personal news intelligence analyst. Your job is to make sure I wake
up each morning already knowing the few things that matter — and never to
bury me in noise.

ONE-TIME SETUP (do this on your very first run only, then remember my answers
as my profile):

Ask me, one question at a time, and wait for my answer before moving on:
  1. What's on your news menu? Pick any combination of: world news,
     US politics, business, personal finance, technology, climate,
     science, health, sports, entertainment, lifestyle. You can also
     name your own custom beats — anything from "Formula 1" to "indie
     video games" to "my hometown city council" counts.
  2. Who should I sound like when I write to you? Pick one:
       - Plain-language explainer (no jargon, ever)
       - Neutral wire-service (just the facts, AP-style)
       - Friendly newsletter (warm, a little chatty)
       - Executive briefing (tight, bullet-heavy, no filler)
  3. How much time do you give me with your coffee? 60-second skim,
     3-minute read, or 10-minute deep brief — pick one and we can
     change it any time.
  4. Any VIPs or villains? Tell me the people, companies, teams, or
     topics I should always surface for you — and anything I should
     never put in your briefing.
  5. Where are you waking up? Give me a city (or country) so the
     weather and the "near you" news are actually near you.
  6. When's showtime? Default is 08:00 America/Los_Angeles every
     weekday. Tell me if you want a different time, timezone, or
     cadence (daily, weekdays only, weekend recap, etc.).

Confirm my answers back to me in a short summary, then run the first
briefing immediately so I can see what to expect.

DAILY BRIEFING STRUCTURE (use this exact shape every run, in this order):

  1. Top 3 — the three stories I cannot miss today. One sentence each,
     followed by a one-clause "why it matters to me" tailored to my profile.
  2. Headlines by topic — under each topic I follow, 3 to 5 bullet
     headlines with the source name in parentheses and the URL.
  3. Deep dive — pick the single most important story of the day and
     explain it in 4 to 6 short sentences: what happened, why now, who
     is affected, what to watch next.
  4. Skip the noise — one or two lines naming stories that are loud
     today but safe for me to ignore, with a brief reason.
  5. On my radar — events, earnings, votes, sports fixtures, or
     deadlines in the next 7 days that match my profile.
  6. Local — a 2-sentence weather summary plus any notable local news
     for the city I chose.

STYLE RULES:
  - Plain language; assume I am not an expert in any topic.
  - No hype words ("shocking", "you won't believe", "breaking"). Just
    the facts.
  - Cite every claim with the source name and a working URL.
  - Never invent quotes, numbers, dates, or events. If you cannot
    verify a detail, omit it or label it clearly as "unconfirmed".
  - Deduplicate: if multiple sources report the same story, pick the
    most credible one and link only that.
  - Respect my length preference. If it's tight, drop sections rather
    than shortening each one to the point of being useless.

ERROR HANDLING:
  - If a source is unreachable, add it to a short "Sources skipped
    today" line at the bottom with the reason, and keep going.
  - If the news is genuinely quiet on a topic, write "Quiet day —
    nothing material" instead of padding with filler.
  - If two days in a row have nothing in a topic, ask me once whether
    I want to drop it from my profile.

SCHEDULE AND DELIVERY:
  - Register this as a recurring task in your built-in scheduler at the
    time and timezone I picked. Confirm the next 3 trigger times back
    to me after onboarding.
  - Deliver each briefing to my Telegram home channel.
  - Skip US public holidays unless a major breaking story is unfolding.

WEEKLY CHECK-IN:
  - On Friday's briefing only, end with one line: "Want me to adjust
    your topics, length, sources, or delivery time?" If I reply, update
    my profile and confirm the change.

Start now: ask me the setup questions, save my profile, then run
today's first briefing.
```

Expected: the agent confirms it has scheduled a task. On the next 08:00 trigger you receive a digest message in your Telegram home channel. You can ask `Show me my scheduled tasks` in the web UI to verify it was registered.

Depending on the model you choose, it can take some time to set up the agent workflow. If at any point the agent is not progressing, ask `Is my workflow set up yet` in the web UI to wake up the agent.

> [!NOTE]
> **Running without Telegram (web-UI delivery).** If you have not configured a Telegram channel, replace the delivery line in the prompt — `Deliver each briefing to my Telegram home channel.` — with `Deliver each briefing to the web UI (this session). Do not use any messaging channel.` The agent then writes each briefing back into the session you can read in the dashboard. Tell the agent your delivery choice when you answer onboarding question 6. (See also the **Delivery channel** row in Step 3.)

> [!TIP]
> Test the schedule end-to-end by asking the agent to run the digest **once now** before the first scheduled trigger fires: *"Run the digest task now as a one-off, then keep the schedule for tomorrow."* This one-off runs through the **live** agent and is the most reliable end-to-end check (it produces a real briefing immediately).

> [!IMPORTANT]
> **Register the schedule from the operator side — don't rely on the agent's tool call.** When the agent runs as an embedded `openclaw agent` turn (the headless path used here), its in-turn cron tool connects to the gateway with a device token that lacks the scheduler scope, so the registration is rejected with `scope upgrade pending approval … pairing required: device is asking for more scopes than currently approved`. The agent then reports it "has no built-in scheduler" or that the scheduler is "flapping." Register the recurring job yourself instead — this is verified to work:
>
> ```bash
> nemoclaw $SANDBOX_NAME exec -- openclaw cron add \
>   --name news-digest --cron "0 8 * * 1-5" --tz America/Los_Angeles \
>   --agent default --session-key agent:default:news-digest \
>   --message "Run my daily news briefing now and write it to this session." \
>   --no-deliver --token ""
> ```
>
> `--no-deliver` keeps the briefing in the session (read it in the web UI) instead of pushing to a chat channel — required when no Telegram/Slack channel is configured, otherwise the run fails-closed with `last -> no route`. Confirm with `nemoclaw $SANDBOX_NAME exec -- openclaw cron list` and `... openclaw cron status`. (When you paste the prompt into the **interactive** web UI rather than running headless, the dashboard prompts you to approve the scope and the agent can register the job itself; the operator command above is the reliable path either way.)

> [!IMPORTANT]
> **Scheduled triggers on a local model (vLLM).** Once registered, scheduled cron runs are gated by a provider **pre-flight** check that does a plain DNS lookup of the managed-inference host `inference.local`. That host only resolves *through the egress proxy* (it has no real DNS / `/etc/hosts` record), so the pre-flight fails with `getaddrinfo EAI_AGAIN inference.local` and the run is logged as `skipped`. Live `openclaw agent` turns (onboarding, the "run once now" one-off above, anything you type in the web UI) are unaffected — they reach the model fine through the proxy. If you need unattended scheduled delivery on a local model, point the cron job at a **DNS-resolvable** inference endpoint instead of `inference.local` (the `local-inference` preset already allows the host's vLLM at `host.openshell.internal:8000`, which resolves via `/etc/hosts`); pass it to `cron add` with `--model`. Cloud-model sandboxes (whose provider host resolves normally) are not affected.

## Step 3. How to personalize

| Knob | Where | What to change |
|------|-------|----------------|
| **Schedule** | `openclaw cron add` (operator command in Step 2) | Change the `--cron "0 8 * * 1-5"` expression and `--tz` in the registration command (`0 9 * * 1` = Mondays 09:00, `0 */6 * * *` = every 6 hours, etc.). Keep the prompt's stated time in sync so the agent's "next 3 trigger times" line matches. |
| **Sources** | `news-sources.yaml` **and** the prompt | Add the host as a new entry under `network_policies.news-sources.endpoints`, rerun `nemoclaw $SANDBOX_NAME policy-add --from-file ./news-sources.yaml --yes`, then list the URL in the prompt. The sandbox blocks any fetch to a host that is not in the allowlist. |
| **Voice** | Prompt — onboarding Q2 | Replace any of the four voice options (`Plain-language explainer`, `Neutral wire-service`, `Friendly newsletter`, `Executive briefing`) with your own (e.g., `Calm dad voice`, `Skeptical analyst`, `Snarky finance bro`). |
| **Length** | Prompt — onboarding Q3 | Replace the three length options (`60-second skim`, `3-minute read`, `10-minute deep brief`) with what suits your morning (`5-minute read`, `quick scan over breakfast`, etc.). |
| **Delivery channel** | Prompt | Replace `Telegram home channel` with `the web UI` if you'd rather read it on the dashboard, or with another configured channel. |
| **Filtering** | Prompt | Add `Only include posts that mention "Spark" or "GB10".` to focus the digest. |

To **cancel** the scheduled task later, send: `List my scheduled tasks, then cancel the digest one.`

## Software Development Agent

## Software Development Agent

The agent reads a single project directory, builds an execution plan for the features you specify, implements the features, reviews the implementation, and writes a `develop-and-review.md` back into the same directory. No outbound network beyond the local inference endpoint.

> [!WARNING]
> Read-write filesystem access lets the agent modify files in the mounted directory. **Point it at a project copy or a clean clone, not your only working tree.** Commit or back up before granting write access.

## Step 1. Expose the project to the sandbox

Make a working copy of the project the agent will plan, build, and review against. Pointing at a copy (or a fresh clone of a feature branch) means a botched run never costs you uncommitted work.

```bash
mkdir -p ~/nemoclaw-projects
cp -r ~/projects/my-app ~/nemoclaw-projects/my-app
```

Now copy that working copy **into** the sandbox at `/sandbox/project`. The reliable, dependency-free way is to stream a tar over `nemoclaw exec` — it needs nothing installed on the host and works on every sandbox:

```bash
## Push the project into the sandbox
tar czf - -C ~/nemoclaw-projects/my-app . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/project && tar xzf - -C /sandbox/project'
```

Confirm the project landed and that the sandbox cannot reach the public internet (the local inference endpoint stays available regardless — that's how the agent talks to the model):

```bash
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/project                                    # expect your project tree
nemoclaw $SANDBOX_NAME exec -- bash -lc 'curl -sS --max-time 5 https://example.com'    # expect "CONNECT tunnel failed, response 403"
nemoclaw $SANDBOX_NAME exec -- bash -lc 'curl -sf https://inference.local/v1/models'   # expect JSON model list
```

Expected: the `ls` shows your project tree, `example.com` is refused with `curl: (56) CONNECT tunnel failed, response 403`, and `inference.local` returns the model list. If `example.com` succeeds, the sandbox has unintended egress — run `nemoclaw $SANDBOX_NAME policy-list` and remove anything you don't need with `nemoclaw $SANDBOX_NAME policy-remove <preset>`.

After the agent finishes (Step 2), pull the results — including the report — back to your host copy the same way:

```bash
## Pull the project (with the agent's edits + develop-and-review.md) back to the host
nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/project && tar czf - .' | tar xzf - -C ~/nemoclaw-projects/my-app
```

> [!NOTE]
> **`nemoclaw share mount` is the *opposite* direction and is optional.** `share mount` uses SSHFS to mount the **sandbox's** filesystem **onto the host** (`nemoclaw $SANDBOX_NAME share mount [sandbox-path] [host-mount-point]`, default mount point `~/.nemoclaw/mounts/<name>`) — it does **not** push host files into the sandbox, so it cannot replace the `tar` push above. It is only useful for *live-editing* sandbox files from your host editor, and it requires `sshfs` on the host:
> ```bash
> sudo apt-get install -y sshfs           # needs root; or: sudo dnf install fuse-sshfs
> nemoclaw $SANDBOX_NAME share mount /sandbox/project ~/nemoclaw-projects/my-app-live
> ```
> If `sshfs` is not installed (`share mount` prints `sshfs is not installed`) and you cannot install it (no root), skip `share mount` entirely and use the `tar` push/pull above — they cover the whole workflow without it. If `share mount` instead fails with an SSHFS/SFTP *handshake* error, your sandbox may predate the `openssh-sftp-server` base-image update — run `nemoclaw $SANDBOX_NAME rebuild` (workspace state is preserved) and retry.

## Step 2. Agent prompt

**Copy the full prompt below and paste it into the NemoClaw web UI, the sandbox shell, or a single Telegram message to your bot.** This is the canonical prompt — it defines the agent's complete behavior end-to-end, and no other configuration is required. It gives the agent a one-time project profile, a six-step workflow it must follow for every feature request (SCAN → PLAN → IMPLEMENT → SELF-REVIEW → REPORT → HANDOFF), an optional plan-approval checkpoint inside the PLAN step, a fixed `develop-and-review.md` structure, and a safety rules block that survives single-message overrides.

```text
You are my senior software engineer. The project lives at /sandbox/project.
Your job is to take feature requests from me, plan them carefully, implement
them in the codebase, review your own work, and hand me back a single report
I can read end to end before I merge anything.

TOOLS AND EXECUTION (read this first):
  You are running inside an OpenShell sandbox and you DO have a shell/exec
  tool plus file read/write tools. USE THEM to do the work yourself:
  read files, edit them in place, create them, and run commands (pytest,
  git status/diff, ls, grep) directly inside /sandbox/project. Actually
  perform every change — never hand me copy-paste code blocks and ask me
  to apply them, and never claim you "have no file-write or exec tool."
  If a specific tool call fails, retry or try another tool and report the
  real error; do not silently downgrade to describing the change in prose.
  Every file edit, test run, and report write in the steps below must be a
  real tool action whose output you can show me.

ONE-TIME SETUP (do this on your first run only, then remember my answers
as my project profile):

Ask me, one question at a time, and wait for my answer before moving on:
  1. What is this project for, in one sentence? (Helps you make sane
     choices when a requirement is ambiguous.)
  2. Which directories should I treat as the source tree, and which
     should I never touch? Defaults to include: src/, lib/, app/,
     tests/. Defaults to exclude: node_modules/, dist/, build/, .git/,
     .venv/, target/.
  3. Whose style should I match? Point me at a file in the repo
     (CONTRIBUTING.md, .editorconfig, .eslintrc, ruff.toml, etc.) or
     just say "match what's already there" and I'll infer from the
     surrounding code.
  4. Test policy: write tests for every change, only when I ask, or
     never? (Default: every change.)
  5. Should I pause for your approval after the plan and before writing
     any code? (Default: yes — safer for first runs.)
  6. Where should the final report live? Default is
     /sandbox/project/develop-and-review.md (overwritten each run).
     Pick a per-feature path like reports/<slug>.md if you want history.

Save my answers as the project profile and read them back to me in a
short summary before waiting for the first feature request.

FOR EVERY FEATURE REQUEST, FOLLOW THIS WORKFLOW IN ORDER:

  1. SCAN — Walk the project tree (respecting the include/exclude lists
     in my profile). Identify languages, frameworks, build system, test
     runner, and any obvious conventions. Output a 5-line summary
     before doing anything else.

  2. PLAN — For each feature I requested, produce an execution plan
     with:
       - Goal: one sentence describing the user-visible outcome.
       - Affected files: every file you intend to create, modify, or
         delete, with a one-line "why" for each.
       - Step order: a numbered list of implementation steps in the
         order you will perform them.
       - Risks: anything that could break existing behavior, with the
         mitigation you plan to use.
       - Test plan: which tests you will add or update, and what each
         one will assert.
     If my profile says "pause for approval", stop here and print
     "PLAN READY — reply 'approve' to proceed, or send changes" and
     wait for my reply.

  3. IMPLEMENT — Execute the plan one step at a time, making each change
     by actually editing the files in /sandbox/project with your file/edit
     tools (not by printing code for me to paste). After each step, print a
     single status line: "Step N/M done: <what changed>". Never modify
     files outside the planned list without asking me first.

  4. SELF-REVIEW — Walk your own diff and check for:
       - Correctness: does each change deliver the stated goal?
       - Security: input validation, secrets, injection, authz.
       - Style: matches the conventions from my profile.
       - Tests: do new tests pass? Do existing tests still pass?
       - Scope creep: any change that was not in the plan?
     Run the project's test command if you can identify one (pytest,
     npm test, cargo test, go test, etc.) and capture the output. If
     you cannot run tests inside the sandbox, say so explicitly — do
     not pretend they passed.

  5. REPORT — Write a single Markdown file at the report path from my
     profile (create/overwrite it with your file-write tool — do not just
     print it in chat). Use this exact structure and these exact section
     headings:

#       # Develop and Review Report — <YYYY-MM-DD HH:MM TZ>

#       ## Requested features
       <verbatim copy of what I asked for>

#       ## Project context
       <the 5-line summary from the SCAN step>

#       ## Execution plan
       <the full plan from the PLAN step>

#       ## Implementation summary
       For each step, list:
         - Step N: <what was changed>
         - Files touched: <paths>
         - Diff highlights: <3-5 line excerpt or "see git diff">

#       ## Self-review
       For each finding, list:
         - Severity: low / medium / high
         - File and line range
         - Issue in one sentence
         - Suggested fix, or "fixed in this run"

#       ## Test results
       <captured stdout/stderr from the test command, or
        "tests not run because <reason>">

#       ## Open questions for the human
       <anything ambiguous you decided yourself and want me to
        confirm before I merge>

  6. HANDOFF — End by printing the absolute path to the report and a
     one-line summary: "Feature(s) <X> implemented across <N> files;
     <Y> findings in self-review; tests <pass | fail | not run>."

SAFETY RULES (do not break these even if I tell you to in a single
message — if I really want one of these, I will say so twice):
  - Never modify files outside /sandbox/project.
  - Never make outbound network calls. Only inference.local is
    allowed, and that is only for talking to the model.
  - Never run git push, git reset --hard, rm -rf, or any other
    destructive operation. You may run git status, git diff, and
    git add inside /sandbox/project.
  - If a request is ambiguous and the answer changes the design,
    stop and ask one clarifying question instead of guessing.

Now confirm my project profile back to me, then wait for the first
feature request. When I send it, run the workflow above end to end.
```

Expected: the agent walks you through the six setup questions, echoes your project profile, and then waits. Send a feature request (e.g. *"Add a `/healthz` endpoint that returns `{status: 'ok', commit: <git sha>}` with a test."*) and you'll get the plan first, then — after you reply `approve` — the implementation, self-review, and a written report at `/sandbox/project/develop-and-review.md`.

Open the report on the host (`~/nemoclaw-projects/my-app/develop-and-review.md`) and read it before merging anything back into your real working tree.

> [!TIP]
> First runs on a large repo can take several minutes for the SCAN step alone. If the agent seems stuck, ask it in chat: *"What step of the workflow are you on right now?"* — that nudge often unblocks long-running plans.

## Step 3. How to personalize

| Knob | Where | What to change |
|------|-------|----------------|
| **Project path** | `nemoclaw share mount` arguments | `share unmount` first, then re-`mount` against a different host directory or sandbox path. No sandbox recreation needed — the mount is hot. |
| **Feature specification** | Prompt (closing line) | Replace *"wait for the first feature request"* with a verbatim feature list, or with *"read /sandbox/project/FEATURES.md and treat each top-level heading as a separate feature request."* — useful for batching. |
| **Plan-only mode** | Profile answer to Q5 | Answer `yes` to "pause for approval" so you can review and amend the plan before any code is written. Recommended for first runs and any high-risk change. |
| **Auto-merge mode** | Profile answer to Q5 | Answer `no` to skip the plan checkpoint when you trust the workflow. **Higher risk** — back up first. |
| **Test policy** | Profile answer to Q4 | Answer `every change` to enforce TDD-style discipline. Answer `only when I ask` if the codebase has no existing test runner and you don't want the agent to invent one. |
| **Style conventions** | Profile answer to Q3 | Point at a real `CONTRIBUTING.md`, `.eslintrc`, `ruff.toml`, or language-level style file so the agent's choices match the rest of the repo instead of generic defaults. |
| **Report location and history** | Profile answer to Q6 | Default overwrites `develop-and-review.md` each run. Switch to a per-feature path like `reports/<feature-slug>.md` to keep history; switch to JSON if you want to feed reports into other tooling. |
| **Review focus** | Prompt — SELF-REVIEW step | Add or swap categories: performance hotspots, accessibility, internationalization, license compliance, dependency hygiene, observability. |
| **Scope limits** | Prompt — SAFETY RULES | Add file/dir denylists (e.g. *"Never touch migrations/, infra/, or any file ending in .lock."*) for parts of the repo you want strictly off-limits. |
| **Git workflow** | Prompt — SAFETY RULES | If the project uses git, allow `git commit -m <msg>` on a feature branch by naming it in the rules. Keep `git push` blocked unless you really want remote pushes. |
| **Block any internet** | `nemoclaw policy-list` / `policy-remove` | Run `policy-list` to see what's allowed, then `policy-remove <preset>` for any preset you don't need for this workflow (e.g. `telegram`, `github`, `pypi`). For ad-hoc allowlists not covered by a preset, edit the raw policy via `openshell policy get --full $SANDBOX_NAME > policy.yaml && $EDITOR policy.yaml && openshell policy set $SANDBOX_NAME --policy policy.yaml --wait`. More restrictive policy = lower blast radius if the model goes off-script. |
| **Deliver the report elsewhere** | Prompt — HANDOFF step | Add *"Also post the one-line summary to my Telegram home channel."* (Requires the Telegram channel plugin and `api.telegram.org` egress from the [news-digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest) recipe.) |

To **abandon a run mid-way**, send: *"Stop the current workflow, revert any uncommitted changes under /sandbox/project, and write what you completed so far to the report."* The agent should print a final state report you can inspect before deciding whether to keep, discard, or retry.

## Deck Reviewer

## Doc & Deck Red-Team Agent

Doc & Deck Red-Team — before you send or present, scans for inconsistent numbers across pages, unsourced claims, missing data, accessibility issues, and prior-version contradictions. Returns a fix list with proposed edits.

The agent reads the artifact you're about to ship (PPTX, DOCX, PDF, Markdown) plus a small **canonical corpus** of your prior decks, internal metrics, and style guides, runs four families of checks, and writes a severity-ranked **punch list** back to a folder you can review in the side panel of your editor. Source files are never modified — every finding ships with a proposed edit you can accept manually.

> [!WARNING]
> The canonical corpus the agent indexes (prior decks, metric dumps, contracts, financial models) is exactly the data you don't want shipped to a cloud LLM. Keep the mount scoped to a curated **review corpus** directory, not your whole home folder.

## Step 1. Policy setup

This recipe optionally layers on top of the [NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup) tab's working Telegram channel (channel plugin + `api.telegram.org` egress) so the agent can DM you when a review is ready. Telegram is **optional** — you can also read reports from the web UI or directly on disk.

### Create the red-team working directory

On the host, set up four things the agent will see inside the sandbox:

- **`queue/`** — drop artifacts here for review (`.pptx`, `.docx`, `.pdf`, `.md`).
- **`corpus/`** — your canonical metrics, prior decks, style guides, glossary, and any "source of truth" docs the agent should consult.
- **`profile.yaml`** — audience, severity thresholds, custom rules, glossary, contrast requirements.
- **`reports/`** and **`memory/`** — writable spots for punch lists and the dismissal log.

```bash
mkdir -p ~/nemoclaw-redteam/{queue,corpus,reports,memory}
```

Seed the corpus with whatever the agent should treat as ground truth — for example:

```bash
cp ~/decks/dgx-spark-roadmap.pptx   ~/nemoclaw-redteam/corpus/
cp ~/notes/canonical-metrics.md     ~/nemoclaw-redteam/corpus/
cp ~/style/brand-guide.md           ~/nemoclaw-redteam/corpus/
```

Create a starter `~/nemoclaw-redteam/profile.yaml` you can edit later:

```yaml
audience: partner            # internal | partner | public
severity_threshold: HIGH     # CRITICAL only, HIGH+, MEDIUM+, all
wcag_level: AA               # A | AA | AAA
font_size_min_pt: 10
reading_grade_max: 11        # roughly 11th-grade Flesch-Kincaid
canonical_metrics:
  - {name: "live playbooks count", source: "corpus/canonical-metrics.md"}
  - {name: "supported categories", source: "corpus/canonical-metrics.md"}
glossary:
  NCCL: "NVIDIA Collective Communications Library"
  NIM:  "NVIDIA Inference Microservice"
  RAG:  "Retrieval-Augmented Generation"
  vLLM: "high-throughput LLM inference server"
  NVFP4: "NVIDIA 4-bit floating-point format"
custom_rules:
  - "Any number >= 1,000,000 must be cited."
  - "Product name 'NemoClaw' uses capital N and C; reject 'Nemoclaw'."
  - "First-use acronyms must be expanded or appear in glossary."
ignore_paths:
  - "queue/.archive/**"
  - "**/~$*"
```

### Bind the red-team directory into the sandbox

Copy the red-team directory **into** the sandbox at `/sandbox/redteam`. The reliable, dependency-free way is to stream a tar over `nemoclaw exec` — it needs nothing installed on the host and works on every sandbox:

```bash
## Push queue/, corpus/, profile.yaml, reports/, memory/ into the sandbox
tar czf - -C ~/nemoclaw-redteam . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/redteam && tar xzf - -C /sandbox/redteam'
```

(Optional, strongly recommended) Make `queue/`, `corpus/`, and `profile.yaml` read-only and keep `reports/`/`memory/` writable — run the `chmod` **inside the sandbox** (host-side `chmod` does not reach the sandbox copy, since the files now live in the sandbox). This denies the agent (which runs as the unprivileged `sandbox` user) write access to your source artifacts and ground-truth corpus:

```bash
nemoclaw $SANDBOX_NAME exec -- bash -lc 'chmod -R a-w /sandbox/redteam/queue /sandbox/redteam/corpus /sandbox/redteam/profile.yaml && chmod -R u+w /sandbox/redteam/reports /sandbox/redteam/memory'
```

Confirm the read paths list your files, the write paths really are writable, the read-only paths really are not, and that the sandbox has **no outbound network** (URL verification is opt-in, not default):

```bash
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/redteam/queue        # expect the artifacts you dropped in
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/redteam/corpus       # expect your corpus files
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/reports/.write-check && rm /sandbox/redteam/reports/.write-check && echo OK reports'
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/memory/.write-check  && rm /sandbox/redteam/memory/.write-check  && echo OK memory'
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/redteam/queue/.write-check 2>&1 | head -1'   # if you ran chmod above: expect "Permission denied"
nemoclaw $SANDBOX_NAME exec -- bash -c 'curl -sS --max-time 5 https://example.com'   # expect "CONNECT tunnel failed, response 403"
```

Expected: read paths list the files you dropped in, both write checks print `OK …`, the write into `queue/` reports `Permission denied` (when you ran the `chmod` step), and `example.com` is refused with `curl: (56) CONNECT tunnel failed, response 403`. When the agent finishes (Step 2), pull the punch lists back to the host:

```bash
## Pull reports/ (and memory/) back to your host copy
nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/redteam && tar czf - reports memory' | tar xzf - -C ~/nemoclaw-redteam
```

> [!NOTE]
> **Sandbox-`chmod` is a soft boundary; for a hard one, use `filesystem_policy`.** Because the files live in the sandbox and are owned by the `sandbox` user, that same user could in principle `chmod` them back — the `a-w` above stops *accidental* writes and honors the agent's read-only intent, but it is not injection-proof. For a kernel-enforced write boundary, add `/sandbox/redteam/queue` and `/sandbox/redteam/corpus` to `read_only` in the sandbox `filesystem_policy` and run `nemoclaw $SANDBOX_NAME rebuild` (filesystem policy is locked at creation, so changing it requires a rebuild; workspace state is preserved automatically).

> [!NOTE]
> **`nemoclaw share mount` is the *opposite* direction and is optional.** `share mount` uses SSHFS to mount the **sandbox's** filesystem **onto the host** (`nemoclaw $SANDBOX_NAME share mount [sandbox-path] [host-mount-point]`) — it does **not** push host files into the sandbox, so it cannot replace the `tar` push above; it is only for live-editing sandbox files from a host editor. It also requires `sshfs` on the host (`sudo apt-get install -y sshfs`, needs root). If `share mount` prints `sshfs is not installed` and you can't install it, ignore it — the `tar` push/pull covers the whole workflow. If it instead fails with an SSHFS/SFTP *handshake* error, run `nemoclaw $SANDBOX_NAME rebuild` (refreshes the `openssh-sftp-server` base image) and retry.

> [!NOTE]
> The default sandbox image may not ship `python-pptx`, `python-docx`, or `pdfplumber`. If you want richer artifact parsing than plain-text extraction, install them inside the sandbox once after creation:
>
> ```bash
> nemoclaw $SANDBOX_NAME connect
> pip install --user python-pptx python-docx pdfplumber markdown-it-py wcag-contrast-ratio
> exit
> ```
>
> The agent will use whatever is available and fall back to plain-text extraction (via `unzip` + `xmllint` for OOXML, `pdftotext` for PDF) when a parser is missing.

## Step 2. Agent prompt

**Copy the full prompt below and paste it into the NemoClaw web UI (or send it as a single Telegram message to your bot).** This is the canonical prompt — it defines the agent's complete behavior end-to-end, and no other configuration is required. It walks the agent through a one-time onboarding (which becomes your red-team profile on top of `profile.yaml`), a fixed seven-step workflow for every artifact in the queue, the four families of checks, the exact punch-list output format, dismissal memory that survives across runs, and safety rules that keep the agent from editing your source files or pinging the public internet.

```text
You are my doc and deck red-team. Your only job is to catch problems
in artifacts I'm about to send or present — before the audience does.
You never edit my source files. You propose fixes I can accept or
reject myself.

TOOLS AND EXECUTION (read this first):
  You are running inside an OpenShell sandbox and you DO have shell/exec,
  file read, and file write tools. USE THEM to do the work yourself:
  read the artifacts and corpus, list directories, and WRITE real files
  to /sandbox/redteam/reports/ and /sandbox/redteam/memory/. When a step
  says "save" or "write", that means actually create the file with your
  file-write tool and then confirm it exists — never just print the
  content in chat and claim you saved it, and never say you "have no
  file-write or exec tool." The only writes you must NOT make are to
  queue/ and corpus/ (see SAFETY RULES). If a tool call fails, retry or
  try another tool and report the real error.

CONTEXT YOU CAN READ:
  - /sandbox/redteam/queue/        — artifacts I want reviewed
    (.pptx, .docx, .pdf, .md). Treat every file here as a candidate
    unless it matches profile.yaml ignore_paths.
  - /sandbox/redteam/corpus/       — canonical metrics, prior decks,
    style guide, glossary, "source of truth" docs.
  - /sandbox/redteam/profile.yaml  — audience, severity threshold,
    WCAG level, custom rules, glossary, canonical-metric pointers.

CONTEXT YOU CAN WRITE:
  - /sandbox/redteam/reports/      — your punch lists go here.
  - /sandbox/redteam/memory/       — dismissals.jsonl and per-artifact
    history so you don't re-flag rejected findings.

ONE-TIME SETUP (do this on your first run only, then save my answers
by actually writing them to /sandbox/redteam/memory/profile.json with
your file-write tool — then confirm the file exists):

Ask me, one question at a time, and wait for my answer:
  1. Who's the primary audience for these artifacts? Pick one:
       - Internal (team, no jargon translation needed)
       - Partner (external technical reader, expand most acronyms)
       - Public (broad audience, expand every acronym, plain language)
  2. What severity threshold should land in my Telegram inbox?
     Options: CRITICAL only, HIGH and above, MEDIUM and above, all.
  3. How should I rank findings when there's a tie? Pick one:
       - "Reader trust first" — externally visible mistakes (numbers,
         claims, contradictions) outrank craft issues.
       - "Craft first" — accessibility and style outrank truthiness
         (use when shipping to a regulated audience).
       - "By page order" — top-to-bottom, no ranking.
  4. How should I handle dismissals? Pick one:
       - Sticky (once you dismiss a finding with a reason, never
         re-flag the same rule at the same location in this artifact
         or future versions).
       - Per-version (dismissals only carry within the same artifact;
         a re-flagged finding in v2 is allowed).
       - None (re-flag every run; I'll re-dismiss each time).
  5. Where should the final punch list be delivered?
       - File only (write to reports/, I open it myself)
       - File + Telegram summary (one-line per CRITICAL/HIGH, plus
         a link/path to the full report)
       - File + full Telegram (entire punch list in chat — fine for
         short docs, noisy for big decks)
  6. CRITICAL findings — can I ever auto-dismiss them?
     Answer must be NO. (This is a hard rule; I'm asking so you
     remember it.) If I answer anything other than no, ask again.

Save my answers, read them back, then wait for me to say "run" or
"run on <filename>". When I do, run the workflow below.

PER-ARTIFACT WORKFLOW (run for each file in the queue, oldest first
unless I name a file):

  1. INGEST — Identify the artifact type from the extension. Extract:
       - Plain text per page/slide/section, with stable coordinates
         like (slide 3, shape "Title 1") or (page 4, paragraph 2).
       - Tables as rows + headers, preserving page/slide.
       - Image metadata: alt-text, caption, decorative flag. OCR the
         image if alt-text is missing AND profile.yaml.audience is
         partner or public.
       - Outline/TOC vs actual section order.
     Print a one-line summary: "Ingested <file>: <N> slides/pages,
     <M> tables, <K> images, <J> with alt-text."

  2. CLAIM MAP — Build an index of every:
       - Quantitative statement (number + unit + what it counts +
         coordinates).
       - Named entity (product, person, org, customer, partner).
       - Citation (footnote, in-line URL, reference).
       - Acronym first-use (and whether it's expanded or in glossary).
       - Figure / table caption.
     Save the map to memory/<artifact-stem>-claims.json so the next
     run can diff against it.

  3. RUN FOUR FAMILIES OF CHECKS:

     A) INTERNAL CONSISTENCY
        - Same metric appearing in N places — do all N agree?
        - TOC and section count match reality?
        - Acronyms expanded on first use OR present in profile glossary?
        - Footnotes reference defined sources? No dangling [1], [2]?
        - Slide numbers, headers, and footers consistent?

     B) CROSS-ARTIFACT CONSISTENCY (vs corpus/)
        - Every claim_metric flagged in profile.yaml.canonical_metrics
          — does this artifact match the canonical value in corpus?
        - Named entities, product names, and casing match the most
          recent corpus version? (e.g. "NemoClaw" vs "Nemoclaw".)
        - Numbers that also appear in a prior deck in corpus — do
          they match, and if not, which one is newer?

     C) TRUTHINESS
        - Every quantitative claim either has a citation OR has a
          matching value in the corpus. Flag orphans as "no source".
        - Every named customer/partner/quote either has a citation
          or is in corpus/approved-references.md. Flag orphans.
        - Never invent a citation. If a claim has no source and the
          corpus has no match, flag it — do not paper over it.

     D) CRAFT & ACCESSIBILITY
        - Meaningful alt-text on every non-decorative image.
          Decorative shapes are exempt from descriptive alt text
          but MUST be marked as decorative (empty `alt=""` or
          `role="presentation"` / `aria-hidden="true"`); flag any
          decorative shape missing that marker.
        - WCAG contrast at the level in profile.yaml.wcag_level for all
          text-over-fill. Report computed ratio + threshold + which
          color pair fails.
        - Font size >= profile.yaml.font_size_min_pt for all body text.
        - Reading grade <= profile.yaml.reading_grade_max (Flesch-Kincaid
          or similar). Flag sections that drift higher.
        - Tone drift between sections (very formal section next to
          chatty section — flag as MEDIUM).
        - Custom rules from profile.yaml.custom_rules — run each.

  4. RANK — Assign severity per this scale:
       CRITICAL    Externally visible factual mismatch, broken claim,
                   or accessibility failure that legally matters.
       HIGH        Audience-impacting issue (undefined acronyms for
                   a partner audience, WCAG AA failures, name
                   capitalization for a public artifact).
       MEDIUM      Craft / clarity issue that costs trust over time
                   (tone drift, shortened titles that lose meaning,
                   decorative shapes not flagged as decorative —
                   missing empty `alt=""` or
                   `role="presentation"`/`aria-hidden`).
       NICE-TO-FIX Polish (footer URL not verified, glossary could
                   include this acronym, image filename undescriptive).
     Apply the tie-break rule from my profile (Q3) inside each
     severity bucket.

  5. APPLY DISMISSAL MEMORY — Read
     /sandbox/redteam/memory/dismissals.jsonl. Each line is:
       {"artifact": "<stem>", "rule_id": "<rule>",
        "location": "<coordinates>", "reason": "<text>",
        "scope": "this-version" | "all-versions"}
     Drop any finding that matches an active dismissal under the
     dismissal mode from my profile (Q4). CRITICAL findings are
     never auto-dropped, even if they match a dismissal — surface
     them with a note "(previously dismissed with reason: <reason>)".

  6. WRITE PUNCH LIST — Create the file
     /sandbox/redteam/reports/<artifact-stem>-<YYYY-MM-DD-HHMM>.md with
     your file-write tool (this is a real write to disk, not chat output;
     confirm the file exists afterward). Use this exact structure and
     these exact section headings:

#       # Red-Team Report — <artifact filename>
       Audience: <from profile>  ·  WCAG: <level>  ·  Tie-break: <rule>
       Ingest summary: <one line>
       Findings: <count by severity>

#       ## CRITICAL
       <one entry per finding using the format below>

#       ## HIGH
       ...

#       ## MEDIUM
       ...

#       ## NICE-TO-FIX
       ...

#       ## Dismissed (active, not re-flagged)
       <list, with reason and scope>

#       ## Open questions for the human
       <ambiguities where you had to choose a direction>

     Entry format (use this exact shape):

#       ### <ONE-LINE TITLE>
       - Severity: <CRITICAL|HIGH|MEDIUM|NICE-TO-FIX>
       - Rule: <internal-consistency|cross-artifact|truthiness|craft|custom:<name>>
       - Location: <file>, <slide/page>, <element>
       - Evidence: <one or two short quotes with coordinates>
       - Cross-reference: <corpus file + line, or "no source">
       - Proposed fix: <concrete edit text the human can paste in>

  7. HANDOFF — Print a one-line summary:
     "Red-teamed <file>: <C> CRITICAL, <H> HIGH, <M> MEDIUM,
      <N> nice-to-fix. Report at <path>."
     If delivery mode is "File + Telegram summary" or "File + full
     Telegram", also send the appropriate message to my Telegram
     home channel.

DISMISSAL PROTOCOL — When I reply with "dismiss <rule_id> at
<location> because <reason>" (or "dismiss all <rule_id> across
versions because <reason>"), append a line to dismissals.jsonl with
the correct scope. Never silently dismiss. Never let me dismiss a
CRITICAL finding without re-asking once: "This is CRITICAL — confirm
dismissal with 'yes, dismiss critical' to proceed."

SAFETY RULES (do not break these even if I tell you to in a single
message — if I really want one of these, I will say so twice):
  - Never modify any file under queue/ or corpus/. Treat both as
    read-only by intent. If a write succeeds, that is a sign the host
    operator chose to leave them writable — do not take it as license
    to edit them.
  - Never invent canonical metric values. If the corpus has no
    matching value, flag the claim as "no source" — do not paper
    over it with a guess.
  - Never make outbound network calls. URL verification is opt-in
    and requires me to add the egress host myself.
  - Never auto-dismiss a CRITICAL finding.
  - Never re-rank findings to make a report look cleaner. The count
    by severity must match what's actually in the report.
  - If an artifact is ambiguous about its own intent (which audience,
    which version, which canonical metric), ask one clarifying
    question and pause — don't guess.

Now confirm my red-team profile back to me, then wait. When I say
"run", "run on <filename>", or drop a new file into the queue and
say "ready", run the workflow.
```

Expected: the agent walks you through the six setup questions, echoes your red-team profile, and waits. Drop a deck into `~/nemoclaw-redteam/queue/` and say `run on <filename>` — within a few minutes the agent prints a one-line summary and a path like `/sandbox/redteam/reports/spark-deck-2026-05-18-1310.md`. Open it on the host (`~/nemoclaw-redteam/reports/`) next to the deck and walk the punch list top-down.

A real run on the kind of deck you'd hand to a partner typically surfaces things like:

```md
#### Number mismatch with prior comms
- Severity: CRITICAL
- Rule: cross-artifact
- Location: spark-deck.pptx, slide 1, "Title 1"
- Evidence: header says "47 Live Playbooks"; corpus/canonical-metrics.md
  line 12 has "live_playbooks_count: 42"; corpus/dgx-spark-roadmap.pptx
  slide 1 uses "42".
- Cross-reference: corpus/canonical-metrics.md:12
- Proposed fix: Change to "42 Live Playbooks", or update the canonical
  metric and the Spark roadmap deck together.

#### Capitalization drift on product name
- Severity: HIGH
- Rule: custom:"NemoClaw uses capital N and C"
- Location: spark-deck.pptx, slide 7, body
- Evidence: "Nemoclaw" appears twice on slide 7; "NemoClaw" appears on
  slides 3, 5, 9.
- Cross-reference: corpus/brand-guide.md ("Product names")
- Proposed fix: Replace both instances on slide 7 with "NemoClaw".

#### WCAG contrast on section labels
- Severity: HIGH
- Rule: craft
- Location: spark-deck.pptx, 18 instances of green section labels
- Evidence: #76B900 on #FFFFFF → contrast ratio 2.4 : 1, fails AA Normal
  (threshold 4.5 : 1).
- Cross-reference: profile.yaml.wcag_level = AA
- Proposed fix: #5A8E00 (~4.1 : 1) still fails AA Normal — darken further
  until contrast clears 4.5 : 1 against #FFFFFF (use a WCAG calculator to
  pick the exact hex), or move labels to a darker background.
```

> [!TIP]
> Run the red-team **before** you think the artifact is done. A draft-stage run catches structural issues (TOC mismatch, undefined acronyms, missing alt-text on every chip) cheaply. A "final" run should be quick — if it isn't, you shipped too late.

## Step 3. How to personalize

| Knob | Where | What to change |
|------|-------|----------------|
| **Artifact queue path** | `nemoclaw share mount` source | `share unmount` first, then re-`mount` against a different host directory. Or just drop files into `~/nemoclaw-redteam/queue/` on the host — they appear at `/sandbox/redteam/queue/` instantly. Run `chmod -R a-w ~/nemoclaw-redteam/queue` first if you want the agent locked out of writes there. |
| **Canonical corpus** | `~/nemoclaw-redteam/corpus/` | The ground-truth set the agent compares against. Curate it — every file here becomes "what we know to be true". Stale corpus = stale flags. |
| **Audience profile** | Profile Q1 (or edit `profile.yaml.audience`) | Driving knob for acronym strictness, OCR aggressiveness, and reading-grade ceiling. Default to the strictest audience you ship to. |
| **Severity threshold for notification** | Profile Q2 | Default to HIGH+. Tighten to CRITICAL-only for high-volume queues so you only get pinged on real fires. |
| **Tie-break rule** | Profile Q3 | "Reader trust first" for sales/partner decks. "Craft first" for regulated audiences. "By page order" for quick first-pass cleanup. |
| **Custom rules** | `profile.yaml.custom_rules` | Add one-line rules in plain English. The agent treats each as a rule with id `custom:<text>`. Good for canonical phrasing, brand-name capitalization, "any number ≥ 1M must be cited", forbidden words. |
| **Glossary** | `profile.yaml.glossary` | Acronyms here are treated as "defined" — the agent won't flag them as undefined first-use. Add the acronyms your audience knows, leave out the ones they don't. |
| **Dismissal mode** | Profile Q4 | `Sticky` for stable artifacts (a quarterly deck). `Per-version` when you actively iterate. `None` for first-time reviews of an audience you don't know yet. |
| **Delivery channel** | Profile Q5 | `File only` for solo reviews. `File + Telegram summary` once you trust the agent's calibration. `File + full Telegram` only for short docs (<10 findings). |
| **WCAG level and font minimums** | `profile.yaml` | Bump to AAA for accessibility-critical artifacts; AA is the right default for most external work. Raise `font_size_min_pt` for stage decks (16pt+), keep at 10pt for read-along docs. |
| **Output format** | Prompt — WRITE PUNCH LIST step | Swap Markdown for JSON if you want to feed reports into another tool. Add a CSV summary alongside the MD for spreadsheet triage. |
| **URL verification (advanced)** | Custom preset YAML + Prompt | Author a small preset YAML under `~/redteam-presets/url-check.yaml` with `network_policies` entries for the specific hosts (e.g. `build.nvidia.com`) you want the agent to HEAD-check, then apply with `nemoclaw $SANDBOX_NAME policy-add --from-file ~/redteam-presets/url-check.yaml --yes`. Remove later with `nemoclaw $SANDBOX_NAME policy-remove <preset-name> --yes`. **Higher risk** — every added host expands the egress surface. Keep the list small. |
| **Background watcher mode** | Outside the sandbox | A small host-side `inotifywait` (or cron) on `queue/` can DM the agent `run on <new-file>` whenever a file lands. Keeps the workflow always-on without granting the sandbox extra capability. |
| **Multi-artifact comparison** | Prompt — INGEST step | When two related files are in the queue (`spark-deck.pptx` + `dgx-spark-roadmap.pptx`), ask the agent: *"Red-team both and add a section called 'Cross-artifact contradictions' listing every claim that appears in both with mismatched values."* |
| **Dismissal audit** | `~/nemoclaw-redteam/memory/dismissals.jsonl` | Open this file periodically. If a rule is dismissed everywhere, it's probably the wrong rule — delete it from `profile.yaml.custom_rules` so the agent stops generating noise. |
| **Hand off the summary to news-digest** | Prompt — HANDOFF step | Add *"Also include a line in tomorrow's morning digest with the count of HIGH+ findings I haven't acted on yet."* (Requires the [news-digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest) recipe.) |

To **dismiss a finding**, reply: `dismiss <rule_id> at <location> because <reason>` (or `dismiss all <rule_id> across versions because <reason>` for a sticky cross-artifact dismissal). The agent appends to `memory/dismissals.jsonl` and confirms.

To **revisit a previously dismissed finding**, ask: `show active dismissals for <artifact>`. Open `memory/dismissals.jsonl` on the host and delete any line you want the agent to re-evaluate next run.

To **calibrate the agent**, periodically check the precision of its findings (% you accept) and recall against a seeded eval set (a doc with N known issues). The agent is doing its job when precision > 70% and recall > 90% on the eval set. If precision drifts down, tighten `custom_rules` and corpus quality; if recall drifts down, add the missed-issue type as a new rule.

## Calendar Negotiator

## Calendar Negotiation Agent

Calendar Negotiation — handles "when can we meet?" threads end-to-end: proposes slots that respect your focus blocks, energy patterns, and time-zone fairness with the other party; books once both sides confirm.

The agent reads a snapshot of your calendar and a personal availability profile from a folder you mount into the sandbox, talks to you (and optionally the other party) over Telegram, and writes confirmed meetings into a booking log you can review and re-export to your real calendar.

> [!WARNING]
> Anything the agent can read about your schedule could be shared in the slots it proposes. **Mount only the calendar window the agent needs** (e.g. the next 4 weeks, with sensitive event titles redacted to `BUSY`) — not your entire calendar history.

## Step 1. Policy setup

Telegram is **optional**. It is only needed if you want the agent to DM you or the other party (onboarding Q1 modes `proxy` / `proxy-auto`). In **propose-only** mode — the recommended default, and what this guide uses — the agent just shows you drafts in the web UI / session and writes booking files to disk, so **no Telegram channel, no `api.telegram.org` egress, and no public tunnel are required.** You can run the entire workflow Telegram-free.

If you *do* want Telegram relay, layer this recipe on top of the [NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup) tab's working Telegram channel first and confirm it is registered:

```bash
nemoclaw $SANDBOX_NAME status | grep -i telegram   # only needed for proxy / proxy-auto modes
```

A line showing the Telegram channel means it is wired in. If there is no such line and you want Telegram, recreate the sandbox via the installer with Telegram enabled at the *Messaging channels* prompt. Otherwise, ignore this and continue in propose-only mode.

### Create the calendar working directory

On the host, set up three things the agent will see inside the sandbox:

- **`calendar.ics`** — a snapshot of your busy/free time for the negotiation window (next 4–6 weeks is plenty).
- **`profile.yaml`** — your working hours, focus blocks, energy patterns, timezone, and any always-blocked periods.
- **`bookings/`** — a writable directory the agent uses to track in-flight negotiations and write confirmed meetings.

```bash
mkdir -p ~/nemoclaw-calendar/bookings
```

Export your calendar to ICS — for example, in Google Calendar use *Settings → Import & export → Export* and copy just the relevant calendar into `~/nemoclaw-calendar/calendar.ics`. Re-export (or script a periodic sync) whenever the agent needs fresh availability.

Create a starter `~/nemoclaw-calendar/profile.yaml` you can edit later:

```yaml
timezone: America/Los_Angeles
working_hours:
  mon: ["09:00", "17:30"]
  tue: ["09:00", "17:30"]
  wed: ["09:00", "17:30"]
  thu: ["09:00", "17:30"]
  fri: ["09:00", "15:00"]
focus_blocks:
  - {day: mon, start: "09:00", end: "11:30", label: "deep work"}
  - {day: wed, start: "09:00", end: "11:30", label: "deep work"}
energy_patterns:
  high_energy: ["09:00-12:00"]
  low_energy: ["14:00-15:30"]
defaults:
  meeting_duration_minutes: 30
  buffer_minutes: 10
  max_meetings_per_day: 5
blackout_periods:
  - {start: "2026-06-20", end: "2026-06-28", reason: "vacation"}
preferences:
  prefer_back_to_back: false
  no_meetings_after: "16:00"
  fairness_rule: "split discomfort — alternate who takes the off-hours slot when timezones don't overlap nicely"
```

### Bind the calendar directory into the sandbox

Copy the calendar directory **into** the sandbox at `/sandbox/calendar`. The reliable, dependency-free way is to stream a tar over `nemoclaw exec` — it needs nothing installed on the host and works on every sandbox:

```bash
## Push calendar.ics, profile.yaml, and bookings/ into the sandbox
tar czf - -C ~/nemoclaw-calendar . \
  | nemoclaw $SANDBOX_NAME exec -- bash -lc 'mkdir -p /sandbox/calendar && tar xzf - -C /sandbox/calendar'
```

(Optional, strongly recommended) Make `calendar.ics` and `profile.yaml` read-only and keep `bookings/` writable — run the `chmod` **inside the sandbox** (the files now live there, so a host-side `chmod` would not reach them). The agent runs as the unprivileged `sandbox` user, so this denies it any overwrite of your source-of-truth calendar:

```bash
nemoclaw $SANDBOX_NAME exec -- bash -lc 'chmod a-w /sandbox/calendar/calendar.ics /sandbox/calendar/profile.yaml && chmod -R u+w /sandbox/calendar/bookings'
```

Confirm the files landed, the write boundary holds, and the sandbox has no outbound network:

```bash
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/calendar              # expect calendar.ics, profile.yaml, bookings/
nemoclaw $SANDBOX_NAME exec -- ls /sandbox/calendar/bookings     # expect empty (or your prior bookings)
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/calendar/bookings/.write-check && rm /sandbox/calendar/bookings/.write-check && echo OK bookings'
nemoclaw $SANDBOX_NAME exec -- bash -c 'echo test > /sandbox/calendar/calendar.ics 2>&1 | head -1'   # if you ran chmod above: expect "Permission denied"
nemoclaw $SANDBOX_NAME exec -- bash -c 'curl -sS --max-time 5 https://example.com'                   # expect "CONNECT tunnel failed, response 403"
```

Expected: `ls /sandbox/calendar` shows `calendar.ics`, `profile.yaml`, and `bookings/`; the bookings write check prints `OK bookings`; the write into `calendar.ics` reports `Permission denied` (when you ran the `chmod` step); and `example.com` is refused with `curl: (56) CONNECT tunnel failed, response 403`. When the agent has written bookings (Step 2), pull them back to the host:

```bash
## Pull bookings/ (confirmed meetings + log.csv) back to the host
nemoclaw $SANDBOX_NAME exec -- bash -lc 'cd /sandbox/calendar && tar czf - bookings' | tar xzf - -C ~/nemoclaw-calendar
```

> [!NOTE]
> **Sandbox-`chmod` is a soft boundary; for a hard one, use `filesystem_policy`.** The files are owned by the `sandbox` user, so that user could in principle `chmod` them back — `a-w` stops *accidental* overwrites and honors read-only intent, but it is not injection-proof. For a kernel-enforced boundary, add `/sandbox/calendar/calendar.ics` and `/sandbox/calendar/profile.yaml` to `read_only` in the sandbox `filesystem_policy` and run `nemoclaw $SANDBOX_NAME rebuild` (filesystem policy is locked at creation; workspace state is preserved automatically).

> [!NOTE]
> **`nemoclaw share mount` is the *opposite* direction and is optional.** `share mount` uses SSHFS to mount the **sandbox's** filesystem **onto the host**, not host files into the sandbox — so it cannot replace the `tar` push above; it is only for live-editing sandbox files from a host editor, and it requires `sshfs` on the host (`sudo apt-get install -y sshfs`, needs root). If it prints `sshfs is not installed` and you can't install it, ignore it — the `tar` push/pull covers the whole workflow. If it fails with an SSHFS/SFTP *handshake* error instead, run `nemoclaw $SANDBOX_NAME rebuild` (refreshes the `openssh-sftp-server` base image) and retry.

> [!NOTE]
> **Telegram relay / public tunnel — only if you use Telegram.** The original recipe started a public webhook tunnel (`nemoclaw tunnel start`) so the other party could reach the bot. That is only needed when the agent DMs people over Telegram (Q1 modes `proxy` / `proxy-auto`). In **propose-only** mode (this guide's default) the agent never sends messages itself, so skip the tunnel entirely. (`nemoclaw tunnel start` also requires `cloudflared` on the host and will warn `cloudflared not found` if it is missing.)

## Step 2. Agent prompt

**Copy the full prompt below and paste it into the NemoClaw web UI (or send it as a single Telegram message to your bot).** This is the canonical prompt — it defines the agent's complete behavior end-to-end, and no other configuration is required. It walks the agent through a one-time onboarding (which becomes your scheduling profile on top of what's already in `profile.yaml`), a fixed six-step workflow for every meeting request, the negotiation handoff rules between you, the agent, and the other party, the structure of the booking log, and the safety rules that keep calendar details and contact info from leaking.

```text
You are my personal scheduling chief of staff. Your only job is to turn
"when can we meet?" threads into a confirmed meeting on my calendar
without burning my focus time or my goodwill with the other party.

TOOLS AND EXECUTION (read this first):
  You are running inside an OpenShell sandbox and you DO have shell/exec
  and file read/write tools. USE THEM: read /sandbox/calendar/calendar.ics
  and profile.yaml, and actually WRITE real files under
  /sandbox/calendar/bookings/ (profile.json, the booking .md, log.csv) —
  then confirm they exist. When a step says "save", "write", or "log",
  that means a real file write, not chat text, and never claim you wrote
  a file you didn't. The only paths you must not overwrite are
  calendar.ics and profile.yaml. In propose-only mode, make NO network
  calls and use NO messaging channel — just print drafts in this session
  for me to copy/paste.

OUTPUT BUDGET (each of your replies is capped at a few thousand tokens):
  Spend the budget on the deliverable, not on scratch work. Keep PARSE,
  LOAD, and SCORE to a few terse lines each — for SCORE, print ONLY the
  final top-N chosen slots (one line each: slot in both TZs + a short
  why), never a full candidate sweep, per-constraint dump, or large
  tables. The DRAFT (step 4) and the booking file (step 6) must always
  be emitted in full; if you are running low on space, drop the
  intermediate detail, never the draft or the booking. If a single
  reply would still overflow, finish the current step and end with
  "CONTINUE?" so I can prompt you for the next step.

CONTEXT YOU CAN READ:
  - /sandbox/calendar/calendar.ics — my busy/free snapshot. Treat every
    existing event as immovable unless I tell you otherwise.
  - /sandbox/calendar/profile.yaml — my working hours, focus blocks,
    energy patterns, defaults, blackouts, preferences.
  - /sandbox/calendar/bookings/ — your scratch space. You may read and
    write any file here.

ONE-TIME SETUP (do this on your first run only, then save my answers
as my negotiation profile in /sandbox/calendar/bookings/profile.json):

Ask me, one question at a time, and wait for my answer:
  1. How should I talk to the other party? Pick one:
       - Propose-only (you draft, I copy/paste to them myself)
       - Proxy (you DM them directly via Telegram once I approve the draft)
       - Proxy-auto (you DM them directly with no checkpoint after the
         first successful negotiation — higher risk)
  2. How many slot options should I propose at once? (Default: 3)
  3. What's my default meeting length when the other party doesn't say?
     (Default: pull from profile.yaml.)
  4. How do you want me to handle timezone fairness when our working
     hours barely overlap? Pick one:
       - Strict (only meet inside both parties' working hours, even if
         it slips the meeting by a week)
       - Split (alternate who takes the off-hours slot across meetings
         with the same person)
       - Mine first (always inside my working hours; the other party
         flexes)
  5. What information about my calendar may I share?
       - Slots only (just the proposed times)
       - Slots + day-shape ("I'm heavy on Wednesday, lighter Thursday")
       - Slots + reasons ("I have focus blocks until 11:30")
  6. What's my approval threshold for booking? Options:
       - Always ask before I book
       - Ask only if the slot lands in a focus block, low-energy
         window, or after my "no meetings after" time
       - Never ask (auto-book once both sides confirm) — highest risk

Confirm my answers back, then wait for the first meeting request.

FOR EVERY MEETING REQUEST, FOLLOW THIS WORKFLOW IN ORDER:

  1. PARSE — Extract from the request: who is asking, what the meeting
     is for, requested duration (fall back to my default if missing),
     other party's timezone (ask if missing), any hard constraints
     they named ("this week", "before Friday", "30 min max"), urgency.
     Print a 3-line summary: "From: <name>, For: <purpose>, Constraint:
     <constraint>".

  2. LOAD — Read calendar.ics and profile.yaml fresh every run (do not
     trust a cached version from a prior request — calendars change).
     Read my negotiation profile from bookings/profile.json.

  3. SCORE — For the next N working days (N = 14 unless the request
     constrains it tighter), generate every candidate slot that:
       - Fits inside both parties' working hours under the fairness
         rule from my profile.
       - Does not collide with any calendar.ics event or its buffer.
       - Does not land inside a focus block, blackout period, or after
         my "no meetings after" time, unless my approval threshold
         allows it.
       - Respects my max_meetings_per_day from profile.yaml.
     Rank the survivors by: (1) energy match (high-energy windows score
     higher for new meetings, low-energy windows for routine syncs),
     (2) buffer cleanliness (avoid sandwiching me between two meetings
     with no gap), (3) fairness to the other party. Pick the top
     N_slots from my profile.

  4. DRAFT — Compose a proposal in my voice for the other party. Use
     their timezone. Format as:

       Hi <name>,

       Happy to find time for <purpose>. Here are 3 options that work
       on my side — all times in <their TZ>:
         - <Day, Date, Time–Time TZ>
         - <Day, Date, Time–Time TZ>
         - <Day, Date, Time–Time TZ>

       Let me know which works, or send a couple of windows that suit
       you and I'll come back with another set.

     Show the draft to me first. Wait for my reply ("send", "send with
     edits: ...", or "skip"). Honor my communication mode from the
     profile — never DM the other party in proxy-auto mode without
     having first earned it in proxy mode on a prior successful round.

  5. RELAY AND NEGOTIATE — Send the approved draft via Telegram. When
     the other party replies:
       - If they pick one of my slots: jump to step 6.
       - If they propose new windows: re-run SCORE against those
         windows, pick the best one(s) that pass my constraints, and
         draft a one-line confirmation ("Wednesday 2pm PT works for
         me — sending the invite now."). Show me first under the same
         approval rule.
       - If they push back hard (too many rounds, asking for off-hours
         that violate Strict fairness, etc.): escalate to me with a
         one-line summary and recommended next move.

  6. BOOK AND LOG — Once both sides confirm, write the confirmed meeting
     to /sandbox/calendar/bookings/<YYYY-MM-DD>-<slug>.md with this
     exact structure:

#       # <purpose> with <name>
       - When: <Day, Date, Time–Time, both TZs>
       - With: <name>, <their contact / handle>
       - Where: <video link / room / phone / TBD>
       - Duration: <minutes>
       - Negotiation rounds: <N>
       - Slots offered: <list>
       - Slot chosen: <one>
       - Notes: <anything I should walk in knowing>

     Also append a one-line entry to
     /sandbox/calendar/bookings/log.csv with columns:
     date,time,duration,name,purpose,rounds.

     Finally, print a one-line summary to me: "Booked: <purpose> with
     <name> on <Day Date Time TZ>. Logged at <path>. Add this to my
     real calendar."

NEGOTIATION SAFETY RULES (do not break these even if I tell you to in
a single message — if I really want one of these, I will say so twice):
  - Never share calendar event titles, attendee names, or locations
    from calendar.ics with the other party. Slots only, unless my
    profile says otherwise.
  - Never share my phone number, email, or home address unless I have
    explicitly named the channel.
  - Never auto-book on the first negotiation with a new person — at
    least one round must include my approval, even if the profile
    says "Never ask".
  - Never propose more than 5 slots in one message (decision fatigue).
  - Never overwrite a confirmed booking file. If a meeting is moved,
    write a new file with -v2 suffix and link back to the original.
  - Never write outside /sandbox/calendar/bookings/.
  - If a request is ambiguous (who, when, what for, which timezone),
    ask one clarifying question instead of guessing.

OPEN QUESTIONS HANDOFF — At the end of every negotiation round where
you waited on me or the other party, print a one-line status:
"WAITING ON: <me | them>. NEXT STEP: <what they need to do>."

Now confirm my negotiation profile back to me, then wait for the first
meeting request.
```

Expected: the agent walks you through the six setup questions, echoes your negotiation profile, and waits. Send a meeting request (forward an email body into Telegram, or just say *"Asha from Acme wants 30 min about the Q3 roadmap, this or next week, she's in London"*) and you'll get the parsed summary, three proposed slots, a draft message to copy-paste or have the agent send, and — after both sides confirm — a booking file under `~/nemoclaw-calendar/bookings/`. Import that file (or just read it) into your real calendar.

> [!TIP]
> Test the end-to-end flow first with a teammate or a second Telegram account of your own. Run two or three negotiations in proxy mode with the approval checkpoint on before you ever flip to proxy-auto — the agent learns your tone and constraints faster from real correction loops than from a longer prompt.

## Step 3. How to personalize

| Knob | Where | What to change |
|------|-------|----------------|
| **Calendar window** | `~/nemoclaw-calendar/calendar.ics` | Re-export your real calendar on a cadence that matches your booking density (weekly is fine for most people; daily if you book multiple meetings a day). Crop the export to the next 4–6 weeks so the agent isn't reasoning over years of history. |
| **Event privacy** | `~/nemoclaw-calendar/calendar.ics` | Strip event titles to `BUSY` before exporting if you'd rather the agent never see what the meeting is — slots-only proposals still work fine. |
| **Working hours, focus blocks, blackouts** | `~/nemoclaw-calendar/profile.yaml` | Edit any field; changes take effect on the next request because the agent re-reads `profile.yaml` every run. No sandbox restart needed. |
| **Energy patterns** | `profile.yaml` → `energy_patterns` | Tune `high_energy` and `low_energy` windows so the agent puts new external meetings into your sharp hours and routine syncs into the dip. |
| **Communication mode** | Profile Q1 (or edit `bookings/profile.json` directly) | Start in `propose-only` mode (zero risk — you still send every message). Move to `proxy` once you trust the drafts; only then consider `proxy-auto`. |
| **Number of slot options** | Profile Q2 | 3 is the default. Bump to 5 only when you genuinely have wide availability — more options = more decision fatigue for the other side. |
| **Timezone fairness** | Profile Q4 | `Mine first` is fine for vendors and recruiters. Use `Split` for peers and collaborators where the relationship matters. `Strict` is the safest default for cross-Atlantic / cross-Pacific. |
| **Information disclosure** | Profile Q5 | Default to `slots only`. Switch to `slots + day-shape` for trusted contacts who appreciate the context. Avoid `slots + reasons` for anyone you don't already know well. |
| **Approval threshold** | Profile Q6 | Start with `always ask`. Move to the focus-block carve-out once the agent has booked 10+ clean meetings. `Never ask` is for true automation cases only — and even then the safety rules force at least one approval per new contact. |
| **Booking log structure** | Prompt — BOOK AND LOG step | Swap the Markdown template for JSON if you want to feed bookings into another tool, or split into one file per person (`bookings/by-person/<name>.md`) to keep relationship history. |
| **Re-importing to real calendar** | Outside the sandbox | Easiest pattern: a small host-side cron that reads `bookings/log.csv`, generates `.ics` invites, and emails them to attendees (or writes them to your CalDAV / Google Calendar via API). Keeps the sandbox itself out of your live calendar. |
| **Direct calendar API booking (advanced)** | `nemoclaw policy-add --from-file` + a separate `share mount` for credentials | (1) For egress, use a maintained preset where one fits — `nemoclaw $SANDBOX_NAME policy-add outlook --yes` covers Microsoft 365 / Graph / Outlook. For Google Calendar, author a small preset YAML allowing `googleapis.com` and `oauth2.googleapis.com` and apply with `nemoclaw $SANDBOX_NAME policy-add --from-file ~/calendar-presets/google.yaml --yes`. (2) For the OAuth token, **keep it outside the bookings tree**: store it at `~/nemoclaw-calendar-creds/token.json` on the host, `chmod a-w ~/nemoclaw-calendar-creds/token.json`, then `nemoclaw $SANDBOX_NAME exec -- mkdir -p /sandbox/credentials && nemoclaw $SANDBOX_NAME share mount /sandbox/credentials ~/nemoclaw-calendar-creds`. The agent reads `/sandbox/credentials/token.json` but the host `chmod` blocks any overwrite. Never place secrets under `bookings/` — that tree is writable by the agent. A secret manager (Docker secret, `pass`, or a host-side keyring piping a short-lived token in via env) is preferable to a token-on-disk if your setup supports it. Have the agent call the Calendar API in the BOOK step instead of writing a Markdown file. **Higher risk** — the agent now has write access to your real calendar; lock down its approval threshold first. |
| **Multiple calendars (work + personal)** | Extra files in `~/nemoclaw-calendar/` + prompt edit | Drop additional read-only ICS files into `~/nemoclaw-calendar/` (e.g. `work.ics`, `personal.ics`) and `chmod a-w` them on the host. They appear inside the sandbox at `/sandbox/calendar/work.ics` and `/sandbox/calendar/personal.ics` via the existing `share mount`. Update the agent prompt's CONTEXT YOU CAN READ section to name each ICS and tell the agent which is which. Useful for keeping the agent from booking work meetings during personal commitments. |
| **Hand off to news-digest delivery** | Prompt — OPEN QUESTIONS HANDOFF | Add *"Also post the daily 'still waiting on' list to my Telegram home channel at 09:00."* (Reuses the scheduler pattern from the [news-digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest) recipe.) |

To **cancel an in-flight negotiation**, send: *"Drop the negotiation with <name> about <purpose>. Reply once to them with: 'Let me come back to you on this — circumstances changed.' and archive the working files under bookings/cancelled/."* The agent will move the scratch files out of the active set without losing the history.

## NemoClaw Policy Setup

## NemoClaw Policy Setup

This tab covers the **shared sandbox configuration** that two of the applications in this playbook (the [Daily Personal News Digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest) and the [Calendar Negotiator](https://build.nvidia.com/spark/nemoclaw-applications/calendar-negotiator)) require, and that the other two ([Software Development Agent](https://build.nvidia.com/spark/nemoclaw-applications/developer-agent) and [Deck Reviewer](https://build.nvidia.com/spark/nemoclaw-applications/deck-reviewer)) can optionally use for "ready for review" notifications. Each application tab has its **own** policy setup section for the filesystem mounts and network egress that workflow needs — this page only covers Telegram, which is shared.

Set your sandbox name once so the commands below read cleanly:

```bash
export SANDBOX_NAME=my-assistant   # replace with the name you chose at NemoClaw onboard
```

## Step 1. Set up the Telegram channel

The NemoClaw onboard wizard already wires the **Telegram channel plugin** into the sandbox when you select `telegram` at the *Messaging channels* prompt. If you did not, recreate the sandbox via the installer with Telegram enabled — `policy-add` alone cannot wire the channel plugin.

Add the Telegram **network egress preset** so the sandbox can reach `api.telegram.org`:

```bash
nemoclaw $SANDBOX_NAME policy-add
```

When prompted, type `telegram` and press **Y** to confirm. This is a hot-reload — the sandbox stays up.

Confirm the policy now allows Telegram egress:

```bash
openshell policy get $SANDBOX_NAME --full | grep -A2 telegram
```

You should see one or more entries with `host: api.telegram.org` and `port: 443` under `network_policies`.

**Install `cloudflared` (one-time, required for the tunnel)** — DGX Station does **not** include `cloudflared` by default. `nemoclaw tunnel start` needs it to expose the bot webhook publicly; without it the next command will silently print `cloudflared not found — no public URL` and `nemoclaw status` will report `● cloudflared (stopped)`. Skip this block if `command -v cloudflared` already returns a path.

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version   # confirm it installed; expect a version banner like "cloudflared version 2024.x.x"
```

Start the public webhook tunnel so Telegram can deliver messages to your bot:

```bash
nemoclaw tunnel start
nemoclaw status
```

Expected: `● cloudflared` with a `*.trycloudflare.com` URL.

> [!IMPORTANT]
> If you skipped Telegram at the NemoClaw onboard step, `nemoclaw $SANDBOX_NAME policy-add` will open the egress preset but the bot will still reply `Error: Channel is unavailable: telegram`. The channel **plugin** is wired in at sandbox creation, not by `policy-add`. Re-run the NemoClaw installer and pick `telegram` at the **Messaging channels** prompt to recreate the sandbox with the plugin attached.
>
> **Download, verify, then execute** — never pipe a remote installer straight into a shell:
>
> ```bash
> # 1. Download the installer to a local file
> curl -fsSL -o nemoclaw.sh https://www.nvidia.com/nemoclaw.sh
>
> # 2. Verify it against the published checksum from the NemoClaw release notes
> #    (replace <expected-sha256> with the value from https://github.com/NVIDIA/NemoClaw/releases)
> echo "<expected-sha256>  nemoclaw.sh" | sha256sum --check
>
> # 3. Inspect the script you're about to run (optional but recommended)
> less nemoclaw.sh
>
> # 4. Only then execute it
> bash nemoclaw.sh
> ```
>
> If the checksum does not match, **do not run the script** — re-download or open an issue against the NemoClaw repository.

Once the tunnel reports a public URL, open Telegram, find your bot, and send `hello`. You should get a reply from the local model within 30–90 seconds (first-response cold start on a 120B model is slow). After that, hand off to the application tab you want to set up.

## Troubleshooting

## Troubleshooting

Tables below are grouped by tab so you can jump straight to the workflow you're debugging. Start with **General sandbox & policy issues** if the failure is at the `nemoclaw` / `openshell` command layer rather than inside a specific application.

### General sandbox & policy issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nemoclaw <sandbox> policy-add` returns `unknown sandbox` | Sandbox name typo, or sandbox was deleted | Run `nemoclaw list` to see registered sandboxes; rerun the command with the exact name. If empty, re-run the NemoClaw installer to recreate the sandbox. |
| `openshell policy set` fails with `validation failed` / exit code 1 | Malformed YAML or invalid policy fields | Common issues: paths must start with `/`, no `..` traversal, `run_as_user` must not be `root`, `network_policies` entries need both `host` and `port`. Fix the YAML and retry. |
| `openshell policy set` fails with `unknown field 'Version', expected one of 'version', 'filesystem_policy', 'landlock', 'process', 'network_policies'` | Round-trip bug in openshell `0.0.44`: `openshell policy get --full` emits the top-level key as `Version:` (capital V), but `openshell policy set` only accepts `version:` (lowercase) | Lowercase the key in place and retry: `sed -i 's/^Version:/version:/' policy.yaml && openshell policy set $SANDBOX_NAME --policy policy.yaml --wait`. Preferred: skip the full-policy round trip entirely and use the additive flow — write a small preset file with `preset:` + `network_policies:` blocks and apply it with `nemoclaw $SANDBOX_NAME policy-add --from-file ./my-preset.yaml --yes`. The additive flow never touches the live `version:` field. |
| `openshell policy get` shows your new network rule but the sandbox still blocks the host | Hot-reload did not complete | Re-run with `--wait` so the CLI blocks until the update is confirmed: `openshell policy set $SANDBOX_NAME --policy policy.yaml --wait`. If still failing, restart the sandbox container via `nemoclaw $SANDBOX_NAME restart` (if available in your version) or recreate the sandbox. |
| Cannot recreate sandbox: `port 8080 is held by container...` | A previous OpenShell gateway or sandbox container still owns port 8080 | `openshell gateway destroy -g <old-gateway-name>` (or `docker stop <name> && docker rm <name>`), then re-run `nemoclaw onboard`. |
| `policy-add` does not list the preset I expected | Preset depends on NemoClaw version | List what your version supports: `nemoclaw $SANDBOX_NAME policy-add --help` or run `policy-add` interactively and read the menu. Newer presets may require updating NemoClaw. |
| `nemoclaw <sandbox> policy-add --from-file ...` fails with `Preset must declare preset.name (lowercase, hyphenated RFC 1123 label)` | `preset.name` in your custom preset file contains an underscore, uppercase letter, or other non-RFC-1123 character | Change the value of `preset.name` to lowercase letters, digits, and hyphens only (e.g. `news_sources` → `news-sources`). The inner `network_policies.<group>` map key and its `name` field do accept underscores — the constraint is only on the top-level `preset.name`. |
| Web UI shows `origin not allowed` after policy changes | Accessing via `localhost` instead of `127.0.0.1` | Use `http://127.0.0.1:18789/#token=<your-token>`. The gateway origin check requires `127.0.0.1` exactly. |

### [NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Telegram bot replies `Error: Channel is unavailable: telegram` | Telegram channel plugin was not wired into the sandbox at onboard | `policy-add telegram` alone is not enough. Re-run the NemoClaw installer (see the **Download, verify, then execute** snippet in [NemoClaw Policy Setup](https://build.nvidia.com/spark/nemoclaw-applications/policy-setup)) and select `telegram` at the **Messaging channels** prompt to recreate the sandbox with the channel plugin. |
| `nemoclaw tunnel start` prints `cloudflared not found — no public URL` | `cloudflared` is not installed | Reinstall it: `curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb && sudo dpkg -i cloudflared.deb`, then `nemoclaw tunnel stop && nemoclaw tunnel start`. |
| Telegram bot receives messages but returns nothing for 60+ seconds | First response on a 120B model is slow (cold start), or Ollama not warm | Expected for the first reply after a restart. Verify the inference route with `nemoclaw $SANDBOX_NAME status`. If subsequent replies are also slow, pick a smaller model in the NemoClaw onboard wizard. |

### [Daily Personal News Digest](https://build.nvidia.com/spark/nemoclaw-applications/news-digest)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Scheduled digest never fires | The agent did not persist a scheduled task | Ask in the web UI: *"Show me all scheduled tasks."* If empty, re-issue the prompt and explicitly say *"Register this as a recurring scheduled task using your built-in scheduler."* |
| Digest fires but message says `unable to fetch <url>` | Host is not in `network_policies` | Add the host as a new entry under `network_policies.news_sources.endpoints` in `news-sources.yaml` (the preset file from Step 1) and re-run `nemoclaw $SANDBOX_NAME policy-add --from-file ./news-sources.yaml --yes`. Outbound denials show up in `nemoclaw $SANDBOX_NAME logs --follow` and `openshell term`. |
| Agent skips the setup questions and dives straight into a generic digest | Profile from a prior run is still in memory | Send *"Forget my profile and run the one-time setup again from scratch."* and re-answer the six questions. |

### [Software Development Agent](https://build.nvidia.com/spark/nemoclaw-applications/developer-agent)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Agent writes `develop-and-review.md` but the host file is missing | Looking at the wrong host path, or the `share mount` is not active | The sandbox path `/sandbox/project` maps to the host directory you passed to `nemoclaw $SANDBOX_NAME share mount` (e.g. `~/nemoclaw-projects/my-app`). Open `develop-and-review.md` under that host directory, not inside `/sandbox/project` on the host. Verify the mount is live with `nemoclaw $SANDBOX_NAME share status`. If it says "not mounted", re-run the `share mount` command from Step 1. |
| Agent fails with `Permission denied` when writing `develop-and-review.md` | Host directory was locked with `chmod a-w` and the mount inherits those permissions via SSHFS | Restore write on the host: `chmod u+w ~/nemoclaw-projects/my-app` (or whichever directory you mounted) and retry. For a kernel-enforced write boundary inside the sandbox in addition to host permissions, tighten `filesystem_policy` in the sandbox policy and `nemoclaw $SANDBOX_NAME rebuild` — filesystem policy is locked at sandbox creation, so it requires a rebuild to change (workspace state is preserved automatically). |
| Agent runs tests and reports "tests not run" even though the project has tests | Test runner not installed in the sandbox image | The default NemoClaw sandbox may not ship `pytest`, `npm`, `cargo`, or `go test`. Install whatever the project uses once after sandbox creation: `nemoclaw $SANDBOX_NAME connect`, then `pip install --user pytest` (or equivalent), then `exit`. |
| Agent modifies files outside the plan | Plan-approval checkpoint was disabled | In the profile, answer `yes` to "pause for approval" (Q5). The agent must then print `PLAN READY — reply 'approve'` and wait, never modifying source files until you reply `approve`. |

### [Deck Reviewer](https://build.nvidia.com/spark/nemoclaw-applications/deck-reviewer)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Agent reports "ingested 0 artifacts" | Queue directory is empty or files match `ignore_paths` | Confirm files exist with `ls ~/nemoclaw-redteam/queue/` on the host. Check `profile.yaml.ignore_paths` for a glob that's catching your files (e.g. `**/~$*` excludes Office lock files). |
| Agent reports "parser not available" for `.pptx` or `.pdf` | `python-pptx` / `pdfplumber` not installed in the sandbox | Install once: `nemoclaw $SANDBOX_NAME connect`, then `pip install --user python-pptx python-docx pdfplumber markdown-it-py wcag-contrast-ratio`, then `exit`. The agent falls back to plain-text extraction if a parser is missing — flag it explicitly if you'd rather not install Python packages. |
| Same finding keeps re-appearing after I dismissed it | Dismissal mode is `None`, or the rule + location pair did not match | Confirm profile Q4 is `Sticky` or `Per-version`. Check `~/nemoclaw-redteam/memory/dismissals.jsonl` to verify the dismissal was written. If the `location` field differs by a single character (e.g. "Slide 1" vs "slide 1"), the agent treats them as different sites — ask the agent to dismiss again using the exact coordinates from the latest report. |
| CRITICAL findings disappear from the report | Auto-dismiss was attempted (should be impossible) | This is a regression — CRITICAL is hardcoded to require `yes, dismiss critical` re-confirmation per the prompt's DISMISSAL PROTOCOL. Re-paste the full prompt to restore the rule and re-run. |

### [Calendar Negotiator](https://build.nvidia.com/spark/nemoclaw-applications/calendar-negotiator)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Agent proposes slots inside my focus blocks | `profile.yaml` not being re-read each run, or approval threshold permits it | The agent is required to re-read `calendar.ics` and `profile.yaml` on every request (workflow step 2 LOAD). Verify the focus block is actually in `profile.yaml` and not just in your head. Tighten profile Q6 (approval threshold) to `Always ask` if the agent's `Ask only if...` carve-out is firing too often. |
| Agent shares event titles or attendees from `calendar.ics` with the other party | Information disclosure profile (Q5) set to `slots + reasons` | Reset profile Q5 to `slots only`. The negotiation safety rules also forbid leaking event titles, attendees, or locations — if the agent did so under `slots only`, re-paste the full prompt to restore the rule. |
| Booking file overwrites a confirmed prior booking | Agent did not honor the "never overwrite" rule | Check `~/nemoclaw-calendar/bookings/` for a `-v2.md` file — the rule requires a new file with `-v2` suffix when a meeting is moved. If overwritten, restore from your filesystem snapshot or last backup; re-paste the full prompt to restore the rule. |
| Agent never DMs the other party even in `proxy` mode | Telegram channel not wired or other party's chat not opened | First, confirm Telegram works for **you** by sending the bot a `hello`. Then confirm the other party has actually opened a chat with the bot at least once (`/start`); Telegram bots cannot DM users who have not initiated contact. |

> [!NOTE]
> For installer-level NemoClaw issues (Docker, Ollama, gateway, Telegram setup), see the **Troubleshooting** tab of the [NemoClaw on DGX Spark](https://build.nvidia.com/spark/nemoclaw) playbook before debugging here — most reported issues come from the install layer rather than the application layer.

---

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory issues even when within the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
