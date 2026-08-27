# Agentic RAG with Provenance-Enforced Refusal on DGX Spark

> Build a fully on-prem retrieval-augmented agent that answers **with a cited source or
> refuses** — never invents — using Nemotron, NeMo Retriever embeddings, a reranker, and
> Milvus, all served locally on your DGX Spark. No cloud APIs, no data leaving the machine.

## Table of Contents

- [Overview](#overview)
  - [Notice & Disclaimers](#notice--disclaimers)
- [What you'll accomplish](#what-youll-accomplish)
- [Prerequisites](#prerequisites)
- [Time & risk](#time--risk)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## Overview

### Basic idea

A retrieval-augmented generation (RAG) agent is only as trustworthy as its weakest failure
mode: **answering confidently when it shouldn't.** A model asked something outside its indexed
knowledge will, by default, produce a fluent answer anyway — and often attach unrelated sources
to it. In a private enterprise setting, that is not a cosmetic bug: it is the difference between
a decision-support tool and a liability.

This playbook builds the opposite behaviour on your DGX Spark, entirely on-prem:

1. **Nemotron** (served via NIM) for reasoning.
2. **NeMo Retriever embeddings** (asymmetric — `passage` when indexing, `query` when searching)
   and a **reranker NIM** for relevance scoring.
3. **Milvus** for the vector store, one collection per tenant.
4. A **provenance-or-refusal** pattern: the agent decides *in code*, from the reranker's
   relevance score, whether it has a good enough source **before** calling the generator. If it
   doesn't, it refuses — and a refusal carries no sources.

The 128 GB of unified memory on DGX Spark is what makes this comfortable: the reasoning model,
the embedding model, the reranker, and the vector store all coexist on one device.

This is the reference pattern behind **X9**, the internal "master" of the
[Tribu SDK](https://sdk.tribucorp.es) — a proprietary agentic SDK built on the NVIDIA AI
Factory stack. This playbook teaches the pattern with public NVIDIA components; the Tribu SDK
is the production-hardened implementation of it (tenant isolation, signed audit, jurisdiction
checks, red-teaming). You can build the pattern yourself from this playbook — the SDK is for
teams who want it industrialised and supported.

### Notice & Disclaimers

#### Quick Start Safety Check

Use a clean environment. Run this playbook on a device or VM with no personal data,
confidential information, or production credentials until you have reviewed the security model
for your own threat model. By installing this playbook you take responsibility for all
third-party components, including their licenses and terms.

#### Key risks with RAG agents

1. **Overconfident answers** — the failure mode this playbook exists to reduce, but no
   mitigation is perfect; measure it for your own corpus (see Step 8).
2. **Prompt injection** — text inside a user's question can carry instructions the model may
   obey. This playbook's refusal gate reduces *irrelevant* answers, but it is **not** an
   injection defence on its own. Treat user input as untrusted.
3. **Data leakage** — anything indexed is reachable by whoever can query. Apply access control
   to the collection and to the endpoint.

This playbook reduces risk through a structural refusal gate; it does not eliminate it. Verify
behaviour against your own data before relying on it.

---

## What you'll accomplish

You will serve three NIMs (reasoning, embedding, reranking) and Milvus on your DGX Spark, index
a small document corpus, and run a query loop that (a) retrieves with hybrid search, (b) scores
relevance with the reranker, (c) refuses below a calibrated threshold, and (d) otherwise answers
citing the exact source. You will then **measure** the refusal behaviour with in-corpus and
out-of-corpus questions — because a refusal gate you haven't measured is a guess.

## Prerequisites

**Hardware:**
- NVIDIA DGX Spark with 128 GB unified memory.
- Enough unified memory for three NIMs plus Milvus (the models below fit comfortably).

**Software:**
- NVIDIA DGX OS (Ubuntu 24.04 base).
- Docker Engine running (`docker info`) with the NVIDIA Container Toolkit configured.
- Python 3.12+ and the `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- An NGC API key to pull the NIM containers (`docker login nvcr.io`).
- Network access to pull containers and model weights (the agent loop itself runs offline).

## Time & risk

- **Estimated time:** 40–60 minutes (plus NIM image and weight download time).
- **Risk level:** Low. Everything runs locally; the query loop makes no outbound calls once the
  NIMs are up. The refusal gate is additive — worst case, a misconfigured threshold refuses too
  much or too little, which Step 8 is designed to catch.
- **Rollback:** `docker compose down` for the NIMs and Milvus; delete the Python venv.

---

## Instructions

## Step 1. Confirm your environment

```bash
head -n 2 /etc/os-release        # expect Ubuntu 24.04 (DGX OS)
nvidia-smi                       # expect a detected GB10 GPU
docker info --format '{{.ServerVersion}}'
python3 --version                # expect 3.12+
```

## Step 2. Authenticate to NGC and pull the NIMs

```bash
docker login nvcr.io             # username: $oauthtoken, password: your NGC API key
```

> [!NOTE]
> If `docker login nvcr.io` fails with a credential-helper error (`wb-svc` or similar), remove
> `credsStore`/`credHelpers` from `~/.docker/config.json` and retry. NVIDIA AI Workbench
> installs a helper that interferes with a plain login.

Pull the three NIMs. Use the DGX Spark (GB10 / ARM64) variants:

```bash
docker pull nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:latest      # reasoning
docker pull nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest                # embeddings
docker pull nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:latest            # reranking
```

## Step 3. Serve the NIMs

Serve reasoning on `:8001`, embeddings on `:8002`, reranking on `:8003`. Keep telemetry off —
this is an on-prem deployment.

```bash
# reasoning
docker run -d --name nim-llm --runtime=nvidia --gpus all \
  -e NIM_TELEMETRY_MODE=0 -p 8001:8000 \
  nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:latest

# embeddings (asymmetric: needs input_type per call — see Step 6)
docker run -d --name nim-embed --runtime=nvidia --gpus all \
  -e NIM_TELEMETRY_MODE=0 -p 8002:8000 \
  nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest

# reranking
docker run -d --name nim-rerank --runtime=nvidia --gpus all \
  -e NIM_TELEMETRY_MODE=0 -p 8003:8000 \
  nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:latest
```

Wait for each to report ready:

```bash
for p in 8001 8002 8003; do
  curl -s -o /dev/null -w "port $p: %{http_code}\n" -m 5 localhost:$p/v1/health/ready
done            # expect 200 on all three (may take a few minutes on first start)
```

## Step 4. Start Milvus

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o milvus.sh
bash milvus.sh start        # serves on localhost:19530
```

## Step 5. Set up the Python environment

```bash
cd ~ && uv venv rag-env && source rag-env/bin/activate
uv pip install pymilvus httpx
```

## Step 6. Know the one gotcha: the embedding NIM is asymmetric

`llama-nemotron-embed-1b-v2` returns **HTTP 400 without an `input_type` field**. Use
`"passage"` when indexing documents and `"query"` when embedding a search. This is the single
most common integration error with this model.

```bash
# indexing a document — input_type: passage
curl -s localhost:8002/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "nvidia/llama-nemotron-embed-1b-v2",
  "input": ["Tribucorp builds on the NVIDIA AI Factory stack."],
  "input_type": "passage"
}' | head -c 200
```

Note the model dimension the endpoint reports at startup — the 2026 ecosystem default is
**2048**, not 1024. Your Milvus collection dimension must match, or retrieval fails closed.

## Step 7. Index a corpus and run the provenance-or-refusal loop

The reference script (`assets/agentic_rag.py`) does four things per query:

1. Embed the question (`input_type: query`) and hybrid-search Milvus.
2. Ask the reranker to score the candidates; keep the **top logit**.
3. **If the top logit is below the refusal threshold, refuse — with an empty source list.**
   This happens *before* any call to the reasoning model.
4. Otherwise, build the context and ask Nemotron to answer **citing the exact source**, with a
   system prompt that forbids using outside knowledge.

```bash
python assets/agentic_rag.py --index ./docs        # index a folder of .md/.txt files
python assets/agentic_rag.py --ask "your question"  # query with the refusal gate active
```

## Step 8. Measure the refusal gate — do not skip this

A refusal threshold you picked by eye is a guess. Calibrate it against **your** corpus:

```bash
python assets/agentic_rag.py --calibrate ./calibration.jsonl
```

Run at least 20 in-corpus questions and 20 out-of-corpus questions. Look at the distribution of
the top reranker logit for each group. A healthy corpus shows a clean gap between the two; set
the threshold in that gap. If the distributions overlap, the reranker cannot separate them for
your data — that is a real finding, not a failure: tune the corpus or the retrieval before
trusting the gate.

> [!TIP]
> Two questions that *look* out-of-corpus may actually be in it (or vice versa). Before
> computing the gap, confirm each question's true membership against what you indexed — a single
> mislabelled question can make a clean gap look like an overlap.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Embedding NIM returns HTTP 400 | Missing `input_type` | Add `"input_type": "passage"` (index) or `"query"` (search) — Step 6 |
| Retrieval returns nothing / dimension error | Milvus collection dimension ≠ NIM dimension | Recreate the collection with the dimension the NIM reports (2048 by default) |
| Reasoning model prepends `<think>…` monologue | Reasoning model emits its chain-of-thought | Strip everything up to and including `</think>` before returning the answer |
| Answer is truncated or empty | Reasoning model spent the token budget inside `<think>` | Raise `max_tokens` (2000 is a safe start for this model) |
| Model answers an out-of-corpus question anyway | Refusal gate not active (no reranker) or threshold too low | Confirm the reranker is reachable on `:8003`; re-run `--calibrate` |
| `docker login nvcr.io` dumps a `wb-svc` error | AI Workbench credential helper | Remove `credsStore`/`credHelpers` from `~/.docker/config.json` |
| NIM slow / OOM | Three NIMs + Milvus contending for memory | Serve fewer models at once, or flush cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |

## Next steps

- **Add tenant isolation**: one Milvus collection per tenant, with the tenant as a hard
  boundary — not a filter that can be forgotten.
- **Add access control per chunk**: pass the requester's roles into every query; never default
  to "see everything".
- **Add jurisdiction checks**: tag each chunk with its data jurisdiction and refuse to send an
  EU chunk to an endpoint outside the EU — evaluated before inference.
- **Red-team the loop**: run the OWASP Agentic preset against your deployment; measure prompt
  injection with n ≥ 20, not a single pass.
- **Industrialise it**: the four items above, plus signed audit, licensing, and continuous
  red-teaming, are what the [Tribu SDK](https://sdk.tribucorp.es) provides as a supported
  library. This playbook is the honest, buildable core of that pattern.

## License

This playbook is contributed under the repository's Apache-2.0 license. The Tribu SDK itself is
proprietary and is referenced here only as the production implementation of the pattern; no SDK
source is included or required to complete this playbook.
