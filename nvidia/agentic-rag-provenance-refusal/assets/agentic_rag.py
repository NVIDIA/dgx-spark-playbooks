#!/usr/bin/env python3
"""Agentic RAG with provenance-enforced refusal on DGX Spark.

Reference implementation for the playbook of the same name. Deliberately minimal: only
`pymilvus` and `httpx`, no framework, no proprietary code. It demonstrates the one behaviour
that matters — answer with a cited source or refuse, decided in code from the reranker score
BEFORE the reasoning model is ever called.

This is the honest, buildable core of the pattern that the Tribu SDK (https://sdk.tribucorp.es)
industrialises. Everything here runs offline once the NIMs and Milvus are up.

Usage:
    python agentic_rag.py --index ./docs
    python agentic_rag.py --ask "your question"
    python agentic_rag.py --calibrate ./calibration.jsonl

Environment (all with sensible DGX Spark defaults):
    NIM_LLM_URL     reasoning NIM        (default http://localhost:8001)
    NIM_EMBED_URL   embedding NIM        (default http://localhost:8002)
    NIM_RERANK_URL  reranker NIM         (default http://localhost:8003)
    MILVUS_URI      Milvus               (default http://localhost:19530)
    REFUSAL_LOGIT   refusal threshold    (default -4.0 — CALIBRATE for your corpus, Step 8)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from pymilvus import MilvusClient

LLM_URL = os.environ.get("NIM_LLM_URL", "http://localhost:8001")
EMBED_URL = os.environ.get("NIM_EMBED_URL", "http://localhost:8002")
RERANK_URL = os.environ.get("NIM_RERANK_URL", "http://localhost:8003")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
REFUSAL_LOGIT = float(os.environ.get("REFUSAL_LOGIT", "-4.0"))

EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
LLM_MODEL = "nvidia/nemotron-nano-9b-v2"
COLLECTION = "playbook_rag"
TIMEOUT = httpx.Timeout(120.0)  # real NIM inference is slower than a normal API

SYSTEM_PROMPT = (
    "You answer ONLY from the provided context, citing the exact source in brackets exactly as "
    "it appears (e.g. [source: notes.md#2]). If the context is not enough to answer, say so "
    "explicitly instead of inventing or using outside knowledge."
)

REFUSAL_TEXT = (
    "I have no indexed source that answers this with confidence, so I will not invent an answer "
    "without verifiable provenance."
)


def _embed(text: str, input_type: str) -> list[float]:
    """The embed NIM is ASYMMETRIC: it returns HTTP 400 without input_type. 'passage' when
    indexing, 'query' when searching. This is the #1 integration gotcha with this model."""
    r = httpx.post(
        f"{EMBED_URL}/v1/embeddings",
        json={"model": EMBED_MODEL, "input": [text], "input_type": input_type},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def _rerank(question: str, passages: list[str]) -> list[tuple[int, float]]:
    """Return (index, logit) sorted by descending relevance. The logit is the signal the
    refusal gate uses — a well-separated corpus scores relevant passages clearly above
    irrelevant ones. We keep the logits; most integrations throw them away and lose the gate."""
    r = httpx.post(
        f"{RERANK_URL}/v1/ranking",
        json={
            "model": RERANK_MODEL,
            "query": {"text": question},
            "passages": [{"text": p} for p in passages],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    ranked = sorted(r.json()["rankings"], key=lambda x: x["logit"], reverse=True)
    return [(x["index"], x["logit"]) for x in ranked]


def _complete(question: str, context: str) -> str:
    r = httpx.post(
        f"{LLM_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "max_tokens": 2000,  # reasoning models spend budget inside <think> before answering
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # Reasoning models (nemotron-nano included) emit their chain-of-thought before </think>.
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):].lstrip() if idx != -1 else text


def _client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


def cmd_index(folder: str) -> int:
    files = sorted(Path(folder).glob("**/*.md")) + sorted(Path(folder).glob("**/*.txt"))
    if not files:
        print(f"No .md/.txt files under {folder}", file=sys.stderr)
        return 1

    # Read the real dimension from the endpoint, fail-closed if the collection disagrees.
    dim = len(_embed("dimension probe", "passage"))
    print(f"Embedding dimension reported by the NIM: {dim} (the 2026 default is 2048).")

    client = _client()
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    client.create_collection(collection_name=COLLECTION, dimension=dim, auto_id=True)

    rows, n = [], 0
    for f in files:
        # One chunk per paragraph — deliberately simple; real chunking overlaps and contextualises.
        for i, para in enumerate(p for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()):
            rows.append({"vector": _embed(para, "passage"), "text": para.strip(),
                         "source": f"{f.name}#{i}"})
            n += 1
    client.insert(collection_name=COLLECTION, data=rows)
    print(f"Indexed {n} chunks from {len(files)} file(s) into '{COLLECTION}'.")
    return 0


def _answer(question: str, *, top_k: int = 5, candidates: int = 20) -> dict:
    client = _client()
    if not client.has_collection(COLLECTION):
        return {"answer": REFUSAL_TEXT, "sources": [], "refused": True, "top_logit": None}

    hits = client.search(
        collection_name=COLLECTION,
        data=[_embed(question, "query")],
        limit=candidates,
        output_fields=["text", "source"],
    )[0]
    if not hits:
        return {"answer": REFUSAL_TEXT, "sources": [], "refused": True, "top_logit": None}

    passages = [h["entity"]["text"] for h in hits]
    ranked = _rerank(question, passages)
    top_logit = ranked[0][1]

    # THE GATE: refuse in code, from the score, BEFORE calling the reasoning model. A refusal
    # never carries sources — a flag that says "refused" while returning 5 sources is a lie
    # that any downstream consumer (evals, guardrails) will miscount.
    if top_logit < REFUSAL_LOGIT:
        return {"answer": REFUSAL_TEXT, "sources": [], "refused": True, "top_logit": top_logit}

    chosen = [hits[i] for i, _ in ranked[:top_k]]
    context = "\n\n".join(f"[source: {h['entity']['source']}]\n{h['entity']['text']}" for h in chosen)
    answer = _complete(question, context)
    return {
        "answer": answer,
        "sources": [h["entity"]["source"] for h in chosen],
        "refused": False,
        "top_logit": top_logit,
    }


def cmd_ask(question: str) -> int:
    out = _answer(question)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_calibrate(path: str) -> int:
    """Each line: {"q": "...", "in_corpus": true|false}. Prints the logit distribution per group
    so you can set REFUSAL_LOGIT in the gap between them (Step 8). n>=20 per group."""
    inside, outside = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        res = _answer(rec["q"])
        (inside if rec["in_corpus"] else outside).append(res["top_logit"])

    def stats(xs: list[float | None]) -> str:
        vals = sorted(v for v in xs if v is not None)
        if not vals:
            return "n=0"
        mid = vals[len(vals) // 2]
        return f"n={len(vals)}  worst={vals[0]:.2f}  median={mid:.2f}  best={vals[-1]:.2f}"

    print(f"in-corpus : {stats(inside)}")
    print(f"out-corpus: {stats(outside)}")
    ins = [v for v in inside if v is not None]
    outs = [v for v in outside if v is not None]
    if ins and outs:
        gap = min(ins) - max(outs)
        if gap > 0:
            print(f"clean gap of {gap:.2f} logits — set REFUSAL_LOGIT near {min(ins) - gap / 2:.2f}")
        else:
            print("distributions OVERLAP — the reranker cannot separate them for this corpus. "
                  "That is a real finding: tune the corpus or retrieval before trusting the gate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", metavar="FOLDER")
    g.add_argument("--ask", metavar="QUESTION")
    g.add_argument("--calibrate", metavar="JSONL")
    a = ap.parse_args()
    if a.index:
        return cmd_index(a.index)
    if a.ask:
        return cmd_ask(a.ask)
    return cmd_calibrate(a.calibrate)


if __name__ == "__main__":
    sys.exit(main())
