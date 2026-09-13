#!/usr/bin/env python3
"""Benchmark multi-turn conversation throughput on SGLang.

Sends parallel multi-turn conversations and measures:
- Per-turn latency (end-to-end wall time; may rise as prompts grow)
- Total throughput in tokens per second
- Optional cached prefill tokens from API usage when exposed by the server

Server cache / metrics text is summarized on stdout; full scrape is written to a
detail file for easier review (avoids dumping large /metrics blobs in the terminal).
"""

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _cached_from_usage(usage: Dict[str, Any]) -> Optional[int]:
    """Return cached prefill token count if the server reports it (OpenAI-style usage)."""
    if not usage:
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        v = details.get("cached_tokens")
        if isinstance(v, int):
            return v
    # Some stacks use alternate keys
    for key in ("cached_prompt_tokens", "cache_read_input_tokens"):
        v = usage.get(key)
        if isinstance(v, int):
            return v
    return None


def chat_completion(base_url, model, messages, max_tokens=128):
    """Send a chat completion request and measure timing."""
    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    cached = _cached_from_usage(usage)
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_prompt_tokens": cached,
        "content": data["choices"][0]["message"]["content"],
    }


def run_conversation(base_url, model, conv_id, num_turns):
    """Run a multi-turn conversation and return per-turn metrics."""
    system_msg = {"role": "system", "content": "You are a helpful assistant. Keep responses concise."}
    topics = [
        "What is photosynthesis?",
        "How does it relate to the carbon cycle?",
        "What role do oceans play in that cycle?",
        "How does climate change affect the oceans?",
        "What can individuals do to reduce their carbon footprint?",
        "How do renewable energy sources compare?",
        "What are the most promising new energy technologies?",
        "How long until fusion energy is practical?",
    ]

    messages = [system_msg]
    results = []

    for turn in range(num_turns):
        user_msg = {"role": "user", "content": topics[turn % len(topics)]}
        messages.append(user_msg)

        result = chat_completion(base_url, model, messages)
        result["conversation_id"] = conv_id
        result["turn"] = turn + 1
        results.append(result)

        messages.append({"role": "assistant", "content": result["content"]})

    return results


def run_structured_output_test(base_url, model):
    """Demonstrate structured JSON output with timing."""
    print("\n--- Structured Output Test ---")

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["name", "category", "score"],
                },
            }
        },
        "required": ["items"],
    }

    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Rate these programming languages for data science: Python, R, Julia. "
                    "Return name, category, and a score from 1-10.",
                }
            ],
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "ratings", "schema": schema},
            },
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    parsed = json.loads(content)
    print(f"  Time: {elapsed:.2f}s")
    print("  Valid JSON: Yes")
    print(f"  Items returned: {len(parsed.get('items', []))}")
    print(f"  Output: {json.dumps(parsed, indent=2)}")


_MAX_HIGHLIGHT_LEN = 200


def _interesting_metric_line(line: str) -> bool:
    stripped = line.strip()
    # /server_info returns a single-line JSON blob whose schema mentions "cache"
    # dozens of times; suppress it (and any other JSON or oversized line) so the
    # terminal summary stays readable. The full body is still in the detail file.
    if not stripped or stripped[0] in "{[":
        return False
    if len(stripped) > _MAX_HIGHLIGHT_LEN:
        return False
    lower = stripped.lower()
    if not any(kw in lower for kw in ("cache", "prefix", "radix", "hit", "reuse")):
        return False
    # Prefer lines that look like Prometheus samples or log counters
    return bool(re.search(r"\d", stripped))


def get_cache_stats(base_url: str, detail_path: Path) -> None:
    """Write full /server_info and /metrics text to detail_path; print a short summary."""
    print("\n--- Server cache / metrics (summary) ---")
    sections: List[Tuple[str, str]] = []
    for endpoint in ("/server_info", "/metrics"):
        try:
            resp = requests.get(f"{base_url}{endpoint}", timeout=30)
            label = f"{endpoint} HTTP {resp.status_code}"
            body = resp.text if resp.text else ""
            sections.append((label, body))
        except requests.RequestException as exc:
            sections.append((f"{endpoint} error", str(exc)))

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as fh:
        for label, body in sections:
            fh.write(f"===== {label} =====\n")
            fh.write(body)
            if body and not body.endswith("\n"):
                fh.write("\n")
            fh.write("\n")

    print(f"  Full raw responses written to: {detail_path.resolve()}")

    highlights: List[str] = []
    for _, body in sections:
        if not body or body.startswith("HTTP"):
            continue
        for line in body.splitlines():
            s = line.strip()
            if _interesting_metric_line(s):
                highlights.append(s)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for h in highlights:
        if h not in seen:
            seen.add(h)
            unique.append(h)

    max_print = 12
    if not unique:
        print("  No cache-related lines matched heuristics in /server_info or /metrics.")
        print("  See the detail file above; also check `docker logs` for #cached-token lines.")
        return

    print(f"  Showing up to {max_print} cache-related lines (of {len(unique)} matched):")
    for line in unique[:max_print]:
        print(f"    {line}")
    if len(unique) > max_print:
        print(f"  … plus {len(unique) - max_print} more in the detail file.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SGLang multi-turn inference")
    parser.add_argument("--base-url", default="http://localhost:30000", help="SGLang server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--num-conversations", type=int, default=20, help="Number of parallel conversations")
    parser.add_argument("--turns-per-conversation", type=int, default=5, help="Turns per conversation")
    parser.add_argument("--max-workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument(
        "--cache-detail-file",
        default="sglang_benchmark_cache_details.log",
        help="Path to write full /server_info and /metrics bodies for offline review",
    )
    args = parser.parse_args()
    detail_path = Path(args.cache_detail_file)

    print("SGLang Multi-Turn Benchmark")
    print(f"  Server: {args.base_url}")
    print(f"  Model: {args.model}")
    print(f"  Conversations: {args.num_conversations}")
    print(f"  Turns each: {args.turns_per_conversation}")
    print()
    print(
        "Note: per-turn **wall time** often **increases** as the transcript grows (more prompt "
        "tokens to attend to, longer assistant decodes, and parallel load). RadixAttention still "
        "reuses KV for shared prefixes—confirm with `docker logs` (#cached-token) or cached_tokens "
        "in usage when the server exposes them."
    )
    print()

    print("Warming up...")
    chat_completion(args.base_url, args.model, [{"role": "user", "content": "Hello"}], max_tokens=8)

    print(f"Running {args.num_conversations} conversations with {args.turns_per_conversation} turns each...")
    all_results = []
    bench_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_conversation, args.base_url, args.model, i, args.turns_per_conversation): i
            for i in range(args.num_conversations)
        }
        for future in as_completed(futures):
            conv_results = future.result()
            all_results.extend(conv_results)

    bench_elapsed = time.perf_counter() - bench_start

    print(f"\n--- Results ({bench_elapsed:.1f}s total) ---\n")

    total_prompt_tokens = sum(r["prompt_tokens"] for r in all_results)
    total_completion_tokens = sum(r["completion_tokens"] for r in all_results)
    total_tokens = total_prompt_tokens + total_completion_tokens

    print(f"Total tokens: {total_tokens:,} ({total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion)")
    print(f"Throughput: {total_tokens / bench_elapsed:.0f} tok/s")
    print()

    any_cached = any(r.get("cached_prompt_tokens") is not None for r in all_results)

    print(f"{'Turn':<6} {'Median wall':<14} {'P90 wall':<14} {'Med prompt tok':<16} {'Med cached prefill':<20} {'N':<6}")
    print("-" * 82)

    for turn in range(1, args.turns_per_conversation + 1):
        turn_results = [r for r in all_results if r["turn"] == turn]
        latencies = sorted(r["elapsed_s"] for r in turn_results)
        median = statistics.median(latencies)
        p90 = latencies[int(len(latencies) * 0.9)] if len(latencies) >= 10 else latencies[-1]
        med_prompt = int(statistics.median([r["prompt_tokens"] for r in turn_results]))
        cached_vals = [r["cached_prompt_tokens"] for r in turn_results if r["cached_prompt_tokens"] is not None]
        med_cached = f"{statistics.median(cached_vals):.0f}" if cached_vals else ("n/a" if not any_cached else "0")
        print(f"  {turn:<4} {median:>10.3f}s   {p90:>10.3f}s   {med_prompt:<16} {med_cached:<20} {len(turn_results):<6}")

    print()
    print("Interpretation: rising median wall time with turn index is normal under load and longer contexts.")
    print("Use single-conversation runs (`--num-conversations 1`) and server logs for the clearest cache story.")

    run_structured_output_test(args.base_url, args.model)
    get_cache_stats(args.base_url, detail_path)


if __name__ == "__main__":
    main()
