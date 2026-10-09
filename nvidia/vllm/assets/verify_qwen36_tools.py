#!/usr/bin/env python3
"""Exercise Qwen3.6 tool calling across repeated multi-step conversations."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
METRIC_ORDER = (45, 0, 23, 12, 44, 1, 31, 7, 39, 15, 22, 45)


def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": f"lookup_metric_{number:02d}",
                "description": (
                    f"Return metric number {number} for a city. "
                    f"Use only when metric {number} is requested."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        for number in range(46)
    ]


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), time.perf_counter() - started


def validate(choice, expected):
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""

    if choice.get("finish_reason") != "tool_calls":
        return False, f"finish_reason={choice.get('finish_reason')!r}"
    if len(tool_calls) != 1:
        return False, f"expected one tool call, received {len(tool_calls)}"

    function = tool_calls[0].get("function") or {}
    if function.get("name") != expected:
        return False, f"expected {expected}, received {function.get('name')!r}"

    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError) as error:
        return False, f"invalid tool arguments: {error}"
    if arguments != {"city": "Seoul"}:
        return False, f"unexpected tool arguments: {arguments!r}"

    if "<tool_call>" in content or "<function=" in content:
        return False, "raw tool markup leaked into response content"
    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check for missing, duplicated, or malformed tool calls while "
            "prefix caching and MTP are active."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be at least 1")

    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    tools = make_tools()
    calls = 0
    latencies = []

    for session in range(args.sessions):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are testing a tool API. When the user names a tool, "
                    "call exactly that tool with city Seoul. Never answer "
                    "directly."
                ),
            }
        ]

        for turn, metric in enumerate(METRIC_ORDER):
            expected = f"lookup_metric_{metric:02d}"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Turn {turn}: call {expected} for Seoul. "
                        "Use the requested tool now and do not answer directly."
                    ),
                }
            )
            payload = {
                "model": args.model,
                "messages": messages,
                "tools": tools,
                "temperature": 0,
                "seed": 0,
                "max_tokens": 1024,
            }

            try:
                result, latency = post_json(endpoint, payload, args.timeout)
            except (urllib.error.URLError, TimeoutError) as error:
                print(
                    f"FAIL session={session + 1} turn={turn + 1}: {error}",
                    file=sys.stderr,
                )
                return 1

            calls += 1
            latencies.append(latency)
            choice = result["choices"][0]
            valid, reason = validate(choice, expected)
            if not valid:
                print(
                    f"FAIL session={session + 1} turn={turn + 1}: {reason}",
                    file=sys.stderr,
                )
                print(json.dumps(choice, indent=2), file=sys.stderr)
                return 1

            tool_call = json.loads(
                json.dumps(choice["message"]["tool_calls"][0])
            )
            tool_call["id"] = f"call_{turn:02d}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": choice["message"].get("content"),
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": expected,
                        "content": json.dumps(
                            {
                                "metric": metric,
                                "city": "Seoul",
                                "value": metric * 10,
                            }
                        ),
                    },
                ]
            )

        print(f"session {session + 1}/{args.sessions}: pass", flush=True)

    print(
        f"PASS: {calls} tool calls; "
        f"mean latency {sum(latencies) / len(latencies):.2f}s; "
        f"max latency {max(latencies):.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
