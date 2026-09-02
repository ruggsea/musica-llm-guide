#!/usr/bin/env python3
"""Measure single-stream TTFT and decode tok/s against a live vLLM OpenAI server.

Prints one shell-sourceable line:  TTFT_MS=<int> DECODE_TPS=<float> GEN_TOKENS=<int>
Values are "--" when they could not be measured. Run AFTER the smoke generation so
the first-request JIT/cudagraph warmup is already paid for.
"""
import json
import sys
import time
import urllib.request

base, model = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

body = json.dumps({
    "model": model,
    "prompt": "Write a detailed paragraph about the history of high performance computing.",
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()

req = urllib.request.Request(
    f"{base}/v1/completions", data=body, headers={"Content-Type": "application/json"}
)

t_send = time.monotonic()
t_first = None
t_last = None
chunks = 0
usage_tokens = None

with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if obj.get("usage"):
            usage_tokens = obj["usage"].get("completion_tokens")
        choices = obj.get("choices") or []
        if choices and choices[0].get("text"):
            now = time.monotonic()
            if t_first is None:
                t_first = now
            t_last = now
            chunks += 1

n_tokens = usage_tokens if usage_tokens else chunks
ttft_ms = "--" if t_first is None else str(int((t_first - t_send) * 1000))

decode_tps = "--"
if t_first is not None and t_last is not None and n_tokens > 1:
    decode_s = t_last - t_first
    # tokens after the first, over the time spent producing them
    if decode_s > 0:
        decode_tps = f"{(n_tokens - 1) / decode_s:.1f}"

print(f"TTFT_MS={ttft_ms} DECODE_TPS={decode_tps} GEN_TOKENS={n_tokens or '--'}")
