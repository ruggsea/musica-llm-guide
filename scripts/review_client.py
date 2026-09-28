#!/usr/bin/env python3
"""Score chat prompts against a live vLLM OpenAI server: first-token top-k logprobs, one call per line.

Input JSONL:  {"id", "messages", optional "chat_template_kwargs"}
Output JSONL: {"id", "text", "top_logprobs": [[token, logprob], ...], "prompt_tokens"}
Resumable: ids already in the output are skipped. Errors are counted and printed, never written.

  review_client.py <base_url> <served_name> <in.jsonl> <out.jsonl> [concurrency] [top_k]
Default chat_template_kwargs come from $REVIEW_TEMPLATE_KWARGS (JSON), e.g. '{"enable_thinking": false}'.
$REVIEW_MAX_TOKENS (default 1) and $REVIEW_LIMIT (first N input lines only) are for a readable text sample.
$REVIEW_LABELS="AGENT,HUMAN": also check that >= $REVIEW_LABEL_MIN (0.95) of top-1 first tokens begin one of those words
(a token is a label start if, stripped and upper-cased, it is a non-empty prefix of a label); below that, exit 4.
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

base, served, in_path, out_path = sys.argv[1:5]
concurrency = int(sys.argv[5]) if len(sys.argv) > 5 else 32
top_k = int(sys.argv[6]) if len(sys.argv) > 6 else 20
default_kwargs = json.loads(os.environ.get("REVIEW_TEMPLATE_KWARGS", "{}"))
max_tokens = int(os.environ.get("REVIEW_MAX_TOKENS", "1"))
limit = int(os.environ.get("REVIEW_LIMIT", "0"))

done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        done = {json.loads(line)["id"] for line in f if line.strip()}
rows = [json.loads(line) for line in open(in_path) if line.strip()]
if limit:
    rows = rows[:limit]
ids = [r["id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate ids in input"
todo = [r for r in rows if r["id"] not in done]
print(f"input {len(rows)}, already done {len(done)}, to score {len(todo)}, concurrency {concurrency}", flush=True)


def score(row):
    body = {
        "model": served,
        "messages": row["messages"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": top_k,
        "chat_template_kwargs": row.get("chat_template_kwargs", default_kwargs),
    }
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    first = choice["logprobs"]["content"][0]
    return {
        "id": row["id"],
        "text": choice["message"]["content"],
        "top_logprobs": [[t["token"], t["logprob"]] for t in first["top_logprobs"]],
        "prompt_tokens": resp["usage"]["prompt_tokens"],
    }


labels = [w.strip().upper() for w in os.environ.get("REVIEW_LABELS", "").split(",") if w.strip()]
label_min = float(os.environ.get("REVIEW_LABEL_MIN", "0.95"))
top1 = []
lock = threading.Lock()
n_ok = n_err = prompt_tokens = 0
t0 = time.monotonic()
with open(out_path, "a") as out, ThreadPoolExecutor(concurrency) as pool:
    futures = {pool.submit(score, r): r["id"] for r in todo}
    for fut in as_completed(futures):
        try:
            res = fut.result()
        except Exception as e:  # one bad page must not stop the batch; it stays unscored and is retried on rerun
            n_err += 1
            print(f"ERROR {futures[fut]}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        with lock:
            out.write(json.dumps(res) + "\n")
            out.flush()
        n_ok += 1
        top1.append(res["top_logprobs"][0][0] if res["top_logprobs"] else "")
        prompt_tokens += res["prompt_tokens"]
        if n_ok % 100 == 0:
            el = time.monotonic() - t0
            print(f"  {n_ok}/{len(todo)} scored, {n_ok/el:.2f} pages/s, {prompt_tokens/el:.0f} prompt tok/s", flush=True)

el = time.monotonic() - t0
print(f"REVIEW scored={n_ok} errors={n_err} seconds={el:.0f} pages_per_s={n_ok/max(el,1e-9):.2f} "
      f"prompt_tok_per_s={prompt_tokens/max(el,1e-9):.0f} concurrency={concurrency}", flush=True)
if labels and top1:
    def is_label(tok):
        t = tok.strip().upper()
        return bool(t) and any(w.startswith(t) for w in labels)
    frac = sum(map(is_label, top1)) / len(top1)
    from collections import Counter
    print(f"LABEL CHECK {'ok' if frac >= label_min else 'FAIL'}: {frac:.1%} of top-1 first tokens begin {labels} "
          f"(need {label_min:.0%}); most common: {Counter(top1).most_common(5)}", flush=True)
    if frac < label_min:
        sys.exit(4)
sys.exit(1 if n_err else 0)
