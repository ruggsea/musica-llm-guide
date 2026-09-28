#!/usr/bin/env python3
"""Does a live vLLM server give real answers? A reply existing is not enough (MiMo-V2.5 "passed" on gibberish, 2026-09-27).

  sanity_check.py <base_url> <served_name>        exit 0 = SANITY PASS, 1 = SANITY FAIL
Every test must pass: known-answer prompts (completion and chat), no degenerate output (one token repeated,
punctuation runs), English answer to an English chat question, and a sane length.
"""
import json
import re
import sys
import urllib.request
from collections import Counter

base, served = sys.argv[1], sys.argv[2]


def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def completion(prompt):
    r = post("/v1/completions", {"model": served, "prompt": prompt, "max_tokens": 24, "temperature": 0})
    return r["choices"][0]["text"]


def chat(question):
    r = post("/v1/chat/completions", {"model": served, "messages": [{"role": "user", "content": question}],
                                      "max_tokens": 1024, "temperature": 0})
    text = r["choices"][0]["message"]["content"] or ""
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()  # judge the answer, not the reasoning


def degenerate(text):
    """Why this reply is not text, or '' if it looks like text."""
    words = text.split()
    if not words:
        return "empty"
    if re.search(r"([^\w\s])\1{3,}", text):
        return "punctuation run"
    top, n = Counter(words).most_common(1)[0]
    if len(words) >= 6 and n / len(words) > 0.5:
        return f"'{top}' is {n}/{len(words)} words"
    alnum = sum(c.isalnum() for c in text)
    if alnum < 0.6 * sum(not c.isspace() for c in text):
        return "mostly symbols"
    return ""


def english(text):
    letters = [c for c in text if c.isalpha()]  # an answer of only digits ("42") has no language to judge
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.9


TESTS = [  # (kind, prompt, required substring, must be English)
    ("completion", "The capital of France is", "paris", False),
    ("completion", "The first three planets from the Sun are Mercury, Venus, and", "earth", False),
    ("chat", "What is the capital of Japan? Answer with one word.", "tokyo", True),
    ("chat", "What is 17 + 25? Answer with just the number.", "42", True),
]

failures = []
for kind, prompt, answer, need_english in TESTS:
    try:
        text = completion(prompt) if kind == "completion" else chat(prompt)
    except Exception as e:  # a server error on a known-answer prompt is a failed check, reported as such
        failures.append(f"{kind} '{prompt[:30]}': {type(e).__name__}: {str(e)[:120]}")
        print(f"  ERROR {kind}: {prompt[:40]!r} -> {e}", flush=True)
        continue
    why = []
    if answer not in text.lower():
        why.append(f"missing '{answer}'")
    bad = degenerate(text)
    if bad:
        why.append(f"degenerate ({bad})")
    if need_english and not english(text):
        why.append("not English")
    if kind == "chat" and len(text) > 400:
        why.append(f"too long for a one-word answer ({len(text)} chars)")
    print(f"  {'ok  ' if not why else 'FAIL'} {kind}: {prompt[:40]!r} -> {text[:80]!r} {'; '.join(why)}", flush=True)
    if why:
        failures.append(f"{kind} '{prompt[:30]}': {'; '.join(why)}")

print("SANITY PASS" if not failures else "SANITY FAIL: " + " | ".join(failures), flush=True)
sys.exit(1 if failures else 0)
