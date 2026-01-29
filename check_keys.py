#!/usr/bin/env python3
"""Quick check if Anthropic API key and base URL work. Run: python check_keys.py"""
import os
import json

# Load .env manually
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'").split("#")[0].strip()

key = os.environ.get("ANTHROPIC_API_KEY", "")
base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")

if not key or not base:
    print("Missing ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL in .env")
    exit(1)

try:
    import urllib.request
    url = base + "/messages" if "bedrock" in base.lower() else base
    body = json.dumps({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say OK"}],
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    r = urllib.request.urlopen(req, timeout=15)
    print("ANTHROPIC: OK (status %s) – key works, request succeeded" % r.status)
except Exception as e:
    if hasattr(e, "code"):
        print("ANTHROPIC: HTTP %s – %s" % (e.code, e.reason))
        if e.code == 402:
            print("  -> No funds/credits left")
    else:
        print("ANTHROPIC: Error –", type(e).__name__, str(e)[:120])
