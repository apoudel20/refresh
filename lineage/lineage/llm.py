"""Minimal OpenRouter chat call returning parsed JSON. Returns None when no key is set (callers fall back)."""
import json
import os
import re

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"


def enabled():
    return bool(os.getenv("OPENROUTER_API_KEY"))


def ask_json(system, user, model=None, temperature=0.4, max_tokens=4000):
    if not enabled():
        return None
    r = requests.post(URL, timeout=180, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      json={"model": model or os.getenv("MODEL", "openai/gpt-4o-mini"),
                            "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
