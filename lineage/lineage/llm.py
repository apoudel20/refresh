"""Minimal chat call returning parsed JSON. Returns None when no backend is available or the call fails (callers fall back).

Backends (LINEAGE_LLM):
  claude_code  the Claude Code CLI (`claude -p`): runs on the user's Claude subscription, no API key
  openrouter   OpenRouter chat completions (OPENROUTER_API_KEY)
Unset: openrouter when OPENROUTER_API_KEY is set, otherwise claude_code.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://openrouter.ai/api/v1/chat/completions"
# API credentials would take precedence over the subscription; the CLAUDECODE markers block nested sessions.
_STRIP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
_ALIASES = ("opus", "sonnet", "haiku", "fable")


def backend():
    b = os.getenv("LINEAGE_LLM", "").strip().lower()
    return b or ("openrouter" if os.getenv("OPENROUTER_API_KEY") else "claude_code")


def claude_bin():
    for cand in (os.getenv("CLAUDE_BIN"), shutil.which("claude"), str(Path.home() / ".local/bin/claude")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def enabled():
    if backend() == "claude_code":
        return claude_bin() is not None
    return bool(os.getenv("OPENROUTER_API_KEY"))


def ask_json(system, user, model=None, temperature=0.4, max_tokens=4000):
    if not enabled():
        return None
    try:
        if backend() == "claude_code":
            text = _ask_claude_code(system, user, model)
        else:
            text = _ask_openrouter(system, user, model, temperature, max_tokens)
    except Exception as e:  # a failed proposal call must not end the search
        print(f"WARNING: lineage LLM call failed ({backend()}): {e}", flush=True)
        return None
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def _ask_openrouter(system, user, model, temperature, max_tokens):
    r = requests.post(URL, timeout=180, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      json={"model": model or os.getenv("MODEL", "openai/gpt-4o-mini"),
                            "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _ask_claude_code(system, user, model):
    cmd = [claude_bin(), "-p", "--output-format", "json", "--system-prompt", system,
           "--tools", "", "--strict-mcp-config", "--permission-mode", "dontAsk",
           "--max-turns", "1", "--no-session-persistence", "--disable-slash-commands"]
    name = claude_model(model)
    if name:
        cmd += ["--model", name]
    effort = os.getenv("LINEAGE_CLAUDE_EFFORT", "high")  # thinking effort: low | medium | high | xhigh | max
    if effort:
        cmd += ["--effort", effort]
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV and (v or not k.startswith("CLAUDE_"))}
    out = subprocess.run(cmd, input=user, capture_output=True, text=True, env=env, cwd=tempfile.gettempdir(),
                         timeout=float(os.getenv("LINEAGE_LLM_TIMEOUT", "600")))
    try:
        res = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude -p exited {out.returncode}: {(out.stderr or out.stdout).strip()[:300]}") from None
    if res.get("is_error"):
        raise RuntimeError(str(res.get("result") or res.get("subtype"))[:300])
    return str(res.get("result") or "")


def claude_model(model):
    """CLI model name: "anthropic/claude-opus-5" -> "claude-opus-5"; non-Claude names -> None (the CLI default)."""
    model = (model or os.getenv("LINEAGE_CLAUDE_MODEL", "")).strip()
    if model.startswith("anthropic/"):
        model = model.split("/", 1)[1]
    if model.startswith("claude-"):
        return re.sub(r"(\d)\.(\d)", r"\1-\2", model)  # OpenRouter-style "claude-opus-5.5" -> "claude-opus-5-5"
    return model if model in _ALIASES else None
