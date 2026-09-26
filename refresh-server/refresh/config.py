"""Settings, read once from the environment (.env in the folder you start from, or any parent)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass
class Settings:
    # Where runs, node snapshots, renders and GLBs live (served under /files).
    workspace: Path = field(default_factory=lambda: Path(_env("REFRESH_WORKSPACE", str(REPO_ROOT / "workspace"))))
    # Absolute base URL the browser uses to reach this API (for GLB / image URLs).
    public_url: str = field(default_factory=lambda: _env("REFRESH_PUBLIC_URL", "http://localhost:8000").rstrip("/"))
    cors_origins: list[str] = field(default_factory=lambda: _env(
        "REFRESH_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","))

    # Agents (subagents = lineage nodes). Default "claude_code": the Claude Code CLI (`claude -p`) on your Claude
    # subscription, no API key. Alternatives: "anthropic" (ANTHROPIC_API_KEY), "openrouter", "openai".
    agent_backend: str = field(default_factory=lambda: _env("REFRESH_AGENT_BACKEND", "claude_code"))
    # Opus 5.5 by default; "sonnet", "opus" or any full model name also work.
    agent_model: str = field(default_factory=lambda: os.getenv("REFRESH_AGENT_MODEL") or {
        "openrouter": "anthropic/claude-opus-5.5", "openai": "gpt-4o"}.get(
        _env("REFRESH_AGENT_BACKEND", "claude_code"), "claude-opus-5-5"))
    agent_effort: str = field(default_factory=lambda: _env("REFRESH_AGENT_EFFORT", "high"))
    agent_max_iterations: int = field(default_factory=lambda: int(_env("REFRESH_AGENT_MAX_ITERATIONS", "6")))
    agent_target_score: float = field(default_factory=lambda: float(_env("REFRESH_AGENT_TARGET", "0.85")))

    # Main model (lineage generator). Default "claude_code" as well; "openrouter" uses OPENROUTER_API_KEY.
    generator_backend: str = field(default_factory=lambda: _env("REFRESH_GENERATOR_BACKEND", "claude_code"))
    generator_model: str = field(default_factory=lambda: os.getenv("REFRESH_GENERATOR_MODEL") or (
        "claude-opus-5-5" if _env("REFRESH_GENERATOR_BACKEND", "claude_code") == "claude_code"
        else "anthropic/claude-opus-5.5"))
    generations: int = field(default_factory=lambda: int(_env("REFRESH_GENERATIONS", "3")))
    k: int = field(default_factory=lambda: int(_env("REFRESH_K", "3")))
    memory: bool = field(default_factory=lambda: _env("REFRESH_MEMORY", "on") == "on")
    seed: int = field(default_factory=lambda: int(_env("REFRESH_SEED", "0")))

    # imagegen backend for agents' image tools (None = imagegen's auto choice).
    imagegen_backend: str | None = field(default_factory=lambda: os.getenv("IMAGEGEN_BACKEND") or None)

    # Stage: long side of the scored render in pixels.
    stage_resolution: int = field(default_factory=lambda: int(_env("REFRESH_STAGE_RESOLUTION", "768")))

    def url(self, path: str | Path) -> str:
        """Absolute URL for a file inside the workspace."""
        rel = Path(path).resolve().relative_to(self.workspace.resolve())
        return f"{self.public_url}/files/{rel.as_posix()}"


settings = Settings()
os.environ.setdefault("LINEAGE_LLM", settings.generator_backend)  # lineage.llm picks its backend from this
