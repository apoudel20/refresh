"""
Standalone agent runner.  Single entry point: run(model_config, goal, ...).

from agent_runner import run, ModelConfig, AgentTraits

result, trace = run(
    ModelConfig(model="gpt-4o-mini", backend="openai", evaluator_backend="claude"),
    goal="Reconstruct Suzanne the monkey head",
    reference_image="/tmp/refs/front.png",
    workspace="/tmp/my_run",
)
print(trace.score_progression())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from blender_agent.agent import (
    AgentConfig,
    AgentTraits,
    BlenderAgent,
    HarnessTrace,
)
from blender_agent.emit import Emitter, MongoSink
from blender_agent.evaluator import EvaluationResult, EvaluatorClient
from blender_agent.mcp_connector import MCPConfig
from blender_agent.pointcloud import PointCloudConfig
from blender_agent.texture_gen import TextureGenConfig


@dataclass
class ModelConfig:
    """
    All model/infrastructure choices needed to spin up a run.
    Every field has a working default — only override what you need.
    """

    # ── Agent LLM ──────────────────────────────────────────────────
    model: str = "gpt-4o-mini"
    backend: Literal["openai", "openrouter", "anthropic"] = "openai"
    api_key: str = ""           # falls back to env var if empty
    max_tokens: int = 4096

    # ── Evaluator ──────────────────────────────────────────────────
    evaluator_backend: Literal["openai", "openrouter", "claude", "codex", "http"] = "openai"
    evaluator_model: str = "gpt-4o"
    evaluator_api_key: str = ""
    evaluator_url: str = ""     # required only for evaluator_backend="http"

    # ── Blender MCP socket ─────────────────────────────────────────
    mcp_host: str = "localhost"
    mcp_port: int = 9876
    mcp_timeout: float = 180.0

    # ── Texture generation ─────────────────────────────────────────
    texture_backend: str = "openai"   # "openai" | "openrouter" | "codex"

    # ── Workspace ──────────────────────────────────────────────────
    workspace: str = "/tmp/blender_agent"


def _build_evaluator(cfg: ModelConfig) -> EvaluatorClient:
    b = cfg.evaluator_backend
    if b == "openai":
        return EvaluatorClient.from_openai(cfg.evaluator_api_key, cfg.evaluator_model)
    if b == "openrouter":
        return EvaluatorClient.from_openrouter(cfg.evaluator_api_key, cfg.evaluator_model)
    if b == "claude":
        return EvaluatorClient.from_claude(cfg.evaluator_api_key, cfg.evaluator_model)
    if b == "codex":
        return EvaluatorClient.from_codex(cfg.evaluator_model or None)
    if b == "http":
        if not cfg.evaluator_url:
            raise ValueError("ModelConfig.evaluator_url is required for evaluator_backend='http'")
        return EvaluatorClient.from_http(cfg.evaluator_url, cfg.evaluator_api_key)
    raise ValueError(f"Unknown evaluator_backend: {b!r}")


def _build_emitter(
    log_to: str | list[str] | None,
    mongo_uri: str | None,
    workspace: str,
) -> Emitter:
    sinks: list[Any] = []

    if log_to is None:
        pass
    elif isinstance(log_to, str):
        sinks.append(log_to)
    else:
        sinks.extend(log_to)

    # Auto-add a log file in the workspace if not already writing to a file
    has_file = any(
        isinstance(s, str) and s not in ("stdout", "stderr") for s in sinks
    )
    if not has_file:
        sinks.append(f"{workspace}/agent.log")

    if mongo_uri:
        sinks.append(MongoSink(mongo_uri))

    return Emitter(*sinks) if sinks else Emitter("stdout")


def run(
    model_config: ModelConfig,
    goal: str,
    reference_image: str | None = None,
    reference_mesh: str | None = None,
    traits: AgentTraits | None = None,
    log_to: str | list[str] | None = "stdout",
    mongo_uri: str | None = None,
) -> tuple[EvaluationResult | None, HarnessTrace]:
    """
    Spin up a BlenderAgent run from a ModelConfig.

    Args:
        model_config:    LLM, evaluator, MCP, and workspace settings.
        goal:            Natural-language description of what to build.
        reference_image: Path to a reference PNG the agent and evaluator compare against.
        reference_mesh:  Optional path to a reference mesh for geometry comparison.
        traits:          AgentTraits to control stopping policy, weights, persona, etc.
                         Defaults to AgentTraits() (sensible defaults) if None.
        log_to:          "stdout", "stderr", a file path, or a list of those.
                         A workspace/agent.log file is always added automatically.
                         Pass [] to suppress all logging except the log file.
        mongo_uri:       Optional MongoDB connection string for event streaming.

    Returns:
        (EvaluationResult | None, HarnessTrace)
        EvaluationResult is None if the agent never called evaluate_render.
    """
    ws = model_config.workspace

    agent_cfg = AgentConfig(
        model=model_config.model,
        llm_backend=model_config.backend,
        max_tokens=model_config.max_tokens,
        mcp=MCPConfig(
            host=model_config.mcp_host,
            port=model_config.mcp_port,
            timeout=model_config.mcp_timeout,
        ),
        texture=TextureGenConfig(backend=model_config.texture_backend),
        pointcloud=PointCloudConfig(),
        workspace=ws,
    )

    evaluator = _build_evaluator(model_config)
    emitter = _build_emitter(log_to, mongo_uri, ws)

    reference: dict[str, Any] = {}
    if reference_image:
        reference["image_path"] = reference_image
    if reference_mesh:
        reference["reference_mesh_path"] = reference_mesh

    agent = BlenderAgent(
        evaluator=evaluator,
        config=agent_cfg,
        api_key=model_config.api_key,
    )

    try:
        return agent.run(
            goal=goal,
            reference=reference,
            traits=traits,
            workspace=ws,
            emitter=emitter,
        )
    finally:
        emitter.close()
