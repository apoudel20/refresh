"""
Structured event emitter for the agent harness.

Events are newline-delimited JSON written to a sink (stdout, file, or both).
Every event has a common envelope:

  {"ts": <unix_ms>, "event": "<type>", ...fields}

Event types:
  run_start     — agent loop begins
  tool_call     — tool dispatched
  tool_result   — tool returned (or errored)
  score         — evaluate_render result
  stop          — loop ended
  run_end       — final summary
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, TextIO


# ── Sinks ──────────────────────────────────────────────────────────────────

class Emitter:
    """
    Writes newline-delimited JSON events to one or more sinks.

    sinks accepts: "stdout", "stderr", a file path string, or an open TextIO.
    """

    def __init__(self, *sinks: str | TextIO | Path):
        self._sinks: list[TextIO] = []
        self._owned: list[TextIO] = []   # files we opened — close on __exit__

        for s in sinks:
            if s == "stdout":
                self._sinks.append(sys.stdout)
            elif s == "stderr":
                self._sinks.append(sys.stderr)
            elif isinstance(s, (str, Path)):
                p = Path(s)
                p.parent.mkdir(parents=True, exist_ok=True)
                fh = open(p, "a", encoding="utf-8")
                self._sinks.append(fh)
                self._owned.append(fh)
            else:
                self._sinks.append(s)

        if not self._sinks:
            self._sinks.append(sys.stdout)

    def emit(self, event_type: str, **fields: Any) -> None:
        payload = {"ts": int(time.time() * 1000), "event": event_type, **fields}
        line = json.dumps(payload, default=str)
        for sink in self._sinks:
            sink.write(line + "\n")
            sink.flush()

    def close(self) -> None:
        for fh in self._owned:
            fh.close()
        self._owned.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Typed emit helpers ─────────────────────────────────────────

    def run_start(self, goal: str, traits_summary: str, model: str, backend: str) -> None:
        self.emit("run_start",
            goal=goal,
            traits=traits_summary,
            model=model,
            backend=backend,
        )

    def tool_call(self, iteration: int, tool: str, args: dict[str, Any]) -> None:
        # Truncate large args (e.g. bpy code blobs) for readability
        safe_args = {k: (v[:200] + "…") if isinstance(v, str) and len(v) > 200 else v
                     for k, v in args.items()}
        self.emit("tool_call", iteration=iteration, tool=tool, args=safe_args)

    def tool_result(
        self,
        iteration: int,
        tool: str,
        duration_ms: float,
        error: str | None,
        result_summary: str,
    ) -> None:
        self.emit("tool_result",
            iteration=iteration,
            tool=tool,
            duration_ms=round(duration_ms, 1),
            ok=error is None,
            error=error,
            result=result_summary,
        )

    def score(
        self,
        iteration: int,
        overall: float,
        visual: float,
        topology: float,
        depth: float,
        vertices: float,
        feedback: list[str],
        stop_reason_preview: str = "",
    ) -> None:
        self.emit("score",
            iteration=iteration,
            overall=round(overall, 4),
            visual=round(visual, 4),
            topology=round(topology, 4),
            depth=round(depth, 4),
            vertices=round(vertices, 4),
            top_feedback=feedback[:3],
            note=stop_reason_preview,
        )

    def stop(self, reason: str, iterations: int, final_score: float | None) -> None:
        self.emit("stop",
            reason=reason,
            iterations=iterations,
            final_score=round(final_score, 4) if final_score is not None else None,
        )

    def artifact(self, kind: str, path: str, meta: dict[str, Any] | None = None) -> None:
        """Emitted when a file artifact (ply, texture, render) is produced."""
        self.emit("artifact", kind=kind, path=path, **(meta or {}))

    def run_end(self, trace_summary: dict[str, Any]) -> None:
        self.emit("run_end", **trace_summary)


# ── Default no-op emitter ──────────────────────────────────────────────────

class NullEmitter(Emitter):
    """Drops all events — used when caller passes emitter=None."""
    def __init__(self):
        self._sinks = []
        self._owned = []

    def emit(self, *_, **__) -> None:
        pass
