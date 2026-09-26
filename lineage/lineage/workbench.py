"""Workbench adapters implementing the spec §9 contract: tools(), call(), evaluate()."""
import os
import random
import time

import requests

from .hashing import canon, content_hash

# Hidden "good pipeline" the mock eval rewards, so the search has a real gradient to climb.
_IDEAL = ["segment", "depth_estimate", "coarse_mesh", "mesh_refine", "texture_bake"]


class MockWorkbench:
    """Fake 3D-reconstruction tools + eval, so the harness runs before the real workbench is wired in."""
    version = "mock-1"
    eval_version = "mock-eval-1"
    _TOOLS = [
        ("segment", "Cut the object out of the background image"),
        ("depth_estimate", "Estimate a depth map from the image"),
        ("coarse_mesh", "Build a coarse mesh from depth or silhouette"),
        ("mesh_refine", "Refine a mesh against the input silhouette"),
        ("texture_bake", "Project image colours onto the mesh"),
        ("silhouette_check", "Compare a mesh render to the input silhouette"),
        ("upscale", "Upscale the input image"),
    ]

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def tools(self):
        return [{"tool_id": t, "version": "1", "description": d, "deterministic": True} for t, d in self._TOOLS]

    def call(self, tool_id, version, args, inputs):
        time.sleep(float(os.getenv("MOCK_DELAY", "0")))  # makes mock runs watchable in the UI
        chain = max((i.get("chain", []) for i in inputs), key=len, default=[])
        chain = chain + [tool_id]
        content = f"{tool_id}@{version}({canon(args)})<-{sorted(i['hash'] for i in inputs)}"
        h = content_hash(content)
        return {"output_ref": f"mock:{h[:12]}", "output_hash": h, "chain": chain,
                "summary": f"{tool_id} produced artifact {h[:8]} (pipeline: {' → '.join(chain)})",
                "cost_usd": 0.001, "error": None}

    def evaluate(self, task_id, structure_hash, artifacts):
        best = 0.0
        for a in artifacts:
            chain, i = a.get("chain", []), 0
            for step in chain:  # longest in-order match against the ideal pipeline
                if i < len(_IDEAL) and step == _IDEAL[i]:
                    i += 1
            waste = max(0, len(chain) - i) * 0.03
            best = max(best, i / len(_IDEAL) - waste)
        fitness = min(1.0, max(0.0, best + self.rng.uniform(-0.04, 0.04)))
        return {"fitness": fitness, "metrics": {"pipeline_match": best}, "per_node": {},
                "eval_version": self.eval_version, "cost_usd": 0.0}


class HttpEval:
    """The eval engine, when it lives apart from the workbench: POST {base}/eval (spec §9)."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.eval_version = "http"

    def evaluate(self, task_id, structure_hash, artifacts):
        r = requests.post(f"{self.base}/eval", json={"task_id": task_id, "structure_hash": structure_hash,
                                                      "artifacts": artifacts}, timeout=600)
        out = r.json()
        self.eval_version = out.get("eval_version", self.eval_version)
        return out


def make_workbench(workbench="mock", eval_url=None, seed=0):
    """Tools from `workbench` ('mock' or URL); evals from `eval_url` if given, else from the workbench."""
    wb = MockWorkbench(seed) if workbench == "mock" else HttpWorkbench(workbench)
    if eval_url:
        ev = HttpEval(eval_url)
        wb.evaluate, wb.eval_version = ev.evaluate, ev.eval_version
    return wb


class HttpWorkbench:
    """The real workbench over HTTP, same shapes as the mock (spec §9)."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.version = "http"
        self.eval_version = "http"

    def tools(self):
        return requests.get(f"{self.base}/tools", timeout=30).json()

    def call(self, tool_id, version, args, inputs):
        r = requests.post(f"{self.base}/call", json={"tool_id": tool_id, "version": version,
                                                      "args": args, "inputs": inputs}, timeout=600)
        return r.json()

    def evaluate(self, task_id, structure_hash, artifacts):
        r = requests.post(f"{self.base}/eval", json={"task_id": task_id, "structure_hash": structure_hash,
                                                      "artifacts": artifacts}, timeout=600)
        out = r.json()
        self.eval_version = out.get("eval_version", self.eval_version)
        return out
