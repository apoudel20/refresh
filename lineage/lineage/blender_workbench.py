"""Workbench over the team's Blender harness (spec §9): GET /tools, POST /call, POST /eval.

Every tool is a stateless step: load the input file into a clean scene, do one thing, export a new file.
Outputs are file paths plus content hashes, so Lineage's node cache stays correct even though Blender is stateful.

Run on the machine with Blender + the MCP add-on (port 9876):
    set BLENDER_HARNESS=<repo>/blender-agent-harness   (the image-and-score-skills branch checkout)
    set LINEAGE_REFERENCE=<path to the reference image>
    python -m uvicorn lineage.blender_workbench:app --port 8150
Then: python -m lineage.search --scope blender-1 --workbench http://localhost:8150 --task-input <same image> --k 3
"""
import hashlib
import os
import pathlib
import sys
import threading
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
HARNESS = pathlib.Path(os.getenv("BLENDER_HARNESS", "../blender-agent-harness")).resolve()
for p in (HARNESS, HARNESS / "blender_agent" / "imagegen-skill", HARNESS / "blender_agent" / "render-eval-skill"):
    sys.path.insert(0, str(p))

from blender_agent.evaluator import EvaluatorClient, RenderPayload  # noqa: E402
from blender_agent.mcp_connector import BlenderMCPConnector, MCPConfig  # noqa: E402

WORK = pathlib.Path(os.getenv("LINEAGE_WORKDIR", "lineage_work")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
REFERENCE = os.getenv("LINEAGE_REFERENCE", "")
app = FastAPI()
_lock = threading.Lock()
_blender = None

MESHES = "meshes = [o for o in bpy.data.objects if o.type == 'MESH']\n"


def _mod(kind, **settings):
    """Add a modifier of `kind` to every mesh and apply it."""
    sets = "".join(f"    m.{k} = {v!r}\n" for k, v in settings.items())
    return (MESHES + "for o in meshes:\n    bpy.context.view_layer.objects.active = o\n"
            f"    m = o.modifiers.new('lineage', {kind!r})\n{sets}    bpy.ops.object.modifier_apply(modifier=m.name)\n")


# tool_id -> (description, python run inside Blender on the imported input, output format). Registry order is
# the default plan order when no LLM picks the calls.
STEPS = {
    "depth_pointcloud": ("Estimate depth from the reference image and lift it to a coloured point cloud", None, "ply"),
    "hull_mesh": ("Wrap a point cloud in a closed mesh (convex hull)", MESHES +
                  "import bmesh\nfor o in meshes:\n    bm = bmesh.new(); bm.from_mesh(o.data)\n"
                  "    bmesh.ops.convex_hull(bm, input=bm.verts)\n    bm.to_mesh(o.data); bm.free()\n", "ply"),
    "remesh": ("Rebuild the mesh as clean, even topology (smooth remesh)", _mod("REMESH", mode="SMOOTH", octree_depth=6), "ply"),
    "smooth": ("Laplacian-smooth the mesh surface", _mod("LAPLACIANSMOOTH", iterations=5, lambda_factor=0.5), "ply"),
    "subdivide": ("Subdivide the mesh for more detail", _mod("SUBSURF", levels=1, render_levels=1), "ply"),
    "decimate": ("Reduce the face count, keeping the shape", _mod("DECIMATE", ratio=0.5), "ply"),
    "uv_unwrap": ("Unwrap UVs so a texture can be applied", MESHES +
                  "for o in meshes:\n    bpy.context.view_layer.objects.active = o; o.select_set(True)\n"
                  "bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')\n"
                  "bpy.ops.uv.smart_project(); bpy.ops.object.mode_set(mode='OBJECT')\n", "ply"),
}
CLEAR = "import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)\n"


def blender():
    global _blender
    if _blender is None:
        _blender = BlenderMCPConnector(MCPConfig(host=os.getenv("BLENDER_HOST", "localhost"),
                                                 port=int(os.getenv("BLENDER_PORT", "9876"))))
        _blender.connect()
    return _blender


def _file_hash(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _load(path):
    b = blender()
    b.execute_python(CLEAR)
    if path and os.path.isfile(path):
        b.import_file(path)


@app.get("/tools")
def tools():
    return [{"tool_id": t, "version": "1", "description": d, "deterministic": t != "depth_pointcloud"}
            for t, (d, _, _) in STEPS.items()]


@app.post("/call")
def call(body: dict):
    tool_id, inputs = body["tool_id"], body.get("inputs") or []
    src = next((i for i in inputs if os.path.isfile(str(i.get("ref", "")))), None)
    chain = max((i.get("chain", []) for i in inputs), key=len, default=[]) + [tool_id]
    desc, code, fmt = STEPS[tool_id]
    out = WORK / f"{tool_id}-{uuid.uuid4().hex[:10]}.{fmt}"
    try:
        with _lock:
            if tool_id == "depth_pointcloud":
                from blender_agent.pointcloud import ImageToPointCloud
                ImageToPointCloud().convert(REFERENCE, str(out))
            else:
                if not src:
                    raise ValueError("needs a mesh or point cloud input")
                _load(src["ref"])
                blender().execute_python("import bpy\n" + code)
                blender().export_file(str(out), fmt)
        h = _file_hash(out)
        return {"output_ref": str(out), "output_hash": h, "chain": chain, "cost_usd": 0.0, "error": None,
                "summary": f"{tool_id} → {out.name} (pipeline: {' → '.join(chain)})"}
    except Exception as e:  # pass the input through so the structure still gets scored
        passthru = src or (inputs[0] if inputs else {"ref": "", "hash": "none"})
        return {"output_ref": passthru["ref"], "output_hash": passthru["hash"], "chain": chain, "cost_usd": 0.0,
                "error": str(e)[:300], "summary": f"{tool_id} failed: {str(e)[:120]}"}


@app.post("/eval")
def evaluate(body: dict):
    """Render the structure's output mesh and score it against the reference with the team's evaluator."""
    arts = [a for a in body.get("artifacts", []) if os.path.isfile(str(a.get("ref", "")))]
    if not arts:
        return {"fitness": 0.0, "metrics": {"reason": "no mesh produced"}, "per_node": {}, "eval_version": "render-eval-1"}
    shots = WORK / f"render-{uuid.uuid4().hex[:10]}"
    with _lock:
        _load(arts[0]["ref"])
        renders = blender().render(str(shots), resolution=(384, 384), engine=os.getenv("BLENDER_ENGINE", "CYCLES"))
    ev = EvaluatorClient.from_openrouter().evaluate(RenderPayload(render_images=renders, ply_path=arts[0]["ref"]),
                                                    {"image_path": REFERENCE} if REFERENCE else {})
    return {"fitness": float(ev.overall_score), "per_node": {}, "eval_version": "render-eval-1",
            "metrics": {"visual_fidelity": ev.visual_fidelity, "topology_quality": ev.topology_quality,
                        "depth_alignment": ev.depth_alignment, "renders": renders},
            "feedback": ev.feedback}
