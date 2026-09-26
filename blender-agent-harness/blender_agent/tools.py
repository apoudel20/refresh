"""
Tool definitions for the BlenderAgent, in Anthropic's input_schema format (the canonical
source). ``openai_tools()`` converts them to OpenAI function-calling format;
``anthropic_tools()`` returns them for the Messages API.

The same names are the "tools" lineage genomes pick from, so a structure's agent can be
restricted to a subset (``AgentTraits.allowed_tools``).
"""

from typing import Any

# Tools every agent gets regardless of its genome: it must be able to look at the scene
# and score its own work.
ALWAYS_ALLOWED = ("blender_get_scene_info", "evaluate_render")


def _to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]},
    }


def openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    return [_to_openai(t) for t in select(names)]


def anthropic_tools(names: list[str] | None = None, eager_input_streaming: bool = False) -> list[dict[str, Any]]:
    out = []
    for t in select(names):
        d = {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        if eager_input_streaming:
            d["eager_input_streaming"] = True
        out.append(d)
    return out


def select(names: list[str] | None) -> list[dict[str, Any]]:
    if not names:
        return list(TOOL_DEFINITIONS)
    allowed = set(names) | set(ALWAYS_ALLOWED)
    return [t for t in TOOL_DEFINITIONS if t["name"] in allowed]


def registry() -> list[dict[str, Any]]:
    """Tool list in lineage's registry shape: [{tool_id, version, description}]."""
    return [{"tool_id": t["name"], "version": "1", "description": t["description"].split(". ")[0]}
            for t in TOOL_DEFINITIONS if t["name"] not in ALWAYS_ALLOWED]


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # ── Blender scene ────────────────────────────────────────────────
    {
        "name": "blender_export_file",
        "description": "Export the current mesh/scene to disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "file_format": {"type": "string"},
                "object_names": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path"],
        },
    },
    {
        "name": "blender_execute_python",
        "description": (
            "Run Python code inside Blender (bpy). Use for any modelling operation not covered by other tools: "
            "adding primitives, editing meshes with bmesh, modifiers, materials, sculpt-like deformations. "
            "Set __result__ (or result = {...}) to return data; print() output is returned too. Never touch the "
            "'RefreshStage' collection, the StageCam camera or stage lights, and never reset or reload the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to execute."}},
            "required": ["code"],
        },
    },
    {
        "name": "blender_get_scene_info",
        "description": "Return names, types, transforms, dimensions and materials of all objects in the scene.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "blender_smooth_mesh",
        "description": "Apply Laplacian-style smoothing to a mesh object.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "iterations": {"type": "integer"},
                "factor": {"type": "number"},
            },
            "required": ["object_name"],
        },
    },
    {
        "name": "blender_apply_subdivision",
        "description": "Apply a subdivision surface modifier to increase mesh resolution.",
        "input_schema": {
            "type": "object",
            "properties": {"object_name": {"type": "string"}, "levels": {"type": "integer"}},
            "required": ["object_name"],
        },
    },
    {
        "name": "blender_unwrap_uv",
        "description": "Unwrap UVs on a mesh so textures can be applied.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "method": {"type": "string", "enum": ["SMART_PROJECT", "ANGLE_BASED", "CONFORMAL"]},
            },
            "required": ["object_name"],
        },
    },
    {
        "name": "blender_get_topology_stats",
        "description": "Return vertex/edge/face/quad counts and the number of non-manifold edges for a mesh.",
        "input_schema": {
            "type": "object",
            "properties": {"object_name": {"type": "string"}},
            "required": ["object_name"],
        },
    },
    {
        "name": "blender_get_vertex_positions",
        "description": "Return [[x,y,z],...] for up to 2000 vertices of a named mesh.",
        "input_schema": {
            "type": "object",
            "properties": {"object_name": {"type": "string"}},
            "required": ["object_name"],
        },
    },
    # ── Render & depth ───────────────────────────────────────────────
    {
        "name": "blender_render",
        "description": (
            "Render the model from orbit angles to inspect it from the sides or back. Returns image paths and "
            "shows you the images. The scored view is fixed; use evaluate_render for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Directory for the PNGs."},
                "camera_angles": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3},
                    "description": "List of [azimuth, elevation] (or [azimuth, elevation, roll]) in degrees; 0 = front.",
                },
                "resolution": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                "engine": {"type": "string", "enum": ["EEVEE", "CYCLES"]},
            },
            "required": ["output_path"],
        },
    },
    {
        "name": "blender_get_depth_map",
        "description": "Render a normalised Z-depth pass from the stage camera and save it as PNG. Returns the path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "resolution": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["output_path"],
        },
    },
    # ── Materials & textures (imagegen) ──────────────────────────────
    {
        "name": "generate_texture",
        "description": (
            "Generate a seamless PBR texture image from a text prompt with the imagegen backend. "
            "Use texture_type to target albedo, normal, roughness or metallic maps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "output_path": {"type": "string"},
                "texture_type": {"type": "string", "enum": ["albedo", "normal", "roughness", "metallic"]},
                "size": {"type": "integer", "description": "Output resolution in pixels (square)."},
            },
            "required": ["prompt", "output_path"],
        },
    },
    {
        "name": "generate_texture_from_reference",
        "description": "Generate a texture by editing an existing image (for example a crop of the reference photo).",
        "input_schema": {
            "type": "object",
            "properties": {
                "reference_image_path": {"type": "string"},
                "output_path": {"type": "string"},
                "prompt": {"type": "string"},
                "strength": {"type": "number"},
                "texture_type": {"type": "string", "enum": ["albedo", "normal", "roughness", "metallic"]},
            },
            "required": ["reference_image_path", "output_path"],
        },
    },
    {
        "name": "generate_image",
        "description": (
            "Create an image with the imagegen backend: a concept view, a missing angle of the object, a part "
            "close-up or a style reference. Pass reference image paths to keep the look consistent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "output_path": {"type": "string"},
                "references": {"type": "array", "items": {"type": "string"}},
                "size": {"type": "integer"},
            },
            "required": ["prompt", "output_path"],
        },
    },
    {
        "name": "edit_image",
        "description": (
            "Edit an image with an instruction, optionally only inside region [x0, y0, x1, y1] (pixels or 0-1 "
            "fractions): clean up the reference, isolate a part, remove the background, recolour a texture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "instruction": {"type": "string"},
                "output_path": {"type": "string"},
                "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            },
            "required": ["image_path", "instruction", "output_path"],
        },
    },
    {
        "name": "retexture_uv",
        "description": (
            "Repaint a UV-unwrapped object's base-colour atlas so it looks like a style image, keeping the UV "
            "layout intact (imagegen UV pipeline, background Blender). Slow (minutes): use once the shape is right."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "style_image_path": {"type": "string"},
                "materials": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Per-surface prompts, e.g. ["up=black fur", "side=white chest fur", "rest=tan fur"].',
                },
                "instruction": {"type": "string"},
            },
            "required": ["object_name", "style_image_path"],
        },
    },
    {
        "name": "blender_set_material",
        "description": "Assign a texture PNG to a mesh object's material (UV, GENERATED or OBJECT mapping).",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "texture_path": {"type": "string"},
                "mapping": {"type": "string", "enum": ["UV", "GENERATED", "OBJECT"]},
            },
            "required": ["object_name", "texture_path"],
        },
    },
    # ── Evaluation ───────────────────────────────────────────────────
    {
        "name": "evaluate_render",
        "description": (
            "Render the fixed stage camera view and score it against the reference image (render-eval: pixel, "
            "depth, normals, silhouette, edges, colour), then check the model from all around (solidity: a "
            "closed surface with real depth; a relief or a shell open at the back scores low). Returns the "
            "overall score (front match x solidity factor), per-step scores, the weakest parts with "
            "explanations, the stage render and a turntable strip (back faces red). Call after every "
            "significant change. A scene that loads the reference photo scores 0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "What you changed since the last evaluation."},
                "object_name": {"type": "string", "description": "Main mesh, for topology stats (optional)."},
            },
            "required": [],
        },
    },
]
