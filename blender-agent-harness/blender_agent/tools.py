"""
Tool definitions for the BlenderAgent — stored in Anthropic's input_schema
format as the canonical source.  Call `openai_tools()` to get the OpenAI
function-calling format.
"""

from typing import Any


def _to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a single Anthropic-style tool dict to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def openai_tools() -> list[dict[str, Any]]:
    return [_to_openai(t) for t in TOOL_DEFINITIONS]


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # ── Point cloud ─────────────────────────────────────────────────
    {
        "name": "image_to_pointcloud",
        "description": (
            "Convert a single RGB image to a PLY point cloud using zero-shot monocular "
            "depth estimation. Use this as the very first step to bootstrap geometry from a "
            "reference photo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the input image."},
                "output_ply_path": {"type": "string", "description": "Where to save the PLY file."},
            },
            "required": ["image_path", "output_ply_path"],
        },
    },

    # ── Blender scene ────────────────────────────────────────────────
    {
        "name": "blender_import_file",
        "description": "Import a mesh or point-cloud file (OBJ, PLY, FBX, glTF) into the Blender scene.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "file_format": {"type": "string", "description": "Optional override (ply, obj, fbx, gltf…)."},
            },
            "required": ["path"],
        },
    },
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
            "Run arbitrary Python code inside Blender's bpy environment. "
            "Use for operations not covered by other tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "blender_get_scene_info",
        "description": "Return names, types, and transforms of all objects in the Blender scene.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "blender_smooth_mesh",
        "description": "Apply Laplacian smoothing to a mesh object.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "iterations": {"type": "integer", "default": 5},
                "factor": {"type": "number", "default": 0.5},
            },
            "required": ["object_name"],
        },
    },
    {
        "name": "blender_apply_subdivision",
        "description": "Apply a subdivision surface modifier to increase mesh resolution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "levels": {"type": "integer", "default": 2},
            },
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
        "description": "Return vertex/edge/face counts and manifold status for a named mesh.",
        "input_schema": {
            "type": "object",
            "properties": {"object_name": {"type": "string"}},
            "required": ["object_name"],
        },
    },
    {
        "name": "blender_get_vertex_positions",
        "description": "Return [[x,y,z],…] for every vertex of a named mesh.",
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
            "Render the Blender scene from one or more camera angles. "
            "Returns a list of saved image paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "camera_angles": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "description": "List of [rx, ry, rz] Euler angles in degrees.",
                },
                "resolution": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                "engine": {"type": "string", "enum": ["CYCLES", "EEVEE"], "default": "CYCLES"},
            },
            "required": ["output_path"],
        },
    },
    {
        "name": "blender_get_depth_map",
        "description": "Render a Z-depth pass and save as PNG. Returns saved path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "resolution": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["output_path"],
        },
    },

    # ── Texture generation ───────────────────────────────────────────
    {
        "name": "generate_texture",
        "description": (
            "Generate a seamless PBR texture image from a text prompt. "
            "Use texture_type to target albedo, normal, roughness, or metallic maps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "output_path": {"type": "string"},
                "texture_type": {
                    "type": "string",
                    "enum": ["albedo", "normal", "roughness", "metallic"],
                    "default": "albedo",
                },
                "negative_prompt": {"type": "string"},
                "size": {"type": "integer", "description": "Output resolution in pixels (square)."},
            },
            "required": ["prompt", "output_path"],
        },
    },
    {
        "name": "generate_texture_from_reference",
        "description": "Generate a texture using an existing image as a style/content seed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reference_image_path": {"type": "string"},
                "output_path": {"type": "string"},
                "prompt": {"type": "string"},
                "strength": {"type": "number", "default": 0.6},
                "texture_type": {"type": "string", "enum": ["albedo", "normal", "roughness", "metallic"]},
            },
            "required": ["reference_image_path", "output_path"],
        },
    },
    {
        "name": "blender_set_material",
        "description": "Assign a texture PNG to a Blender mesh object's material.",
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
            "Submit current render images, depth map, mesh, and geometry data to the "
            "evaluator API. Returns scores and actionable feedback. Call after every "
            "significant refinement iteration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "render_image_paths": {"type": "array", "items": {"type": "string"}},
                "depth_map_path": {"type": "string"},
                "ply_path": {"type": "string"},
                "object_name": {
                    "type": "string",
                    "description": "If provided, topology stats and vertex positions are fetched automatically.",
                },
            },
            "required": ["render_image_paths"],
        },
    },
]
