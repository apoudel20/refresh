"""Quick test run — reconstruct Suzanne from reference images, no pointcloud step."""
import os
import sys
sys.path.insert(0, "/Users/pranavagrawal/Documents/refresh")

from blender_agent import BlenderAgent, AgentTraits, Emitter, EvaluatorClient
from blender_agent.agent import AgentConfig
from blender_agent.emit import MongoSink
from blender_agent.mcp_connector import MCPConfig
from blender_agent.pointcloud import PointCloudConfig
from blender_agent.texture_gen import TextureGenConfig

evaluator = EvaluatorClient.from_openai(model="gpt-4o-mini")

cfg = AgentConfig(
    model="gpt-4o-mini",
    llm_backend="openai",
    max_tokens=1024,
    mcp=MCPConfig(host="localhost", port=9876),
    workspace="/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test",
    texture=TextureGenConfig(backend="openai"),
    pointcloud=PointCloudConfig(
        backend="multi_view",
        multi_view_image_paths=[
            "/tmp/suzanne_refs/front.png",
            "/tmp/suzanne_refs/side_right.png",
            "/tmp/suzanne_refs/back_left.png",
            "/tmp/suzanne_refs/side_left.png",
        ],
        multi_view_angles_deg=[0.0, 90.0, 135.0, -90.0],
        multi_view_resolution=128,
    ),
)

traits = AgentTraits(
    max_iterations=12,
    target_score=0.75,
    plateau_window=4,
    allowed_tools=[
        "image_to_pointcloud",
        "blender_import_file",
        "blender_execute_python",
        "blender_get_scene_info",
        "blender_render",
        "blender_get_depth_map",
        "blender_get_topology_stats",
        "blender_get_vertex_positions",
        "blender_smooth_mesh",
        "blender_apply_subdivision",
        "blender_unwrap_uv",
        "blender_set_material",
        "generate_texture",
        "generate_texture_from_reference",
        "evaluate_render",
    ],
    persona=(
        "You have front, side-left, side-right and back-left reference images of Suzanne the monkey head. "
        "Workflow:\n"
        "1. Run image_to_pointcloud with "
        "image_path='/tmp/suzanne_refs/front.png' and "
        "output_ply_path='/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test/suzanne.ply' "
        "to bootstrap geometry from the reference projections.\n"
        "2. Import the PLY into Blender with blender_import_file.\n"
        "3. Check the scene (blender_get_scene_info). Ensure at least one sun lamp exists; "
        "if not, add one via blender_execute_python.\n"
        "4. Clean up the mesh: blender_smooth_mesh then blender_apply_subdivision.\n"
        "5. Unwrap UVs (blender_unwrap_uv).\n"
        "6. Generate an albedo texture with generate_texture_from_reference "
        "(reference_image_path='/tmp/suzanne_refs/front.png').\n"
        "7. Apply it (blender_set_material).\n"
        "8. Render from 4 angles (blender_render).\n"
        "9. Evaluate (evaluate_render) and act on feedback. Never reset the scene."
    ),
    render_angles=[(0, 0, 0), (0, 45, 0), (0, 90, 0), (0, 135, 0)],
    render_resolution=(512, 512),
)

sinks = ["stdout", "/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test/agent.log"]
mongo_uri = os.environ.get("MONGO_URI")
if mongo_uri:
    sinks.append(MongoSink(mongo_uri))
emitter = Emitter(*sinks)

agent = BlenderAgent(evaluator=evaluator, config=cfg)

result, trace = agent.run(
    goal=(
        "Reconstruct Suzanne the monkey head using the provided reference images. "
        "Generate a texture from the reference, apply it, render from 4 angles, and evaluate."
    ),
    reference={
        "image_path":       "/tmp/suzanne_refs/front.png",
        "side_left_path":   "/tmp/suzanne_refs/side_left.png",
        "side_right_path":  "/tmp/suzanne_refs/side_right.png",
        "back_left_path":   "/tmp/suzanne_refs/back_left.png",
    },
    traits=traits,
    emitter=emitter,
)

print("\n--- TRACE SUMMARY ---")
print("Tool counts    :", trace.tool_counts())
print("Score progress :", trace.score_progression())
print("Stop reason    :", trace.stop_reason)
print("Errors         :", [(e.tool, e.error) for e in trace.errors()])
if result:
    print(f"Final score    : {result.overall_score:.3f}")
