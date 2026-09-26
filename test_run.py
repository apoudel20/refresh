"""Quick test run — Suzanne in the live Blender session."""
import sys
sys.path.insert(0, "/Users/pranavagrawal/Documents/refresh")

from blender_agent import BlenderAgent, AgentTraits, Emitter, EvaluatorClient
from blender_agent.agent import AgentConfig
from blender_agent.mcp_connector import MCPConfig
from blender_agent.pointcloud import PointCloudConfig

evaluator = EvaluatorClient.from_claude(model="claude-haiku-4-5-20251001")

cfg = AgentConfig(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    llm_backend="openrouter",
    max_tokens=1024,
    mcp=MCPConfig(host="localhost", port=9876),
    workspace="/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test",
    # skip depth estimation — no Replicate key
    pointcloud=PointCloudConfig(backend="local_depth_png"),
)

traits = AgentTraits(
    max_iterations=12,
    target_score=0.75,
    plateau_window=4,   # give agent time to fix scene issues before plateau triggers
    # disable pointcloud tool so agent goes straight to bpy
    allowed_tools=[
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
        "evaluate_render",
    ],
    persona=(
        "Suzanne is already in the scene. "
        "First check the scene (blender_get_scene_info) and ensure there is at least one light. "
        "If there are no lights, add a sun lamp and an area fill light via blender_execute_python. "
        "Then re-render from 4 angles into workspace/suzanne_test/ with blender_render. "
        "After that, call evaluate_render with the new renders and object_name='Suzanne'. "
        "Act on the feedback and iterate. Never reset the scene."
    ),
    render_angles=[(0,0,0),(0,45,0),(0,90,0),(0,135,0)],
    render_resolution=(512, 512),
)

emitter = Emitter("stdout", "/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test/agent.log")

agent = BlenderAgent(evaluator=evaluator, config=cfg)

result, trace = agent.run(
    goal="Add Suzanne to the scene, apply a simple orange PBR material, render from 4 angles, and evaluate.",
    reference={"image_path": "/tmp/suzanne_refs/front.png"},
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
