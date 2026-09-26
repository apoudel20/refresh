"""
BlenderAgent Dashboard — end-to-end pipeline viewer.

Run with:
    .venv/bin/streamlit run dashboard.py

Reads a newline-delimited JSON log file written by emit.Emitter and a
workspace directory containing renders, textures, and PLY files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BlenderAgent",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar controls ───────────────────────────────────────────────────────

with st.sidebar:
    st.title("BlenderAgent")
    st.caption("End-to-end pipeline viewer")

    workspace = st.text_input("Workspace directory", value="/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test")
    log_file  = st.text_input("Log file", value="/Users/pranavagrawal/Documents/refresh/workspace/suzanne_test/agent.log")

    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_secs = st.slider("Refresh interval (s)", 2, 30, 4)

    st.divider()
    if st.button("Refresh now"):
        st.rerun()

ws   = Path(workspace)
log  = Path(log_file)

# ── Load events ────────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


events = load_events(log)

# ── Derive pipeline stage status ──────────────────────────────────────────

STAGES = [
    ("model",       "Modelling",       "blender_execute_python"),
    ("texture",     "Texture gen",     "generate_texture"),
    ("render",      "Render",          "blender_render"),
    ("evaluate",    "Evaluate",        "evaluate_render"),
]

completed_tools: set[str] = set()
errored_tools:   set[str] = set()
active_tool: str | None = None

for e in events:
    if e.get("event") == "tool_call":
        active_tool = e.get("tool")
    elif e.get("event") == "tool_result":
        t = e.get("tool", "")
        if e.get("ok"):
            completed_tools.add(t)
        else:
            errored_tools.add(t)
        if active_tool == t:
            active_tool = None

run_started  = any(e.get("event") == "run_start"  for e in events)
run_finished = any(e.get("event") == "stop"        for e in events)

score_events = [e for e in events if e.get("event") == "score"]
stop_event   = next((e for e in events if e.get("event") == "stop"), None)
run_end      = next((e for e in events if e.get("event") == "run_end"), None)

# ── Header ─────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("BlenderAgent Pipeline")
    if run_started:
        run_meta = next((e for e in events if e.get("event") == "run_start"), {})
        st.caption(f"Goal: **{run_meta.get('goal', '—')}**   ·   Model: `{run_meta.get('model', '—')}`")
    else:
        st.caption("Waiting for a run to start…")

with col_status:
    if run_finished:
        reason = stop_event.get("reason", "") if stop_event else ""
        score  = stop_event.get("final_score") if stop_event else None
        if score is not None:
            st.metric("Final score", f"{score:.3f}")
        st.success(f"Done — {reason}")
    elif run_started:
        itr = max((e.get("iteration", 0) for e in events if "iteration" in e), default=0)
        st.info(f"Running · iter {itr + 1}")
    else:
        st.warning("No run data")

# ── Pipeline stage bar ─────────────────────────────────────────────────────

st.subheader("Pipeline stages")
stage_cols = st.columns(len(STAGES) + 1)

with stage_cols[0]:
    icon  = "✅" if run_started else "⏳"
    color = "green" if run_started else "gray"
    st.markdown(f":{color}[{icon} **Start**]")

for i, (key, label, tool_name) in enumerate(STAGES):
    with stage_cols[i + 1]:
        if tool_name in errored_tools:
            st.markdown(f":red[❌ **{label}**]")
        elif tool_name in completed_tools:
            st.markdown(f":green[✅ **{label}**]")
        elif active_tool == tool_name:
            st.markdown(f":orange[⏳ **{label}**]")
        else:
            st.markdown(f":gray[⬜ **{label}**]")

st.divider()

# ── Main layout ────────────────────────────────────────────────────────────

left, right = st.columns([1, 1])

# ── Score progression chart ───────────────────────────────────────────────

with left:
    st.subheader("Score progression")
    if score_events:
        iters    = [e["iteration"] + 1 for e in score_events]
        overall  = [e["overall"]   for e in score_events]
        visual   = [e.get("visual",   0) for e in score_events]
        topology = [e.get("topology", 0) for e in score_events]
        depth    = [e.get("depth",    0) for e in score_events]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iters, y=overall,  mode="lines+markers", name="Overall",  line=dict(width=3)))
        fig.add_trace(go.Scatter(x=iters, y=visual,   mode="lines+markers", name="Visual",   line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=iters, y=topology, mode="lines+markers", name="Topology", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=iters, y=depth,    mode="lines+markers", name="Depth",    line=dict(dash="dot")))

        # Target score line
        run_meta = next((e for e in events if e.get("event") == "run_start"), {})
        fig.add_hline(y=0.80, line_dash="dash", line_color="red",
                      annotation_text="target", annotation_position="bottom right")

        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scores yet.")

    # Latest feedback
    if score_events:
        latest = score_events[-1]
        feedback = latest.get("top_feedback", [])
        if feedback:
            st.caption("Latest evaluator feedback")
            for f in feedback:
                st.markdown(f"- {f}")

# ── Turntable (solidity check) ────────────────────────────────────────────

with right:
    st.subheader("Turntable (back faces red)")
    sheets = sorted(ws.glob("**/turntable.png"), key=lambda p: p.stat().st_mtime) if ws.exists() else []
    if sheets:
        st.image(str(sheets[-1]), use_container_width=True)
        st.caption("Model seen from 8 sides, starting at the stage camera. Red = the inside of an open surface.")
    else:
        st.info(f"No turntable yet in `{workspace}` (evaluate_render makes one).")

st.divider()

# ── Render gallery ─────────────────────────────────────────────────────────

st.subheader("Render gallery")

render_imgs = sorted(ws.glob("**/*.png")) if ws.exists() else []
# Separate textures (in a textures/ subdir) from renders
texture_imgs = [p for p in render_imgs if "texture" in str(p) or "tex_" in p.name]
render_only  = [p for p in render_imgs if p not in texture_imgs]

if render_only:
    n_cols = min(4, len(render_only))
    cols   = st.columns(n_cols)
    for i, img in enumerate(render_only):
        with cols[i % n_cols]:
            st.image(str(img), caption=img.name, use_container_width=True)
else:
    st.info(f"No render PNGs found in `{workspace}`.")

if texture_imgs:
    with st.expander(f"Textures ({len(texture_imgs)})"):
        t_cols = st.columns(min(4, len(texture_imgs)))
        for i, img in enumerate(texture_imgs):
            with t_cols[i % len(t_cols)]:
                st.image(str(img), caption=img.name, use_container_width=True)

st.divider()

# ── Tool activity + event log ──────────────────────────────────────────────

col_tools, col_log = st.columns([1, 2])

with col_tools:
    st.subheader("Tool activity")
    if run_end:
        tool_counts = run_end.get("tool_counts", {})
        if tool_counts:
            import pandas as pd
            df = pd.DataFrame(
                sorted(tool_counts.items(), key=lambda x: x[1], reverse=True),
                columns=["Tool", "Calls"],
            )
            st.dataframe(df, hide_index=True, use_container_width=True)
    elif events:
        from collections import Counter
        counts = Counter(
            e["tool"] for e in events if e.get("event") == "tool_call"
        )
        if counts:
            import pandas as pd
            df = pd.DataFrame(
                sorted(counts.items(), key=lambda x: x[1], reverse=True),
                columns=["Tool", "Calls"],
            )
            st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No tool data yet.")

    # Error summary
    errors = [
        e for e in events
        if e.get("event") == "tool_result" and not e.get("ok")
    ]
    if errors:
        st.caption(f"⚠️ {len(errors)} error(s)")
        for err in errors[-5:]:
            st.markdown(f"- `{err.get('tool')}`: {err.get('error', '')[:80]}")

with col_log:
    st.subheader("Event log")

    EVENT_COLORS = {
        "run_start":   "🟢",
        "tool_call":   "🔵",
        "tool_result": "⚪",
        "score":       "🟡",
        "stop":        "🔴",
        "run_end":     "🏁",
    }

    if events:
        # Show most recent first, capped at 60 lines
        shown = list(reversed(events[-60:]))
        lines = []
        for e in shown:
            etype = e.get("event", "?")
            icon  = EVENT_COLORS.get(etype, "⬜")
            ts    = e.get("ts", 0)
            ts_s  = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if ts else "??:??:??"

            if etype == "tool_call":
                tool = e.get("tool", "")
                args = e.get("args", {})
                # Summarise args without dumping huge blobs
                arg_str = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3])
                lines.append(f"{icon} `{ts_s}` **{tool}**({arg_str})")
            elif etype == "tool_result":
                tool = e.get("tool", "")
                ok   = "✅" if e.get("ok") else "❌"
                dur  = e.get("duration_ms", 0)
                lines.append(f"{icon} `{ts_s}` {ok} `{tool}` — {dur:.0f}ms")
            elif etype == "score":
                overall = e.get("overall", 0)
                itr     = e.get("iteration", 0)
                lines.append(f"{icon} `{ts_s}` **score** iter={itr} overall={overall:.3f}")
            elif etype == "run_start":
                lines.append(f"{icon} `{ts_s}` **run started** goal={e.get('goal','')[:60]}")
            elif etype == "stop":
                lines.append(f"{icon} `{ts_s}` **stopped** reason={e.get('reason')} final={e.get('final_score')}")
            elif etype == "run_end":
                lines.append(f"{icon} `{ts_s}` **run ended**")
            else:
                lines.append(f"{icon} `{ts_s}` {etype}")

        log_text = "\n\n".join(lines)
        st.markdown(log_text)
    else:
        st.info("No events yet. Start a run with `python test_run.py` (see sidebar note).")
        st.caption(
            "Point your Emitter at the log file path above:\n"
            "```python\nemitter = Emitter('stdout', '/tmp/suzanne_test/agent.log')\n```"
        )

# ── Auto-refresh ───────────────────────────────────────────────────────────

if auto_refresh and not run_finished:
    time.sleep(refresh_secs)
    st.rerun()
