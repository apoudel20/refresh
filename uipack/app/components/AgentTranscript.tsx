"use client";

import { useEffect, useRef } from "react";
import type { AgentEvent } from "../lib/harness-client";

const pct = (v?: number | null) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const time = (ts?: number) => (ts ? new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "");

function preview(args?: Record<string, unknown>) {
  if (!args) return "";
  const first = Object.entries(args).find(([, v]) => typeof v === "string" || typeof v === "number");
  if (!first) return "";
  const text = String(first[1]).replace(/\s+/g, " ");
  return `${first[0]}: ${text.length > 80 ? `${text.slice(0, 80)}…` : text}`;
}

function Args({ args }: { args?: Record<string, unknown> }) {
  if (!args || !Object.keys(args).length) return <div className="tx-empty">no arguments</div>;
  return <>{Object.entries(args).map(([k, v]) => <div key={k} className="tx-arg">
    <span>{k}</span>
    <pre>{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</pre>
  </div>)}</>;
}

function Entry({ e }: { e: AgentEvent }) {
  switch (e.event) {
    case "run_start":
      return <div className="tx-divider"><b>{e.role || e.node_id}</b> agent started{e.model ? ` · ${e.model}` : ""}<small>{time(e.ts)}</small></div>;
    case "message":
      return <div className="tx-msg"><span className="tx-who">{e.role || "agent"}</span><p>{e.text}</p></div>;
    case "thinking":
      return <details className="tx-thinking"><summary>Thinking</summary><p>{e.text}</p></details>;
    case "tool_call":
      return <details className="tx-call"><summary><b>{e.tool}</b><span>{preview(e.args)}</span></summary><Args args={e.args} /></details>;
    case "tool_result":
      if (!e.ok) return <div className="tx-error"><b>{e.tool} failed</b><pre>{e.error}</pre></div>;
      return <details className="tx-result"><summary>{e.tool} returned{e.duration_ms != null ? ` · ${(e.duration_ms / 1000).toFixed(1)}s` : ""}</summary><pre>{e.result}</pre></details>;
    case "score":
      return <div className="tx-score"><b>score {pct(e.overall)}</b>{(e.top_feedback || []).map((f, i) => <span key={i}>{f}</span>)}</div>;
    case "stop":
      return <div className="tx-divider"><b>{e.role || e.node_id}</b> finished · {e.reason}<small>{time(e.ts)}</small></div>;
    default:
      return null;
  }
}

export default function AgentTranscript({ events, title }: { events: AgentEvent[]; title: string }) {
  const box = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    if (box.current && pinned.current) box.current.scrollTop = box.current.scrollHeight;
  }, [events.length]);
  const shown = events.filter(e => e.event !== "run_end" && e.event !== "artifact");
  return <div className="agent-transcript">
    <div className="tx-head"><span>{title}</span><small>{shown.length} entries · live</small></div>
    <div className="tx-body" ref={box} onScroll={ev => {
      const el = ev.currentTarget;
      pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    }}>
      {shown.map((e, i) => <Entry key={`${e.ts}-${i}`} e={e} />)}
      {!shown.length && <div className="tx-empty">Nothing recorded for this agent yet.</div>}
    </div>
  </div>;
}
