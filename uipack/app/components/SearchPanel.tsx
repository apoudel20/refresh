"use client";

import { useEffect, useMemo, useState } from "react";
import AgentTranscript from "./AgentTranscript";
import LiveRenders from "./LiveRenders";
import {
  fetchAgentEvents, fetchRun, fetchStructure, fetchStructures, stopRun,
  type AgentEvent, type RunSummary, type StructureCard, type StructureDetail,
} from "../lib/harness-client";

const STEPS = ["pixel", "depth", "normals", "silhouette", "edges", "embedding", "color", "judge", "solidity"];
const ago = (ts?: number) => (ts ? `${Math.max(0, Math.round((Date.now() - ts) / 1000))}s ago` : "");
const pct = (v?: number | null) => (v == null ? "–" : `${Math.round(v * 100)}%`);

function layout(card: StructureCard, width: number, height: number) {
  const preds: Record<string, string[]> = {};
  card.nodes.forEach(n => { preds[n.nodeId] = []; });
  card.edges.forEach(e => { preds[e.to]?.push(e.from); });
  const depth: Record<string, number> = {};
  const d = (id: string, seen: string[] = []): number => {
    if (depth[id] != null) return depth[id];
    if (seen.includes(id)) return 1;
    depth[id] = 1 + Math.max(0, ...(preds[id] || []).map(p => d(p, [...seen, id])));
    return depth[id];
  };
  card.nodes.forEach(n => d(n.nodeId));
  const layers: Record<number, string[]> = {};
  card.nodes.forEach(n => { (layers[depth[n.nodeId]] ||= []).push(n.nodeId); });
  const maxLayer = Math.max(1, ...Object.keys(layers).map(Number));
  const pos: Record<string, { x: number; y: number }> = {};
  Object.entries(layers).forEach(([layer, ids]) => ids.forEach((id, i) => {
    pos[id] = { x: ((Number(layer) - 0.5) * width) / maxLayer, y: ((i + 1) * height) / (ids.length + 1) };
  }));
  return pos;
}

function TeamGraph({ card, selected, onSelect }: { card: StructureCard; selected?: string; onSelect: (id: string) => void }) {
  const W = 520, H = 150;
  const pos = layout(card, W, H);
  return <svg className="team-graph" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Agent team structure">
    {card.edges.map((e, i) => pos[e.from] && pos[e.to] ? <line key={i} x1={pos[e.from].x + 56} y1={pos[e.from].y} x2={pos[e.to].x - 56} y2={pos[e.to].y} /> : null)}
    {card.nodes.map(n => <g key={n.nodeId} className={`team-node${selected === n.nodeId ? " selected" : ""}`} onClick={() => onSelect(n.nodeId)} transform={`translate(${pos[n.nodeId]?.x ?? 0},${pos[n.nodeId]?.y ?? 0})`}>
      <rect x={-56} y={-16} width={112} height={32} rx={7} />
      <text textAnchor="middle" dominantBaseline="central">{n.role}</text>
      <title>{`${n.nodeId}: ${n.tools.join(", ") || "all tools"}`}</title>
    </g>)}
  </svg>;
}

function Curve({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="fitness-curve empty">The fitness curve appears after two scored teams.</div>;
  const W = 600, H = 100;
  let best = 0;
  const bestPts = values.map((v, i) => { best = Math.max(best, v); return `${(i / (values.length - 1)) * W},${H - best * H}`; });
  const raw = values.map((v, i) => `${(i / (values.length - 1)) * W},${H - v * H}`);
  return <div className="fitness-curve"><span>Best match by scored team · {pct(best)}</span>
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Fitness by evaluation">
      <polyline className="raw" points={raw.join(" ")} />
      <polyline className="best" points={bestPts.join(" ")} />
    </svg>
  </div>;
}

export default function SearchPanel({ runId }: { runId?: string }) {
  const [run, setRun] = useState<RunSummary | null>(null);
  const [cards, setCards] = useState<StructureCard[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<StructureDetail | null>(null);
  const [node, setNode] = useState<string | undefined>();
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    const tick = async () => {
      try {
        const [summary, list] = await Promise.all([fetchRun(runId), fetchStructures(runId)]);
        if (!alive) return;
        setRun(summary); setCards(list); setError("");
      } catch (e) { if (alive) setError(e instanceof Error ? e.message : "Search data unavailable."); }
    };
    void tick();
    const timer = setInterval(tick, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [runId]);

  const current = selected || run?.best?.structureHash || cards[cards.length - 1]?.structureHash || null;
  useEffect(() => {
    if (!runId || !current) { setDetail(null); return; }
    let alive = true;
    fetchStructure(runId, current).then(d => { if (alive) setDetail(d); }).catch(() => { if (alive) setDetail(null); });
    return () => { alive = false; };
  }, [runId, current, cards.length, run?.counts?.eval]);

  useEffect(() => {
    if (!runId || !current) { setEvents([]); return; }
    let alive = true;
    const load = () => fetchAgentEvents(runId, current, node).then(e => { if (alive) setEvents(e); }).catch(() => undefined);
    void load();
    const timer = setInterval(load, 4000);
    return () => { alive = false; clearInterval(timer); };
  }, [runId, current, node]);

  const curve = useMemo(() => cards.filter(c => c.fitness != null).map(c => c.fitness as number), [cards]);
  const byGen = useMemo(() => {
    const groups: Record<number, StructureCard[]> = {};
    cards.forEach(c => { (groups[c.generation ?? 0] ||= []).push(c); });
    return Object.entries(groups).sort((a, b) => Number(a[0]) - Number(b[0]));
  }, [cards]);
  const bestHash = run?.best?.structureHash;

  if (!runId) return <div className="search-view"><div className="search-empty big">Start a reconstruction to watch the agent teams search.</div></div>;

  return <div className="search-view">
    <div className="search-top">
      <div className="search-counters">
        <div><b>{pct(run?.best?.fitness)}</b><small>best match</small></div>
        <div><b>{run?.counts?.eval ?? 0}</b><small>teams scored</small></div>
        <div><b>{byGen.length}</b><small>generations</small></div>
        <div><b>{(run?.counts?.blocked ?? 0) + (run?.counts?.near_dup ?? 0)}</b><small>repeats skipped</small></div>
        <div><b>{run?.counts?.cache_hit ?? 0}</b><small>agents reused from memory</small></div>
        <div><b>{run?.counts?.agent_calls ?? 0}</b><small>agent tool calls</small></div>
      </div>
      <div className="search-status"><span className={`status-dot${run?.running ? " live" : ""}`} />{run?.running ? "Searching" : (run?.status || "Starting").replace("_", " ")}{run?.running && <button className="text-button" onClick={() => void stopRun(runId)}>Stop search</button>}</div>
    </div>
    {run?.running && run.current && <div className="search-now">Now: generation {run.current.generation ?? "?"} · team {run.current.structureHash?.slice(0, 8)} · <b>{run.current.role}</b> {run.current.tool || (run.current.event || "").replace("_", " ")} · {ago(run.current.ts)}</div>}
    {!run?.running && run?.statusReason && <div className="inline-error">{run.statusReason}</div>}
    {!run?.running && run?.status === "interrupted" && <div className="search-now">This search was interrupted (the server restarted). Start a new reconstruction to continue.</div>}
    {error && <div className="inline-error">{error}</div>}
    <LiveRenders runId={runId} title="Live renders · every team, newest first" />
    <Curve values={curve} />

    <div className="generation-board">
      {byGen.map(([gen, list]) => <section key={gen} className="generation-column">
        <header><b>Generation {gen}</b><small>{list.length} teams · best {pct(Math.max(...list.map(c => c.fitness ?? 0)))}</small></header>
        {list.map(c => <button key={c.structureHash} className={`team-card${c.structureHash === current ? " selected" : ""}${c.structureHash === bestHash ? " best" : ""}`} onClick={() => { setSelected(c.structureHash); setNode(undefined); }}>
          <span className="team-thumb">{c.render ? <img src={c.render} alt="" /> : <i>not scored yet</i>}</span>
          <span className="team-roles">{c.nodes.map(n => n.role).join(" → ")}</span>
          <span className="team-meta"><em>{c.origin}</em>{c.structureHash === bestHash && <em className="best-tag">best</em>}{!!c.disqualified?.length && <em className="dq-tag">pasted photo</em>}{c.solidity && <em>solid {pct(c.solidity.solidity)}</em>}</span>
          <span className="team-fit"><i style={{ width: `${Math.round((c.fitness ?? 0) * 100)}%` }} /></span>
          <b>{pct(c.fitness)}</b>
        </button>)}
      </section>)}
      {!cards.length && <div className="search-empty big">The main model is proposing the first agent teams…</div>}
    </div>

    {detail && <div className="team-detail">
      <div className="team-detail-head"><b>Team {detail.structureHash.slice(0, 8)}</b><small>{detail.origin} · generation {detail.generation} · fitness {pct(detail.fitness)}{detail.frontMatch != null ? ` = front match ${pct(detail.frontMatch)}` : ""}{detail.solidity ? ` × solidity (${pct(detail.solidity.solidity)}: closed ${pct(detail.solidity.closure)}, depth ${pct(detail.solidity.thickness)})` : ""}{detail.costUsd ? ` · $${detail.costUsd.toFixed(2)}` : ""}</small></div>
      {!!detail.disqualified?.length && <div className="inline-error">Disqualified: the scene loads the reference photo ({detail.disqualified.join(", ")}) instead of modelling it.</div>}
      {detail.turntable && <a className="turntable" href={detail.turntable} target="_blank" rel="noreferrer"><img src={detail.turntable} alt="The model from 8 sides; back faces red" /><span>All the way round, starting at the stage camera · red = the inside of an open surface</span></a>}
      <div className="team-detail-grid">
        <div className="team-detail-col">
          <TeamGraph card={detail} selected={node} onSelect={id => setNode(id === node ? undefined : id)} />
          {!!detail.modules?.length && <div className="agent-status"><span>Agents</span>{detail.modules.map(m => <button key={m.node_id} className={`agent-row${m.node_id === node ? " selected" : ""}`} onClick={() => setNode(m.node_id === node ? undefined : m.node_id)}>
            <b>{m.role}</b><small>{m.calls} tool calls · {m.output?.summary || m.note || "done"}</small>
          </button>)}</div>}
          {detail.scores && <div className="step-scores">{STEPS.filter(s => detail.scores?.[s] != null).map(s => <div key={s} className="step-score"><span>{s}</span><i><em style={{ width: `${Math.round((detail.scores?.[s] ?? 0) * 100)}%` }} /></i><b>{(detail.scores?.[s] ?? 0).toFixed(2)}</b></div>)}</div>}
        </div>
        <div className="team-detail-col">
          {detail.render && <a className="overview-link" href={detail.overview || detail.render} target="_blank" rel="noreferrer"><img src={detail.render} alt="Stage render of this team's model" /><span>Stage render · open every eval step</span></a>}
          {!!detail.critiqueFixes?.length && <div className="critic-fixes"><span>Critic&apos;s top fixes</span><ol>{detail.critiqueFixes.map((f, i) => <li key={i}>{f}</li>)}</ol></div>}
          {!!detail.similar?.length && <div className="similar-outcomes"><span>Teams with similar results</span>{detail.similar.map(s => <button key={s.structureHash} onClick={() => setSelected(s.structureHash)}>{s.roles.join(" → ")} · {pct(s.fitness)} · similarity {s.similarity.toFixed(2)}</button>)}</div>}
        </div>
      </div>
      <LiveRenders runId={runId} structureHash={detail.structureHash} title={`Team ${detail.structureHash.slice(0, 8)} · renders`} />
      <AgentTranscript events={events} title={node ? `What agent ${node} (${detail.modules?.find(m => m.node_id === node)?.role || detail.nodes.find(n => n.nodeId === node)?.role || "agent"}) is saying` : "What the agents are saying · pick an agent in the graph to follow one"} />
    </div>}
  </div>;
}
