"use client";

import { useEffect, useState } from "react";
import { fetchRenders, type RenderItem } from "../lib/harness-client";

const LABEL: Record<RenderItem["kind"], string> = {
  preview: "live preview", stage: "scored view", turntable: "all sides", image: "render / image",
};
const pct = (v?: number) => (v == null ? "" : ` · ${Math.round(v * 100)}%`);
const ago = (t: number) => {
  const s = Math.max(0, Math.round(Date.now() / 1000 - t));
  return s < 60 ? `${s}s ago` : `${Math.round(s / 60)}m ago`;
};

export default function LiveRenders({ runId, structureHash, title }: { runId: string; structureHash?: string; title: string }) {
  const [items, setItems] = useState<RenderItem[]>([]);
  const [kind, setKind] = useState<RenderItem["kind"] | "all">("all");
  const [open, setOpen] = useState<RenderItem | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => fetchRenders(runId, structureHash).then(r => { if (alive) setItems(r); }).catch(() => undefined);
    void load();
    const timer = setInterval(load, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [runId, structureHash]);
  const shown = items.filter(i => kind === "all" || i.kind === kind);
  return <div className="live-renders">
    <div className="lr-head">
      <span>{title}</span>
      <div className="lr-filters">{(["all", "preview", "stage", "turntable", "image"] as const).map(k =>
        <button key={k} className={k === kind ? "on" : ""} onClick={() => setKind(k)}>{k === "all" ? `all ${items.length}` : LABEL[k]}</button>)}</div>
    </div>
    <div className="lr-strip">
      {shown.map(i => <button key={i.url} className={`lr-item ${i.kind}`} onClick={() => setOpen(i)} title={i.name}>
        <img src={`${i.url}?t=${Math.round(i.t)}`} alt="" loading="lazy" />
        <span><b>{i.role || "agent"}</b> · {LABEL[i.kind]}{pct(i.overall)}</span>
        <small>{i.structureHash?.slice(0, 8)} · {ago(i.t)}</small>
      </button>)}
      {!shown.length && <div className="tx-empty">No renders yet: they appear as soon as an agent changes the scene.</div>}
    </div>
    {open && <div className="lr-lightbox" onClick={() => setOpen(null)}>
      <figure onClick={e => e.stopPropagation()}>
        <img src={`${open.url}?t=${Math.round(open.t)}`} alt="" />
        <figcaption><b>{open.role || "agent"}</b> · {LABEL[open.kind]}{pct(open.overall)}{open.solidity != null ? ` · solidity ${Math.round(open.solidity * 100)}%` : ""} · team {open.structureHash?.slice(0, 8)} · {open.name}
          <a href={open.url} target="_blank" rel="noreferrer">open</a><button onClick={() => setOpen(null)}>close</button></figcaption>
      </figure>
    </div>}
  </div>;
}
