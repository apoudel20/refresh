"use client";

import { useCallback, useRef, useState } from "react";
import HandCamera from "../components/HandCamera";
import ModelViewport, { type HandSample, type SelectionPacket } from "../components/ModelViewport";

export default function PinchDemo() {
  const [sample, setSample] = useState<HandSample | null>(null);
  const [ready, setReady] = useState(false);
  const [lastSculpt, setLastSculpt] = useState("");
  const sampleRef = useRef<HandSample | null>(null);
  const onSample = useCallback((next: HandSample | null) => { sampleRef.current = next; setSample(next); }, []);
  const onSelection = useCallback((selection: SelectionPacket) => { setLastSculpt(`Sculpt stroke captured · ${selection.points.length} points`); }, []);

  return <main className="pinch-demo-page">
    <header className="pinch-demo-header"><a href="/" className="demo-back">← Refresh</a><div><span className="kicker">INTERACTION DEMO</span><h1>Pigeon pinch sculpt</h1></div><span className={`demo-presence${sample ? " active" : ""}`}><i />{sample?.pinching ? "Pinch active" : sample ? "Hand detected" : "Show hand to camera"}</span></header>
    <div className="pinch-demo-layout"><section className="pinch-demo-model"><div className="demo-section-heading"><div><b>Pigeon model</b><span>Pinch and move over the surface to sculpt in real time.</span></div><span className="model-ready-dot">{ready ? "Ready" : "Loading"}</span></div><ModelViewport modelUrl="/assets/demo-pigeon.glb" demoBird handSample={sampleRef} onSelection={onSelection} onModelReady={setReady} autoRotate={false}/><div className="demo-model-foot"><span>Live geometry deformation</span>{lastSculpt && <span>{lastSculpt}</span>}</div></section>
      <aside className="pinch-demo-camera"><HandCamera enabled onSample={onSample}/><p className="demo-note">The camera feed stays on this device. Hand landmarks control a local sculpt brush; no AI service is required.</p></aside></div>
  </main>;
}
