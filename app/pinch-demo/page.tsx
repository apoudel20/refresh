"use client";

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
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
    <header className="pinch-demo-header"><Link href="/" className="demo-back">← Refresh</Link><div><span className="kicker">INTERACTION DEMO</span><h1>Pigeon pinch sculpt</h1></div><span className={`demo-presence${sample ? " active" : ""}`}><i />{sample?.pinching ? "Pinch active" : sample ? "Hand detected" : "Show hand to camera"}</span></header>
    <div className="pinch-demo-layout"><section className="pinch-demo-model"><div className="demo-section-heading"><div><b>Pigeon model · live camera underneath</b><span>Move your hand in the camera view; pinch to sculpt the model.</span></div><span className="model-ready-dot">{ready ? "Ready" : "Loading"}</span></div><div className="demo-model-stack"><HandCamera enabled overlay onSample={onSample}/><ModelViewport modelUrl="/assets/demo-pigeon.glb" demoBird handSample={sampleRef} onSelection={onSelection} onModelReady={setReady} autoRotate={false}/></div><div className="demo-model-foot"><span>Move hand to position the blue cursor · pinch to deform</span>{lastSculpt && <span>{lastSculpt}</span>}</div></section></div>
  </main>;
}
