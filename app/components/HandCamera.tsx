"use client";

import { useEffect, useRef, useState } from "react";
import type { HandSample } from "./ModelViewport";

type Props = { onSample: (sample: HandSample | null) => void; enabled: boolean; compact?: boolean; captureMode?: boolean; overlay?: boolean; onCapture?: (file: File) => void };
const WASM_ROOT = "/mediapipe/wasm";
const MODEL_PATH = "/assets/hand_landmarker.task";

type FilterState = { raw: number; value: number; derivative: number; time: number };
function oneEuro(state: FilterState | null, value: number, time: number) {
  if (!state) return { state: { raw: value, value, derivative: 0, time }, value };
  const dt = Math.max(1 / 120, Math.min(0.1, (time - state.time) / 1000));
  const alpha = (cutoff: number) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };
  const rawDerivative = (value - state.raw) / dt;
  const derivative = state.derivative + alpha(1.1) * (rawDerivative - state.derivative);
  const adaptiveCutoff = 1.35 + 0.045 * Math.abs(derivative);
  const filtered = state.value + alpha(adaptiveCutoff) * (value - state.value);
  return { state: { raw: value, value: filtered, derivative, time }, value: filtered };
}

export default function HandCamera({ onSample, enabled, compact = false, captureMode = false, overlay = false, onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<{ detectForVideo: (video: HTMLVideoElement, timestamp: number) => { landmarks?: Array<Array<{ x: number; y: number; z: number }>> }; close: () => void } | null>(null);
  const rafRef = useRef(0);
  const attemptRef = useRef(0);
  const sampleRef = useRef<HandSample | null>(null);
  const calibrateWidthRef = useRef<number | null>(null);
  const palmWidthRef = useRef<number | null>(null);
  const filterRef = useRef<{ x: FilterState | null; y: FilterState | null }>({ x: null, y: null });
  const distanceFilterRef = useRef<FilterState | null>(null);
  const pinchRef = useRef(false);
  const lastUIRef = useRef(0);
  const pointUpdateRef = useRef(0);
  const [status, setStatus] = useState<"off" | "loading" | "ready" | "error">("off");
  const [tracking, setTracking] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [message, setMessage] = useState("");
  const [handFound, setHandFound] = useState(false);
  const [handPoint, setHandPoint] = useState<{ x: number; y: number } | null>(null);
  const [pinching, setPinching] = useState(false);
  const [distance, setDistance] = useState(500);
  const [calibrated, setCalibrated] = useState(false);

  function stop() {
    attemptRef.current += 1;
    cancelAnimationFrame(rafRef.current);
    landmarkerRef.current?.close(); landmarkerRef.current = null;
    streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    sampleRef.current = null; onSample(null);
    setStatus("off"); setTracking("idle"); setHandFound(false); setPinching(false);
  }

  useEffect(() => () => {
    attemptRef.current += 1;
    cancelAnimationFrame(rafRef.current);
    landmarkerRef.current?.close();
    streamRef.current?.getTracks().forEach(track => track.stop());
  }, []);

  async function startTracker(attempt: number, video: HTMLVideoElement) {
    setTracking("loading"); setMessage("Loading hand tracking model…");
    try {
      const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
      const common = { runningMode: "VIDEO" as const, numHands: 1, minHandDetectionConfidence: 0.55, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 };
      let landmarker;
      try {
        landmarker = await HandLandmarker.createFromOptions(vision, { ...common, baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" } });
      } catch {
        landmarker = await HandLandmarker.createFromOptions(vision, { ...common, baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" } });
      }
      if (attempt !== attemptRef.current) { landmarker.close(); return; }
      landmarkerRef.current?.close(); landmarkerRef.current = landmarker;
      setTracking("ready"); setMessage("");
      let previousVideoTime = -1;
      const loop = () => {
        rafRef.current = requestAnimationFrame(loop);
        const tracker = landmarkerRef.current;
        if (!tracker || video.readyState < 2 || video.currentTime === previousVideoTime) return;
        previousVideoTime = video.currentTime;
        const result = tracker.detectForVideo(video, performance.now());
        const hand = result.landmarks?.[0];
        if (!hand || !video.videoWidth) {
          sampleRef.current = null; onSample(null);
          filterRef.current = { x: null, y: null }; distanceFilterRef.current = null; pinchRef.current = false;
          setHandPoint(null);
          if (performance.now() - lastUIRef.current > 180) { lastUIRef.current = performance.now(); setHandFound(false); setPinching(false); }
          return;
        }
        const indexMcp = hand[5], thumbTip = hand[4], indexTip = hand[8], pinkyMcp = hand[17];
        const palmWidthPx = Math.hypot((indexMcp.x - pinkyMcp.x) * video.videoWidth, (indexMcp.y - pinkyMcp.y) * video.videoHeight);
        palmWidthRef.current = palmWidthPx;
        const referenceWidth = calibrateWidthRef.current || video.videoWidth * 0.18;
        const rawDistance = Math.max(180, Math.min(1300, 500 * referenceWidth / Math.max(palmWidthPx, 1)));
        const filteredDistance = oneEuro(distanceFilterRef.current, rawDistance, performance.now());
        distanceFilterRef.current = filteredDistance.state;
        const estimatedMm = Math.round(filteredDistance.value);
        const pinchDistance = Math.hypot((thumbTip.x - indexTip.x) * video.videoWidth, (thumbTip.y - indexTip.y) * video.videoHeight) / Math.max(video.videoWidth, video.videoHeight);
        if (!pinchRef.current && pinchDistance < 0.047) pinchRef.current = true;
        else if (pinchRef.current && pinchDistance > 0.068) pinchRef.current = false;
        const targetX = pinchRef.current ? (thumbTip.x + indexTip.x) / 2 : indexTip.x;
        const filteredX = oneEuro(filterRef.current.x, 1 - targetX, performance.now());
        const filteredY = oneEuro(filterRef.current.y, pinchRef.current ? (thumbTip.y + indexTip.y) / 2 : indexTip.y, performance.now());
        filterRef.current = { x: filteredX.state, y: filteredY.state };
        const next = { x: filteredX.value, y: filteredY.value, pinching: pinchRef.current, distanceMm: estimatedMm };
        sampleRef.current = next; onSample(next);
        if (performance.now() - pointUpdateRef.current > 28) { pointUpdateRef.current = performance.now(); setHandPoint({ x: next.x, y: next.y }); }
        if (performance.now() - lastUIRef.current > 180) { lastUIRef.current = performance.now(); setHandFound(true); setPinching(next.pinching); setDistance(estimatedMm); }
      };
      loop();
    } catch (error) {
      if (attempt !== attemptRef.current) return;
      landmarkerRef.current?.close(); landmarkerRef.current = null;
      setTracking("error");
      setMessage(error instanceof Error ? error.message : error && typeof error === "object" && "message" in error ? String(error.message) : "Hand tracking assets could not load. Check your connection and retry.");
    }
  }

  async function start() {
    if (status === "loading" || tracking === "loading") return;
    if (status === "ready") { void startTracker(++attemptRef.current, videoRef.current!); return; }
    const attempt = ++attemptRef.current;
    setStatus("loading"); setTracking("idle"); setMessage("Allow camera access to capture views and track your hand.");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: overlay ? 360 : 480 }, ...(overlay ? { aspectRatio: { ideal: 16 / 9 } } : {}) }, audio: false });
      if (attempt !== attemptRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Camera preview is unavailable.");
      video.srcObject = stream;
      await video.play();
      if (attempt !== attemptRef.current) return;
      setStatus("ready");
      void startTracker(attempt, video);
    } catch (error) {
      if (attempt !== attemptRef.current) return;
      streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = null;
      setStatus("error"); setMessage(error instanceof Error ? error.message : "Camera access could not start. Check browser permissions and retry.");
    }
  }

  useEffect(() => {
    if (enabled && status === "off") void start();
    if (!enabled && status !== "off") {
      const timer = window.setTimeout(() => stop(), 0);
      return () => window.clearTimeout(timer);
    }
  }, [enabled, status]);

  function calibrate() {
    if (!sampleRef.current || !videoRef.current) { setMessage("Show your palm to the camera first."); return; }
    if (!palmWidthRef.current) { setMessage("Wait until your palm is clearly tracked."); return; }
    setMessage("Calibration uses your current tracked palm width as the 50 cm reference.");
    setCalibrated(true); calibrateWidthRef.current = palmWidthRef.current; setDistance(500);
  }

  function captureView() {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !onCapture) return;
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const context = canvas.getContext("2d"); if (!context) return;
    context.translate(canvas.width, 0); context.scale(-1, 1); context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => { if (blob) onCapture(new File([blob], `camera-view-${Date.now()}.jpg`, { type: "image/jpeg" })); }, "image/jpeg", .92);
  }

  if (captureMode) return <div className="capture-camera capture-camera-tracked">
    <div className="capture-video-wrap"><video ref={videoRef} autoPlay playsInline muted className="capture-video" />
      {status !== "ready" && <div className="capture-camera-overlay"><b>{status === "loading" ? "Starting camera…" : "Camera permission needed"}</b>{status === "error" && <button className="text-button" onClick={start}>Retry camera</button>}{message && <small>{message}</small>}</div>}
      {status === "ready" && <span className={`camera-live${handFound ? " hand-found" : ""}`}><i />{tracking === "loading" ? "Loading hand tracking…" : tracking === "error" ? "Camera ready · tracking unavailable" : pinching ? "Pinch detected" : handFound ? `Hand tracked · ${distance} mm` : "Hand tracking active · show your hand"}</span>}
      {status === "ready" && tracking === "error" && <button className="capture-tracking-retry" onClick={() => void start()}>Retry hand tracking</button>}
      {handFound && handPoint && <span className={`capture-hand-point${pinching ? " pinching" : ""}`} style={{ left: `${handPoint.x * 100}%`, top: `${handPoint.y * 100}%` }} />}
    </div><div className="capture-camera-controls"><div><b>Capture camera views</b><small>{tracking === "error" ? "Camera capture is available; retry hand tracking above." : tracking === "loading" ? "Hand tracking is starting. Add several angles, then build the base model." : "Add several angles, then build the base model."}</small></div><button className="primary-button" onClick={captureView} disabled={status !== "ready"}>Capture view <span>＋</span></button></div>
  </div>;

  if (overlay) return <div className="hand-camera-overlay">
    <video ref={videoRef} autoPlay playsInline muted className="mirrored" />
    {handFound && handPoint && <span className={`hand-cursor-dot${pinching ? " pinching" : ""}`} style={{ left: `${handPoint.x * 100}%`, top: `${handPoint.y * 100}%` }} />}
    <span className={`overlay-camera-status${handFound ? " found" : ""}`}><i />{pinching ? "Pinch active" : handFound ? `Hand detected · ${distance} mm` : tracking === "loading" ? "Loading hand tracking…" : tracking === "error" ? "Tracking unavailable" : status === "error" ? "Camera access needed" : "Show your hand"}</span>
    {status !== "ready" && <div className="overlay-camera-message">{status === "loading" ? "Starting camera…" : status === "error" ? <button onClick={start}>Allow or retry camera</button> : null}</div>}
    {status === "ready" && tracking === "error" && <button className="overlay-camera-retry" onClick={() => void start()}>Retry tracking</button>}
  </div>;

  return <div className={`hand-panel${compact ? " compact" : ""}`}>
    <div className="hand-video"><video ref={videoRef} playsInline muted className={status === "ready" ? "mirrored visible" : "mirrored"} />
      {handFound && handPoint && <span className={`hand-cursor-dot${pinching ? " pinching" : ""}`} style={{ left: `${handPoint.x * 100}%`, top: `${handPoint.y * 100}%` }} />}
      {status !== "ready" && <div className="camera-empty"><span>{status === "error" ? "CAMERA ACCESS NEEDED" : "STARTING CAMERA"}</span></div>}
      <div className={`hand-badge ${handFound ? "found" : ""}`}><i />{pinching ? "Pinch active" : handFound ? "Hand detected" : tracking === "loading" ? "Loading hand tracking" : tracking === "error" ? "Tracking unavailable" : status === "ready" ? "Show your hand" : status === "error" ? "Allow camera access" : "Starting camera"}</div>
    </div>
    {!compact && <>
      <div className="hand-controls"><div><b>Hand selection</b><small>{status === "ready" && tracking === "ready" ? "Pinch and move to sculpt · release to refine" : status === "ready" && tracking === "error" ? "Camera is live; retry hand tracking" : status === "error" ? "Allow camera access, then retry" : "Starting camera and hand tracking…"}</small></div>
        {status === "ready" && tracking === "ready" ? <span className="cam-status">Hand tracking active</span> : <button className="cam-button" onClick={start} disabled={!enabled || status === "loading" || tracking === "loading"}>{status === "loading" || tracking === "loading" ? "Starting…" : status === "ready" ? "Retry tracking" : "Retry camera"}</button>}
      </div>
      <div className="depth-control"><span><b>Hand distance</b><small>{calibrated ? "Approximate · calibrated at 50 cm" : "Approximate · calibrate at 50 cm"}</small></span><strong>{handFound ? `${distance} mm` : "—"}</strong></div>
      <button className="calibrate-button" onClick={calibrate} disabled={tracking !== "ready" || !handFound}>{calibrated ? "Calibrated at 50 cm" : "Calibrate at 50 cm"}</button>
      {message && (status === "error" || tracking === "error") && <p className="camera-message error">{message}</p>}
    </>}
    {compact && status !== "off" && <div className="hand-compact-status"><span className={`connection-dot${handFound ? " is-found" : ""}`} />{status === "error" ? "Allow camera access" : tracking === "error" ? "Tracking unavailable" : tracking === "loading" ? "Loading hand tracking" : pinching ? "Pinch detected" : handFound ? `Hand tracked · ${distance} mm` : status === "ready" ? "Hand tracking active · show your hand" : "Starting hand tracking"}{(status === "error" || tracking === "error") && <button onClick={start}>Retry</button>}</div>}
  </div>;
}
