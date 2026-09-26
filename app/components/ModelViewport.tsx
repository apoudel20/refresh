"use client";

import { MutableRefObject, useEffect, useRef, useState } from "react";
import type * as THREEType from "three";

export type HandSample = { x: number; y: number; pinching: boolean; distanceMm: number };
export type SelectionPacket = { points: Array<[number, number, number]>; faces: Array<{ meshId: string; faceIndex: number }>; geometryEdits?: Array<{ meshId: string; meshName: string; vertices: Array<{ index: number; original: [number, number, number]; position: [number, number, number] }> }>; handDistanceMm: number; screenshot?: string };

type Props = {
  modelUrl?: string;
  handSample: MutableRefObject<HandSample | null>;
  onSelection: (selection: SelectionPacket) => void;
  onModelReady: (ready: boolean) => void;
  onModelEdited?: (blob: Blob, projectKey: string) => void;
  projectKey?: string;
  autoRotate: boolean;
  demoBird?: boolean;
};
type OrbitController = { update: () => void; dispose: () => void; enabled: boolean; enableDamping: boolean; dampingFactor: number; minDistance: number; maxDistance: number; minPolarAngle: number; maxPolarAngle: number; autoRotate: boolean; autoRotateSpeed: number; target: THREEType.Vector3 };
type Grab = {
  mesh: THREEType.Mesh;
  anchor: THREEType.Vector3;
  startWorld: THREEType.Vector3;
  plane: THREEType.Plane;
  startHandDistance: number;
  towardCamera: THREEType.Vector3;
  radius: number;
  originals: Map<number, THREEType.Vector3>;
  weights: Map<number, number>;
};

export default function ModelViewport({ modelUrl, handSample, onSelection, onModelReady, onModelEdited, projectKey = "demo", autoRotate }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const callbacksRef = useRef({ handSample, onModelReady, onSelection, onModelEdited });
  const controlsRef = useRef<OrbitController | null>(null);
  useEffect(() => {
    callbacksRef.current = { handSample, onModelReady, onSelection, onModelEdited };
    if (controlsRef.current) controlsRef.current.autoRotate = autoRotate;
  }, [handSample, onModelReady, onSelection, onModelEdited, autoRotate]);

  useEffect(() => {
    if (!hostRef.current || !canvasRef.current || !modelUrl) { setReady(false); callbacksRef.current.onModelReady(false); return; }
    let alive = true;
    let frame = 0;
    let renderer: THREEType.WebGLRenderer | null = null;
    let controls: OrbitController | null = null;
    let observer: ResizeObserver | null = null;
    let model: THREEType.Object3D | null = null;
    let line: THREEType.Line | null = null;
    let brush: THREEType.Mesh | null = null;
    let grab: Grab | null = null;
    const selectedPoints: THREEType.Vector3[] = [];
    const selectedFaces: Array<{ meshId: string; faceIndex: number }> = [];
    const editedVertices = new Map<string, { meshName: string; vertices: Map<number, { original: [number, number, number]; position: [number, number, number] }> }>();
    let wasPinching = false;
    let lastSample = 0;

    const finishSelection = async (THREE: typeof import("three")) => {
      if (model && editedVertices.size) {
        const points = selectedPoints.map(point => [point.x, point.y, point.z] as [number, number, number]);
        const screenshot = renderer?.domElement.toDataURL("image/png");
        const geometryEdits = [...editedVertices.entries()].map(([meshId, edit]) => ({ meshId, meshName: edit.meshName, vertices: [...edit.vertices.entries()].map(([index, vertex]) => ({ index, ...vertex })) }));
        callbacksRef.current.onSelection({ points, faces: [...selectedFaces], geometryEdits, handDistanceMm: callbacksRef.current.handSample.current?.distanceMm || 500, screenshot });
        try {
          const { GLTFExporter } = await import("three/addons/exporters/GLTFExporter.js");
          const exporter = new GLTFExporter();
          exporter.parse(model, result => {
            if (alive && result instanceof ArrayBuffer) callbacksRef.current.onModelEdited?.(new Blob([result], { type: "model/gltf-binary" }), projectKey);
          }, exportError => { if (alive) setError(exportError.message || "Could not save the sculpted model."); }, { binary: true });
        } catch (exportError) {
          if (alive) setError(exportError instanceof Error ? exportError.message : "Could not save the sculpted model.");
        }
      }
      selectedPoints.length = 0; selectedFaces.length = 0; editedVertices.clear(); grab = null;
      if (line) { line.parent?.remove(line); line.geometry.dispose(); (line.material as THREEType.Material).dispose(); line = null; }
      if (brush) brush.visible = false;
      void THREE;
    };

    void (async () => {
      try {
        const THREE = await import("three");
        const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
        const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
        if (!alive || !canvasRef.current || !hostRef.current) return;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color("#f4f8fc");
        scene.add(new THREE.HemisphereLight(0xeaf5ff, 0x8192a5, 2.2));
        const key = new THREE.DirectionalLight(0xffffff, 2.1); key.position.set(3, 5, 5); scene.add(key);
        const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
        renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: false, preserveDrawingBuffer: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.08;
        controls = new OrbitControls(camera, renderer.domElement) as OrbitController;
        controlsRef.current = controls;
        controls.enableDamping = true; controls.dampingFactor = 0.08;
        controls.autoRotate = autoRotate; controls.autoRotateSpeed = 0.7;
        const floor = new THREE.GridHelper(8, 20, 0xc7d8e8, 0xe0e9f1);
        floor.position.y = -1.35; scene.add(floor);
        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const frameModel = (object: THREEType.Object3D) => {
          object.updateWorldMatrix(true, true);
          const bounds = new THREE.Box3().setFromObject(object);
          const center = bounds.getCenter(new THREE.Vector3());
          object.position.sub(center);
          const dimensions = bounds.getSize(new THREE.Vector3());
          const largest = Math.max(dimensions.x, dimensions.y, dimensions.z) || 1;
          object.scale.multiplyScalar(2.15 / largest);
          object.updateWorldMatrix(true, true);
          const sphere = new THREE.Box3().setFromObject(object).getBoundingSphere(new THREE.Sphere());
          const verticalFov = THREE.MathUtils.degToRad(camera.fov);
          const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(camera.aspect, 0.1));
          const narrowFov = Math.min(verticalFov, horizontalFov);
          const distance = sphere.radius / Math.sin(narrowFov / 2) * 1.18;
          camera.position.set(distance * 0.06, distance * 0.08, distance);
          camera.near = Math.max(0.01, distance / 100); camera.far = distance * 20;
          camera.updateProjectionMatrix(); camera.lookAt(0, 0, 0);
          controls!.target.set(0, 0, 0);
          controls!.minDistance = sphere.radius * 1.1; controls!.maxDistance = sphere.radius * 9;
          controls!.minPolarAngle = 0.06; controls!.maxPolarAngle = Math.PI - 0.04;
          controls!.update();
        };
        const onLoaded = (object: THREEType.Object3D) => {
          if (!alive) return;
          model = object;
          frameModel(model);
          model.traverse(child => { const mesh = child as THREEType.Mesh; if (mesh.isMesh) { mesh.castShadow = true; mesh.receiveShadow = true; } });
          scene.add(model);
          brush = new THREE.Mesh(new THREE.SphereGeometry(.065, 18, 12), new THREE.MeshBasicMaterial({ color: 0x1674df, transparent: true, opacity: .62, depthTest: false }));
          brush.renderOrder = 30; brush.visible = false; scene.add(brush);
          setError(""); setReady(true); callbacksRef.current.onModelReady(true);
        };
        const loader = new GLTFLoader();
        loader.load(modelUrl, gltf => onLoaded(gltf.scene), undefined, loadError => { if (alive) { setError(loadError instanceof Error ? loadError.message : "Could not load the generated model."); setReady(false); callbacksRef.current.onModelReady(false); } });

        const resize = () => {
          if (!hostRef.current || !renderer) return;
          const rect = hostRef.current.getBoundingClientRect();
          renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
          camera.aspect = rect.width / Math.max(1, rect.height); camera.updateProjectionMatrix();
        };
        observer = new ResizeObserver(resize); observer.observe(hostRef.current); resize();
        const beginGrab = (hit: THREEType.Intersection, handDistanceMm: number) => {
          const mesh = hit.object as THREEType.Mesh;
          if (!mesh.isMesh || !hit.face) return null;
          mesh.updateWorldMatrix(true, false);
          const position = mesh.geometry.getAttribute("position") as THREEType.BufferAttribute;
          const localAnchor = mesh.worldToLocal(hit.point.clone());
          const scale = mesh.getWorldScale(new THREE.Vector3());
          const averageScale = Math.max(.2, (scale.x + scale.y + scale.z) / 3);
          const radius = .2 / averageScale;
          const originals = new Map<number, THREEType.Vector3>();
          const weights = new Map<number, number>();
          for (let index = 0; index < position.count; index++) {
            const local = new THREE.Vector3().fromBufferAttribute(position, index);
            const distance = local.distanceTo(localAnchor);
            if (distance >= radius) continue;
            originals.set(index, local);
            const t = 1 - distance / radius;
            weights.set(index, t * t * (3 - 2 * t));
          }
          const normal = camera.getWorldDirection(new THREE.Vector3());
          const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, hit.point);
          return { mesh, anchor: localAnchor, startWorld: hit.point.clone(), plane, startHandDistance: handDistanceMm, towardCamera: normal.negate(), radius, originals, weights } satisfies Grab;
        };
        const tick = () => {
          frame = requestAnimationFrame(tick);
          const sample = callbacksRef.current.handSample.current;
          if (controls) controls.enabled = !sample?.pinching;
          controls?.update();
          if (model && sample && performance.now() - lastSample > 16) {
            lastSample = performance.now();
            pointer.set(sample.x * 2 - 1, 1 - sample.y * 2);
            raycaster.setFromCamera(pointer, camera);
            if (sample.pinching && !wasPinching) {
              selectedPoints.length = 0; selectedFaces.length = 0; editedVertices.clear();
              const hits = raycaster.intersectObject(model, true);
              if (hits.length) {
                grab = beginGrab(hits[0], sample.distanceMm);
                if (grab && hits[0].faceIndex != null) selectedFaces.push({ meshId: grab.mesh.uuid, faceIndex: hits[0].faceIndex });
              }
            }
            if (sample.pinching && grab) {
              const currentWorld = raycaster.ray.intersectPlane(grab.plane, new THREE.Vector3());
              if (currentWorld) {
                const depthOffset = (grab.startHandDistance - sample.distanceMm) * 0.002;
                const deltaWorld = currentWorld.clone().sub(grab.startWorld).addScaledVector(grab.towardCamera, depthOffset).clampLength(0, grab.radius * 2.5);
                const localCurrent = grab.mesh.worldToLocal(grab.startWorld.clone().add(deltaWorld));
                const localDelta = localCurrent.sub(grab.anchor);
                const geometry = grab.mesh.geometry;
                const position = geometry.getAttribute("position") as THREEType.BufferAttribute;
                const changes = new Map<number, { original: [number, number, number]; position: [number, number, number] }>();
                for (const [index, original] of grab.originals) {
                  const weight = grab.weights.get(index) || 0;
                  const next = original.clone().addScaledVector(localDelta, weight);
                  position.setXYZ(index, next.x, next.y, next.z);
                  changes.set(index, { original: [original.x, original.y, original.z], position: [next.x, next.y, next.z] });
                }
                position.needsUpdate = true; geometry.computeVertexNormals(); geometry.computeBoundingSphere();
                const meshEdit = { meshName: grab.mesh.name || "mesh", vertices: changes };
                editedVertices.set(grab.mesh.uuid, meshEdit);
                const worldCursor = grab.startWorld.clone().add(deltaWorld);
                if (brush) { brush.position.copy(worldCursor); brush.visible = true; }
                if (!selectedPoints.length || selectedPoints[selectedPoints.length - 1].distanceToSquared(worldCursor) > 0.0008) selectedPoints.push(worldCursor);
                if (line) { scene.remove(line); line.geometry.dispose(); (line.material as THREEType.Material).dispose(); }
                const path = [...selectedPoints];
                if (path.length > 2) path.push(path[0]);
                line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path), new THREE.LineBasicMaterial({ color: 0x3178ce, linewidth: 2, depthTest: false }));
                line.renderOrder = 20; scene.add(line);
              }
            }
            if ((!sample.pinching || !grab) && brush) brush.visible = false;
            if (!sample.pinching && wasPinching) void finishSelection(THREE);
            wasPinching = sample.pinching;
          } else if (!sample && wasPinching) { void finishSelection(THREE); wasPinching = false; if (brush) brush.visible = false; }
          renderer!.render(scene, camera);
        };
        tick();
      } catch (loadError) {
        if (alive) setError(loadError instanceof Error ? loadError.message : "3D viewer could not start.");
      }
    })();

    return () => { alive = false; observer?.disconnect(); cancelAnimationFrame(frame); controls?.dispose(); controlsRef.current = null; renderer?.dispose(); };
  }, [modelUrl, autoRotate, projectKey]);

  return <div className="model-viewport" ref={hostRef}>
    <canvas ref={canvasRef} aria-label="Generated 3D model viewport" />
    {!modelUrl && <div className="model-empty"><span className="empty-ring"/><b>Waiting for generated model</b><small>The model from this project’s reconstruction run will appear here.</small></div>}
    {error && <div className="model-error">{error}</div>}
    {ready && <div className="model-help">Drag to orbit <span>·</span> Scroll to zoom <span>·</span> Pinch and move to sculpt</div>}
  </div>;
}
