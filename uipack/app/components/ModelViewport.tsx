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
  autoRotate: boolean;
  demoBird?: boolean;
};
type OrbitController = { update: () => void; dispose: () => void; enableDamping: boolean; dampingFactor: number; minDistance: number; maxDistance: number; autoRotate: boolean; autoRotateSpeed: number };

export default function ModelViewport({ modelUrl, handSample, onSelection, onModelReady, autoRotate, demoBird = false }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const callbacksRef = useRef({ handSample, onModelReady, onSelection });
  const controlsRef = useRef<OrbitController | null>(null);
  callbacksRef.current = { handSample, onModelReady, onSelection };
  useEffect(() => { if (controlsRef.current) controlsRef.current.autoRotate = autoRotate; }, [autoRotate]);

  useEffect(() => {
    if (!hostRef.current || !canvasRef.current || (!modelUrl && !demoBird)) { setReady(false); callbacksRef.current.onModelReady(false); return; }
    let alive = true;
    let frame = 0;
    let renderer: THREEType.WebGLRenderer | null = null;
    let controls: OrbitController | null = null;
    let observer: ResizeObserver | null = null;
    let model: THREEType.Object3D | null = null;
    let line: THREEType.Line | null = null;
    let brush: THREEType.Mesh | null = null;
    let selectedPoints: THREEType.Vector3[] = [];
    const selectedFaces: Array<{ meshId: string; faceIndex: number }> = [];
    const editedVertices = new Map<string, { meshName: string; vertices: Map<number, { original: [number, number, number]; position: [number, number, number] }> }>();
    let wasPinching = false;
    let lastSample = 0;

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
        const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
        camera.position.set(0, 0.15, 4.7);
        renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: false, preserveDrawingBuffer: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.08;
        controls = new OrbitControls(camera, renderer.domElement);
        controlsRef.current = controls;
        controls.enableDamping = true; controls.dampingFactor = 0.08;
        controls.minDistance = 1.2; controls.maxDistance = 12;
        controls.autoRotate = autoRotate; controls.autoRotateSpeed = 0.7;
        const floor = new THREE.GridHelper(8, 20, 0xc7d8e8, 0xe0e9f1);
        floor.position.y = -1.15; scene.add(floor);
        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const onLoaded = (object: THREEType.Object3D) => {
          if (!alive) return;
          model = object;
          if (modelUrl !== "demo:bird") {
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            model.position.sub(center);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z) || 1;
            model.scale.setScalar(2.15 / maxDim);
          }
          model.traverse(child => { const mesh = child as THREEType.Mesh; if (mesh.isMesh) { mesh.castShadow = true; mesh.receiveShadow = true; } });
          scene.add(model);
          brush = new THREE.Mesh(new THREE.SphereGeometry(.075, 18, 12), new THREE.MeshBasicMaterial({ color: 0x1674df, transparent: true, opacity: .58, depthTest: false }));
          brush.renderOrder = 30; brush.visible = false; scene.add(brush);
          setError(""); setReady(true); callbacksRef.current.onModelReady(true);
        };
        if (demoBird && modelUrl === "demo:bird") {
          // Local fallback bird for development if the demo GLB path is intentionally omitted.
          const bird = new THREE.Group();
          const blue = new THREE.MeshStandardMaterial({ color: 0x4e8fc7, roughness: .76 });
          const pale = new THREE.MeshStandardMaterial({ color: 0xd6e7f6, roughness: .84 });
          const orange = new THREE.MeshStandardMaterial({ color: 0xe4a35b, roughness: .74 });
          const dark = new THREE.MeshStandardMaterial({ color: 0x263c50, roughness: .55 });
          const bodyGeo = new THREE.SphereGeometry(.77, 48, 32);
          const body = new THREE.Mesh(bodyGeo, blue); body.name = "bird-body"; body.scale.set(1.12, .72, .58); body.position.set(0, -.05, 0); bird.add(body);
          const head = new THREE.Mesh(new THREE.SphereGeometry(.42, 32, 24), blue); head.position.set(.72, .48, 0); bird.add(head);
          const beak = new THREE.Mesh(new THREE.ConeGeometry(.19, .58, 5), orange); beak.rotation.z = -Math.PI / 2; beak.position.set(1.17, .44, 0); bird.add(beak);
          const wing = new THREE.Mesh(new THREE.SphereGeometry(.47, 20, 12), pale); wing.scale.set(1.18, .38, .12); wing.rotation.z = -.22; wing.position.set(-.08, .1, .48); bird.add(wing);
          const tail = new THREE.Mesh(new THREE.ConeGeometry(.28, .68, 5), blue); tail.rotation.z = Math.PI / 2; tail.position.set(-.95, .08, 0); bird.add(tail);
          for (const side of [-1, 1]) { const eye = new THREE.Mesh(new THREE.SphereGeometry(.055, 12, 10), dark); eye.position.set(.83, .57, side * .31); bird.add(eye); }
          for (const side of [-1, 1]) { const leg = new THREE.Mesh(new THREE.CylinderGeometry(.025, .035, .44, 6), orange); leg.position.set(.02, -.62, side * .2); bird.add(leg); }
          onLoaded(bird);
        } else {
          const loader = new GLTFLoader();
          loader.load(modelUrl!, gltf => onLoaded(gltf.scene), undefined, loadError => { if (alive) { setError(loadError instanceof Error ? loadError.message : "Could not load the generated model."); setReady(false); callbacksRef.current.onModelReady(false); } });
        }

        const resize = () => {
          if (!hostRef.current || !renderer) return;
          const rect = hostRef.current.getBoundingClientRect();
          renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
          camera.aspect = rect.width / Math.max(1, rect.height); camera.updateProjectionMatrix();
        };
        observer = new ResizeObserver(resize); observer.observe(hostRef.current); resize();
        let moved = false;
        const finishSelection = () => {
          if (selectedPoints.length > 2) {
            const points = selectedPoints.map(point => [point.x, point.y, point.z] as [number, number, number]);
            const screenshot = renderer?.domElement.toDataURL("image/png");
            const geometryEdits = [...editedVertices.entries()].map(([meshId, edit]) => ({ meshId, meshName: edit.meshName, vertices: [...edit.vertices.entries()].map(([index, vertex]) => ({ index, ...vertex })) }));
            callbacksRef.current.onSelection({ points, faces: [...selectedFaces], geometryEdits, handDistanceMm: callbacksRef.current.handSample.current?.distanceMm || 500, screenshot });
          }
          selectedPoints = []; selectedFaces.length = 0; editedVertices.clear(); moved = false;
          if (line) { scene.remove(line); line.geometry.dispose(); (line.material as THREEType.Material).dispose(); line = null; }
        };
        const tick = () => {
          frame = requestAnimationFrame(tick);
          controls?.update();
          const sample = callbacksRef.current.handSample.current;
          if (model && sample && performance.now() - lastSample > 28) {
            lastSample = performance.now();
            pointer.set(sample.x * 2 - 1, 1 - sample.y * 2);
            raycaster.setFromCamera(pointer, camera);
            const hits = raycaster.intersectObject(model, true);
            let hit: THREEType.Intersection | undefined;
            if (hits.length) {
              const handDepthTarget = THREE.MathUtils.clamp(3.2 + (sample.distanceMm - 500) / 420, 1.4, 5.8);
              hit = hits.reduce((best, current) => Math.abs(current.distance - handDepthTarget) < Math.abs(best.distance - handDepthTarget) ? current : best);
            }
            if (sample.pinching && !wasPinching) { selectedPoints = []; selectedFaces.length = 0; moved = true; }
            if (sample.pinching && hit && moved) {
              const point = hit.point.clone();
              if (brush) { brush.position.copy(point); brush.visible = true; }
              const sculptMesh = hit.object as THREEType.Mesh;
              const sculptGeometry = sculptMesh.geometry;
              const position = sculptGeometry.getAttribute("position") as THREEType.BufferAttribute;
              const localPoint = sculptMesh.worldToLocal(point.clone());
              const worldScale = sculptMesh.getWorldScale(new THREE.Vector3());
              const radius = .2 / Math.max(.2, (worldScale.x + worldScale.y + worldScale.z) / 3);
              let deformed = false;
              for (let vertex = 0; vertex < position.count; vertex++) {
                const vertexPoint = new THREE.Vector3().fromBufferAttribute(position, vertex);
                const distance = vertexPoint.distanceTo(localPoint);
                if (distance < radius) {
                  const weight = Math.pow(1 - distance / radius, 2);
                  const push = .0035 * weight;
                  const normal = sculptGeometry.getAttribute("normal") as THREEType.BufferAttribute;
                  const normalVector = new THREE.Vector3().fromBufferAttribute(normal, vertex).normalize();
                  const nextPosition: [number, number, number] = [vertexPoint.x + normalVector.x * push, vertexPoint.y + normalVector.y * push, vertexPoint.z + normalVector.z * push];
                  position.setXYZ(vertex, nextPosition[0], nextPosition[1], nextPosition[2]);
                  let meshEdit = editedVertices.get(sculptMesh.uuid);
                  if (!meshEdit) { meshEdit = { meshName: sculptMesh.name || "mesh", vertices: new Map() }; editedVertices.set(sculptMesh.uuid, meshEdit); }
                  const existing = meshEdit.vertices.get(vertex);
                  meshEdit.vertices.set(vertex, { original: existing?.original || [vertexPoint.x, vertexPoint.y, vertexPoint.z], position: nextPosition });
                  deformed = true;
                }
              }
              if (deformed) { position.needsUpdate = true; sculptGeometry.computeVertexNormals(); sculptGeometry.computeBoundingSphere(); }

              if (!selectedPoints.length || selectedPoints[selectedPoints.length - 1].distanceToSquared(point) > 0.0015) {
                selectedPoints.push(point);
                const mesh = hit.object as THREEType.Mesh;
                if (hit.faceIndex != null && !selectedFaces.some(face => face.meshId === mesh.uuid && face.faceIndex === hit!.faceIndex)) selectedFaces.push({ meshId: mesh.uuid, faceIndex: hit.faceIndex });
                if (line) { scene.remove(line); line.geometry.dispose(); (line.material as THREEType.Material).dispose(); }
                const path = [...selectedPoints];
                if (path.length > 2) path.push(path[0]);
                line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path), new THREE.LineBasicMaterial({ color: 0x3178ce, linewidth: 2, depthTest: false }));
                line.renderOrder = 20; scene.add(line);
              }
            }
            if ((!sample.pinching || !hit) && brush) brush.visible = false;
            if (!sample.pinching && wasPinching) finishSelection();
            wasPinching = sample.pinching;
          } else if (!sample && wasPinching) { finishSelection(); wasPinching = false; if (brush) brush.visible = false; }
          renderer!.render(scene, camera);
        };
        tick();
      } catch (loadError) {
        if (alive) setError(loadError instanceof Error ? loadError.message : "3D viewer could not start.");
      }
    })();

    return () => { alive = false; observer?.disconnect(); cancelAnimationFrame(frame); controls?.dispose(); controlsRef.current = null; renderer?.dispose(); };
  }, [modelUrl, demoBird]);

  return <div className="model-viewport" ref={hostRef}>
    <canvas ref={canvasRef} aria-label="Generated 3D model viewport" />
    {!modelUrl && <div className="model-empty"><span className="empty-ring"/><b>Waiting for generated model</b><small>The model from this project’s reconstruction run will appear here.</small></div>}
    {error && <div className="model-error">{error}</div>}
    {ready && <div className="model-help">Drag to orbit <span>·</span> Scroll to zoom <span>·</span> Pinch and move to sculpt</div>}
  </div>;
}
