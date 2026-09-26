"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import HandCamera from "./components/HandCamera";
import ModelViewport, { type HandSample, type SelectionPacket } from "./components/ModelViewport";
import SearchPanel from "./components/SearchPanel";
import { fetchRenders, fetchRun, refineSelection, startReconstruction, type RenderItem, type RunSummary, type RunUpdate } from "./lib/harness-client";
import { deleteProject, listProjects, newProject, type ProjectEvent, type ProjectRecord, type ReferenceAsset, saveProject } from "./lib/project-store";

type Screen = "dashboard" | "project";
type Stage = "intake" | "processing" | "studio";

const clockLabel = (time: number) => new Date(time).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const eventId = () => crypto.randomUUID();

function summarizeStatus(value?: string) {
  const status = (value || "").toLowerCase();
  if (status.includes("image") || status.includes("reference")) return "Reading reference views";
  if (status.includes("segment") || status.includes("mask")) return "Segmenting object views";
  if (status.includes("handoff") || status.includes("agent")) return "Agent handoff";
  if (status.includes("similar") || status.includes("metric")) return "Comparing model similarity";
  if (status.includes("refin")) return "Refining geometry";
  if (status.includes("model") || status.includes("geometry") || status.includes("reconstruct") || status.includes("generat")) return "Generating 3D representation";
  return "Updating reconstruction";
}

export default function Home() {
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const projectRef = useRef<ProjectRecord | null>(null);
  const [projectDialog, setProjectDialog] = useState<{ mode: "create" } | { mode: "rename"; target: ProjectRecord } | null>(null);
  const [projectNameInput, setProjectNameInput] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ProjectRecord | null>(null);
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [stage, setStage] = useState<Stage>("intake");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [previews, setPreviews] = useState<Array<{ id: string; name: string; url: string }>>([]);
  const [dashboardImages, setDashboardImages] = useState<File[]>([]);
  const [dashboardPreviews, setDashboardPreviews] = useState<Array<{ file: File; url: string }>>([]);
  const [error, setError] = useState("");
  const [refining, setRefining] = useState(false);
  const [lastSelection, setLastSelection] = useState<SelectionPacket | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [localModelUrl, setLocalModelUrl] = useState("");
  const [handSample, setHandSample] = useState<HandSample | null>(null);
  const handSampleRef = useRef<HandSample | null>(null);
  const handUiUpdateAt = useRef(0);
  const uploadRef = useRef<HTMLInputElement>(null);
  const modelRef = useRef<HTMLInputElement>(null);
  const runAbortRef = useRef<AbortController | null>(null);
  const localModelUrlRef = useRef("");

  useEffect(() => { void listProjects().then(setProjects).catch(() => setError("Local project storage is unavailable in this browser.")); }, []);
  useEffect(() => { projectRef.current = project; }, [project]);
  useEffect(() => {
    const items = dashboardImages.map(file => ({ file, url: URL.createObjectURL(file) }));
    setDashboardPreviews(items);
    return () => items.forEach(item => URL.revokeObjectURL(item.url));
  }, [dashboardImages]);
  useEffect(() => {
    if (!project) { setPreviews([]); return; }
    const next = project.references.map(asset => ({ id: asset.id, name: asset.name, url: URL.createObjectURL(asset.blob) }));
    setPreviews(next);
    return () => next.forEach(preview => URL.revokeObjectURL(preview.url));
  }, [project]);
  useEffect(() => {
    if (!project?.modelBlob || project.modelUrl) { setLocalModelUrl(""); return; }
    const url = URL.createObjectURL(project.modelBlob);
    localModelUrlRef.current = url; setLocalModelUrl(url);
    return () => { URL.revokeObjectURL(url); if (localModelUrlRef.current === url) localModelUrlRef.current = ""; };
  }, [project?.id, project?.modelBlob, project?.modelUrl]);
  useEffect(() => () => { runAbortRef.current?.abort(); if (localModelUrlRef.current) URL.revokeObjectURL(localModelUrlRef.current); }, []);

  async function persist(next: ProjectRecord) {
    const saved = { ...next, updatedAt: Date.now() };
    setProject(saved);
    setProjects(current => [saved, ...current.filter(item => item.id !== saved.id)].sort((a, b) => b.updatedAt - a.updatedAt));
    await saveProject(saved);
    return saved;
  }

  function createProject() {
    setProjectNameInput(""); setProjectDialog({ mode: "create" }); setError("");
  }

  function openProject(item: ProjectRecord) {
    setProject(item); setScreen("project"); setSidebarCollapsed(item.status === "processing" || item.status === "ready");
    setStage(item.status === "ready" || item.status === "error" ? "studio" : item.status === "processing" ? "processing" : "intake");
    setError(""); setLastSelection(null); setModelReady(false);
  }

  function renameProject(item: ProjectRecord) {
    setProjectNameInput(item.title); setProjectDialog({ mode: "rename", target: item });
  }

  async function confirmDeleteProject() {
    if (!deleteTarget) return;
    const target = deleteTarget;
    try {
      await deleteProject(target.id);
      setProjects(current => current.filter(item => item.id !== target.id));
      if (projectRef.current?.id === target.id) {
        runAbortRef.current?.abort();
        projectRef.current = null;
        setProject(null); setScreen("dashboard"); setStage("intake"); setSidebarCollapsed(false);
        setLastSelection(null); setModelReady(false); setLocalModelUrl(""); setError("");
      }
      setDeleteTarget(null);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Could not delete the project.");
      setDeleteTarget(null);
    }
  }

  async function submitProjectDialog(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const title = projectNameInput.trim();
    if (!title) return;
    if (projectDialog?.mode === "create") {
      const created = await newProject(title);
      setProjects(current => [created, ...current]); setProject(created); setScreen("project"); setStage("intake"); setSidebarCollapsed(false); setError("");
    } else if (projectDialog?.mode === "rename") {
      if (title !== projectDialog.target.title) await persist({ ...projectDialog.target, title });
    }
    setProjectDialog(null);
  }

  const pushEvent = useCallback((label: string, state: ProjectEvent["state"]) => {
    setProject(current => {
      if (!current) return current;
      const updated: ProjectRecord = { ...current, events: [...current.events.map(entry => entry.state === "active" ? { ...entry, state: "done" as const } : entry), { id: eventId(), label, state, at: Date.now() }], updatedAt: Date.now() };
      setProjects(list => [updated, ...list.filter(entry => entry.id !== updated.id)].sort((a, b) => b.updatedAt - a.updatedAt));
      void saveProject(updated);
      return updated;
    });
  }, []);

  const consumeRunUpdate = useCallback((update: RunUpdate) => {
    if (update.type === "status") pushEvent(summarizeStatus(update.status), "active");
    if (update.type === "metric") pushEvent(`Similarity updated${update.similarity == null ? "" : ` · ${Math.round(update.similarity)}%`}`, "done");
    if (update.type === "model" && update.modelUrl) {
      setProject(current => {
        if (!current) return current;
        const firstIteration = { label: `Iteration ${(current.iterations?.length || 0) + 1} · ${current.iterations?.length ? "Refined model" : "Base model"}`, modelId: update.modelId || current.modelId, modelUrl: update.modelUrl, createdAt: Date.now() };
        const updated = { ...current, modelUrl: update.modelUrl, modelId: update.modelId || current.modelId, status: "ready" as const, iterations: [...(current.iterations || []), firstIteration] };
        setProjects(list => [updated, ...list.filter(entry => entry.id !== updated.id)].sort((a, b) => b.updatedAt - a.updatedAt));
        void saveProject(updated); return updated;
      });
      setStage("studio"); pushEvent("Model updated", "done");
    }
  }, [pushEvent]);

  const beginReconstruction = useCallback(async (target: ProjectRecord) => {
    if (target.references.length < 1) return;
    setStage("processing"); setSidebarCollapsed(true); setError(""); setModelReady(false); setLastSelection(null);
    runAbortRef.current?.abort();
    const controller = new AbortController(); runAbortRef.current = controller;
    const running = { ...target, status: "processing" as const, events: [
      ...target.events.map(entry => entry.state === "active" ? { ...entry, state: "done" as const } : entry),
      { id: eventId(), label: "Sending camera and reference captures", state: "active" as const, at: Date.now() },
    ] };
    const saved = await persist(running);
    try {
      const files = saved.references.map(asset => new File([asset.blob], asset.name, { type: asset.type }));
      const result = await startReconstruction(saved.id, files, consumeRunUpdate, controller.signal, runId => {
        // Record the run as soon as it exists so the search panel can follow it live.
        setProject(current => {
          if (!current || current.id !== saved.id) return current;
          const updated = { ...current, runId };
          projectRef.current = updated; void saveProject(updated);
          return updated;
        });
      });
      const latest = projectRef.current?.id === saved.id ? projectRef.current : saved;
      const ready = { ...latest, runId: result.runId, modelId: result.modelId || latest.modelId, modelUrl: result.modelUrl || latest.modelUrl, status: (result.modelUrl || latest.modelUrl) ? "ready" as const : "processing" as const };
      await persist(ready);
      if (result.modelUrl || latest.modelUrl) { setStage("studio"); pushEvent("Search finished", "done"); }
    } catch (runError) {
      if (controller.signal.aborted) return;
      const message = runError instanceof Error ? runError.message : "Reconstruction could not start.";
      setError(message);
      const latest = projectRef.current?.id === saved.id ? projectRef.current : saved;
      const failed: ProjectRecord = { ...latest, status: "error", events: [...latest.events.map(item => item.state === "active" ? { ...item, state: "done" as const } : item), { id: eventId(), label: message, state: "error", at: Date.now() }] };
      projectRef.current = failed; setProject(failed);
      setProjects(current => current.map(item => item.id === saved.id ? failed : item));
      void saveProject(failed);
      setStage("studio");
    }
  }, [consumeRunUpdate, persist, pushEvent]);

  async function addImages(files: FileList | File[]) {
    if (!project || stage !== "intake") return;
    const accepted = Array.from(files).filter(file => file.type.startsWith("image/"));
    if (!accepted.length) { setError("Choose image files to continue."); return; }
    const additions: ReferenceAsset[] = accepted
      .filter(file => !project.references.some(item => item.name === file.name && item.blob.size === file.size))
      .map(file => ({ id: eventId(), name: file.name, type: file.type, blob: file, addedAt: Date.now() }));
    if (!additions.length) return;
    const next = await persist({ ...project, references: [...project.references, ...additions], status: "new" });
    setError("");
  }

  function addDashboardImages(files: FileList | File[]) {
    const accepted = Array.from(files).filter(file => file.type.startsWith("image/"));
    if (!accepted.length) { setError("Choose image files to continue."); return; }
    setDashboardImages(current => [...current, ...accepted.filter(file => !current.some(item => item.name === file.name && item.size === file.size))]);
    setError("");
  }

  async function startDashboardRun() {
    if (dashboardImages.length < 1) { setError("Add a reference image to generate a model."); return; }
    const now = Date.now();
    const title = `Reconstruction · ${new Date(now).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    const created = await newProject(title);
    const saved = { ...created, references: dashboardImages.map(file => ({ id: eventId(), name: file.name, type: file.type, blob: file, addedAt: now })) };
    await saveProject(saved);
    setProjects(current => [saved, ...current]); setProject(saved); projectRef.current = saved; setScreen("project"); setStage("processing");
    void beginReconstruction(saved);
    setDashboardImages([]); setError("");
  }

  function onFileInput(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      if (screen === "dashboard") addDashboardImages(event.target.files);
      else void addImages(event.target.files);
    }
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (screen === "dashboard") { addDashboardImages(event.dataTransfer.files); return; }
    void addImages(event.dataTransfer.files);
  }

  async function removeReference(id: string) {
    if (!project || stage !== "intake") return;
    await persist({ ...project, references: project.references.filter(asset => asset.id !== id) });
  }

  async function loadLocalModel(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = "";
    if (!file || !project) return;
    if (!file.name.toLowerCase().endsWith(".glb")) { setError("Choose a binary .glb model."); return; }
    if (localModelUrlRef.current) URL.revokeObjectURL(localModelUrlRef.current);
    localModelUrlRef.current = URL.createObjectURL(file);
    setLocalModelUrl(localModelUrlRef.current);
    const updated = await persist({ ...project, modelBlob: file, modelName: file.name, status: "ready" });
    setProject(updated); setStage("studio"); setSidebarCollapsed(true); setModelReady(false); setError("");
    pushEvent("Local model loaded", "done");
  }

  const [runInfo, setRunInfo] = useState<RunSummary | null>(null);
  const runId = project?.runId;
  useEffect(() => {
    if (!runId) { setRunInfo(null); return; }
    let alive = true;
    const tick = () => fetchRun(runId).then(r => { if (alive) setRunInfo(r); }).catch(() => undefined);
    void tick();
    const timer = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(timer); };
  }, [runId]);
  // The best scored team's model, from polling (the live stream can be missed, e.g. after a reload).
  useEffect(() => {
    const glb = runInfo?.best?.glb;
    if (!glb) return;
    setProject(current => {
      if (!current || current.runId !== runInfo?.runId || current.iterations?.some(i => i.modelUrl === glb)) return current;
      const item = { label: `Iteration ${(current.iterations?.length || 0) + 1} · Best team ${runInfo.best?.structureHash.slice(0, 8)} · ${Math.round((runInfo.best?.fitness || 0) * 100)}%`, modelId: runInfo.best?.structureHash || current.modelId, modelUrl: glb, createdAt: Date.now() };
      const updated = { ...current, modelUrl: glb, modelId: item.modelId, iterations: [...(current.iterations || []), item] };
      projectRef.current = updated; void saveProject(updated);
      setProjects(list => list.map(entry => entry.id === updated.id ? updated : entry));
      return updated;
    });
  }, [runInfo]);
  const previewUrl = !project?.modelUrl && !localModelUrl ? runInfo?.latestModel?.url : undefined;
  // Before any .glb exists, the latest live render stands in for the model.
  const [latestRender, setLatestRender] = useState<RenderItem | null>(null);
  const needRender = !!runId && !project?.modelUrl && !localModelUrl && !previewUrl;
  useEffect(() => {
    if (!needRender || !runId) { setLatestRender(null); return; }
    let alive = true;
    const load = () => fetchRenders(runId, undefined, 12).then(r => {
      if (alive) setLatestRender(r.find(i => i.kind === "preview" || i.kind === "stage") || null);
    }).catch(() => undefined);
    void load();
    const timer = setInterval(load, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [needRender, runId]);
  const modelUrl = project?.modelUrl || localModelUrl || previewUrl || undefined;

  const handleSample = useCallback((sample: HandSample | null) => {
    handSampleRef.current = sample;
    const now = performance.now();
    if (!sample || now - handUiUpdateAt.current > 120) {
      handUiUpdateAt.current = now;
      setHandSample(sample);
    }
  }, []);

  const handleSelection = useCallback((selection: SelectionPacket) => {
    setLastSelection(selection);
    pushEvent("Sculpt captured · ready to refine", "done");
  }, [pushEvent]);

  const submitRefinement = useCallback(async () => {
    if (!project || !lastSelection) return;
    if (!project.runId || !project.modelUrl) {
      setError("Connect a reconstruction run and refinement endpoint to send this sculpt.");
      pushEvent("Selection saved locally · refinement endpoint needed", "queued"); return;
    }
    setRefining(true); setError(""); pushEvent("Sending sculpt to refinement model", "active");
    try {
      const updated = await refineSelection(project.runId, project.modelId || project.runId, lastSelection);
      const nextIteration = { label: `Iteration ${(project.iterations?.length || 1) + 1} · AI refinement`, modelId: updated.modelId || project.modelId, modelUrl: updated.modelUrl, createdAt: Date.now(), selection: lastSelection };
      await persist({ ...project, modelId: updated.modelId || project.modelId, modelUrl: updated.modelUrl, status: "ready", iterations: [...(project.iterations || []), nextIteration] });
      setLastSelection(null); pushEvent("Refinement complete · model updated", "done");
    } catch (refineError) {
      setError(refineError instanceof Error ? refineError.message : "Refinement could not start.");
      pushEvent("Refinement service unavailable", "error");
    } finally { setRefining(false); }
  }, [project, lastSelection, persist, pushEvent]);


  return <main className={`app-shell${sidebarCollapsed && screen === "project" ? " sidebar-collapsed" : ""}`} onDrop={onDrop} onDragOver={event => event.preventDefault()}>
    <input ref={uploadRef} type="file" accept="image/*" multiple hidden onChange={onFileInput} />
    <input ref={modelRef} type="file" accept=".glb,model/gltf-binary" hidden onChange={loadLocalModel} />
    {projectDialog && <div className="dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setProjectDialog(null); }}><form className="project-dialog" onSubmit={event => void submitProjectDialog(event)}><span className="kicker">PROJECT</span><h2>{projectDialog.mode === "create" ? "Create a project" : "Rename project"}</h2><label htmlFor="project-name">Project name</label><input id="project-name" autoFocus maxLength={60} value={projectNameInput} onChange={event => setProjectNameInput(event.target.value)} placeholder="e.g. Ceramic lamp"/><div className="dialog-actions"><button className="text-button" type="button" onClick={() => setProjectDialog(null)}>Cancel</button><button className="primary-button" type="submit" disabled={!projectNameInput.trim()}>{projectDialog.mode === "create" ? "Create project" : "Save name"}</button></div></form></div>}
    {deleteTarget && <div className="dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setDeleteTarget(null); }}><section className="project-dialog delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-title" aria-describedby="delete-description"><span className="kicker">DELETE PROJECT</span><h2 id="delete-title">Delete “{deleteTarget.title}”?</h2><p id="delete-description">This removes the project, saved images, and model history from this browser.</p><div className="dialog-actions"><button className="text-button" onClick={() => setDeleteTarget(null)}>Cancel</button><button className="danger-button" onClick={() => void confirmDeleteProject()}>Delete project</button></div></section></div>}
    <aside className="project-sidebar">
      <div className="sidebar-brand"><img src="/assets/refresh-logo.svg" alt=""/><span>Refresh</span></div>
      <div className="sidebar-section"><span>PROJECTS</span></div>
      <nav className="sidebar-projects" aria-label="Projects">
        {projects.map(item => <div key={item.id} className="sidebar-project-row"><button className={`sidebar-project${project?.id === item.id && screen === "project" ? " active" : ""}`} onClick={() => openProject(item)} title={item.title}>{item.title}</button><button className="sidebar-project-delete" title={`Delete ${item.title}`} aria-label={`Delete ${item.title}`} onClick={() => setDeleteTarget(item)}>×</button></div>)}
        {!projects.length && <span className="sidebar-empty">No projects yet</span>}
      </nav>
      <div className="sidebar-profile"><span>T</span><b>test</b></div>
    </aside>

    {screen === "dashboard" ? <section className="dashboard" onDrop={onDrop} onDragOver={event => event.preventDefault()}>
      <div className="dashboard-intake">
        <div className="dashboard-capture"><HandCamera enabled captureMode onCapture={file => addDashboardImages([file])} onSample={handleSample}/></div>
        <div className={`dashboard-drop${dashboardPreviews.length ? " has-images" : ""}`} onClick={() => uploadRef.current?.click()} onDragOver={event => event.preventDefault()} onDragEnter={event => event.currentTarget.classList.add("is-dragging")} onDragLeave={event => event.currentTarget.classList.remove("is-dragging")} onDrop={event => { event.preventDefault(); event.stopPropagation(); event.currentTarget.classList.remove("is-dragging"); addDashboardImages(event.dataTransfer.files); }} role="button" tabIndex={0} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") uploadRef.current?.click(); }}>
          {!dashboardPreviews.length ? <><span className="upload-mark"><svg viewBox="0 0 32 32"><path d="M16 22V8m0 0-5 5m5-5 5 5M7 20v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/></svg></span><b>Drop images here</b><small>or browse files</small></> : <div className="dashboard-image-grid">{dashboardPreviews.map((image, index) => <div key={`${image.file.name}-${index}`} className="dashboard-image"><img src={image.url} alt={image.file.name}/><button onClick={event => { event.stopPropagation(); setDashboardImages(items => items.filter((_, itemIndex) => itemIndex !== index)); }} aria-label={`Remove ${image.file.name}`}>×</button></div>)}<span className="dashboard-add-image">＋ Add images</span></div>}
        </div>
        <div className="dashboard-intake-footer"><span>{dashboardImages.length} views · camera or upload</span><a href="/pinch-demo">Pigeon hand sculpt demo →</a><button className="primary-button" onClick={() => void startDashboardRun()} disabled={dashboardImages.length < 1}>Generate base model <span>→</span></button></div>
      </div>
      {error && <div className="toast-message">{error}<button onClick={() => setError("")}>×</button></div>}
    </section> : <section className="project-workspace">
      <header className="app-topbar project-topbar"><div className="topbar-title"><button className="nav-toggle" onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-label="Toggle projects">☰</button><span className="project-breadcrumb">Projects</span><span className="slash">/</span><button className="project-heading-button" onClick={() => project && renameProject(project)}>{project?.title}</button></div><div className="topbar-actions">{stage === "studio" && modelUrl && <a className="text-button" href={modelUrl} download={project?.modelName || "refresh-model.glb"}>Export .glb</a>}{stage === "studio" && modelReady && lastSelection && <button className="primary-button finish-button" onClick={() => void submitRefinement()} disabled={refining}>{refining ? "Refining…" : "Refine selection"} <span>→</span></button>}<button className="text-button" onClick={() => { runAbortRef.current?.abort(); setScreen("dashboard"); setSidebarCollapsed(false); setError(""); }}>Back to projects</button></div></header>
      {stage === "intake" ? <div className="intake-page"><div className="intake-intro"><span className="kicker">NEW RECONSTRUCTION</span><h1>Capture your object</h1><p>Take a few views with your camera, then generate the base model. You can add reference photos too.</p></div>
        <div className="project-capture-stack"><HandCamera enabled captureMode onCapture={file => void addImages([file])} onSample={handleSample}/></div>
        <div className="reference-drop" onClick={() => uploadRef.current?.click()} onDragEnter={event => { event.preventDefault(); event.currentTarget.classList.add("is-dragging"); }} onDragOver={event => event.preventDefault()} onDragLeave={event => event.currentTarget.classList.remove("is-dragging")} onDrop={event => { event.preventDefault(); event.stopPropagation(); event.currentTarget.classList.remove("is-dragging"); void addImages(event.dataTransfer.files); }} role="button" tabIndex={0} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") uploadRef.current?.click(); }}>
          {!previews.length ? <div className="drop-prompt"><div className="upload-mark"><svg viewBox="0 0 32 32"><path d="M16 22V8m0 0-5 5m5-5 5 5M7 20v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/></svg></div><b>Drop reference images here</b><span>or <em>browse files</em></span><small>PNG, JPG, WEBP · or capture views above</small></div> : <div className="reference-grid">{previews.map((image, index) => <div className="reference-tile" key={image.id} onClick={event => event.stopPropagation()}><img src={image.url} alt={image.name}/><span>{String(index + 1).padStart(2, "0")}</span><button onClick={() => void removeReference(image.id)} aria-label={`Remove ${image.name}`}>×</button><small>{image.name}</small></div>)}<button className="add-reference" onClick={event => { event.stopPropagation(); uploadRef.current?.click(); }}>＋<small>Add views</small></button></div>}
        </div>
        <div className="intake-bottom"><span>{previews.length} captured views</span><span>{previews.length < 1 ? "Capture or add a reference view" : "Ready to generate a base model"}</span><button className="primary-button" disabled={previews.length < 1} onClick={event => { event.stopPropagation(); if (project) void beginReconstruction(project); }}>Generate base model <span>→</span></button></div>
        {error && <div className="inline-error">{error}</div>}
      </div> : <div className="studio-layout search-first">
        <div className="studio-main search-main">
          <div className="studio-title"><div><span className="kicker">AGENT SEARCH</span><h1>Generations</h1></div>{project?.status === "error" && <div className="studio-status">Service unavailable</div>}</div>
          <SearchPanel runId={project?.runId}/>
          {error && <div className="inline-error">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
        </div>
        <aside className="process-panel model-panel">
          <div className="model-panel-head"><span className="kicker">BEST MODEL SO FAR</span><b>{project?.modelName || (previewUrl ? `Work in progress · ${runInfo?.latestModel?.role || "agent"}'s model` : modelUrl ? "Current best team" : latestRender ? `Live · ${latestRender.role || "agent"} is building` : "Waiting for the first agent to finish")}</b><small>{previewUrl ? "Not scored yet · the best scored team replaces it" : modelReady ? "Pinch to select a region, then refine" : "Updates each time a team beats the best score"}</small></div>
          <div className="model-stage compact"><ModelViewport modelUrl={modelUrl} handSample={handSampleRef} onSelection={handleSelection} onModelReady={setModelReady} autoRotate={false}/>
            {!modelUrl && latestRender && <><img className="model-preview-img" src={`${latestRender.url}?t=${Math.round(latestRender.t)}`} alt="Latest live render" /><span className="model-preview-tag">live stage view · {latestRender.role}</span></>}
            {!modelUrl && !latestRender && <button className="load-model" onClick={() => modelRef.current?.click()}>Load .glb</button>}
            {lastSelection && <div className="selection-chip">Region selected · {lastSelection.faces.length} faces <span>{refining ? "Refining…" : ""}</span></div>}
          </div>
          <div className="model-toolbar"><span>{project?.references.length || 0} reference views</span><span className="tool-divider"/><button onClick={() => modelRef.current?.click()}>Load .glb</button><span className="tool-spacer"/><span>{handSample ? `Hand ${Math.round(handSample.distanceMm)} mm` : "Pinch to sculpt"}</span></div>
          <div className="iteration-strip">{(project?.iterations?.length ? project.iterations : modelUrl ? [{ label: "Iteration 1 · Base model", modelUrl, createdAt: project?.updatedAt || 0 }] : []).map((item, index) => <span key={`${item.label}-${index}`} className="iteration-item"><i>{index + 1}</i>{item.label}{index < (project?.iterations?.length || 1) - 1 && <b>→</b>}</span>)}</div>
          {modelReady && stage === "studio" && <HandCamera enabled compact onSample={handleSample}/>}
          <details className="thinking-trace" open><summary><span className="thinking-icon">✳</span><b>Run log</b><small>{project?.events?.length || 0} updates</small><i>⌄</i></summary>
            <div className="process-timeline">{(project?.events || []).length ? project!.events.slice(-8).map(item => <div className={`process-event ${item.state}`} key={item.id}><span className="event-marker">{item.state === "done" ? "✓" : item.state === "error" ? "!" : item.state === "active" ? <i/> : "·"}</span><div><b>{item.label}</b><small>{new Date(item.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small></div></div>) : <div className="process-placeholder">Run activity will appear here.</div>}</div>
          </details>
        </aside>
      </div>}
    </section>}
  </main>;
}
