export type ReferenceAsset = { id: string; name: string; type: string; blob: Blob; addedAt: number };
export type ProjectEvent = { id: string; label: string; state: "done" | "active" | "queued" | "error"; at: number };
export type ModelIteration = { label: string; modelId?: string; modelUrl?: string; createdAt: number; selection?: unknown };
export type ProjectRecord = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  references: ReferenceAsset[];
  runId?: string;
  modelId?: string;
  modelUrl?: string;
  modelBlob?: Blob;
  modelName?: string;
  iterations?: ModelIteration[];
  status: "new" | "processing" | "ready" | "error";
  events: ProjectEvent[];
};

const DB_NAME = "nimbus-projects";
const DB_VERSION = 1;
const STORE = "projects";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open local project storage"));
  });
}

export async function listProjects(): Promise<ProjectRecord[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE, "readonly").objectStore(STORE).getAll();
    request.onsuccess = () => resolve((request.result as ProjectRecord[]).sort((a, b) => b.updatedAt - a.updatedAt));
    request.onerror = () => reject(request.error || new Error("Could not read projects"));
    request.transaction?.addEventListener("complete", () => db.close(), { once: true });
  });
}

export async function saveProject(project: ProjectRecord): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(project);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("Could not save project"));
    tx.onabort = () => reject(tx.error || new Error("Project save was interrupted"));
  }).finally(() => db.close());
}

export async function deleteProject(id: string): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("Could not delete project"));
    tx.onabort = () => reject(tx.error || new Error("Project deletion was interrupted"));
  }).finally(() => db.close());
}

export async function newProject(title: string): Promise<ProjectRecord> {
  const now = Date.now();
  const project: ProjectRecord = { id: crypto.randomUUID(), title, createdAt: now, updatedAt: now, references: [], status: "new", events: [] };
  await saveProject(project);
  return project;
}
