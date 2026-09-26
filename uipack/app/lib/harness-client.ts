export type RunUpdate = {
  type: "status" | "model" | "metric" | "error";
  status?: string;
  message?: string;
  modelUrl?: string;
  modelId?: string;
  similarity?: number;
};

export function baseUrl() {
  return process.env.NEXT_PUBLIC_REFRESH_API_URL?.replace(/\/$/, "") || "";
}

export async function startReconstruction(projectId: string, files: File[], onUpdate: (update: RunUpdate) => void, signal: AbortSignal, onStarted?: (runId: string) => void) {
  const base = baseUrl();
  if (!base) throw new Error("Set NEXT_PUBLIC_REFRESH_API_URL to connect the reconstruction service.");
  const body = new FormData();
  body.append("projectId", projectId);
  files.forEach(file => body.append("images", file, file.name));
  const response = await fetch(`${base}/api/reconstructions`, { method: "POST", body, signal });
  if (!response.ok) throw new Error(`Reconstruction service returned ${response.status}.`);
  const result = await response.json() as { runId: string; modelId?: string; modelUrl?: string; eventsUrl?: string };
  if (!result.runId) throw new Error("The reconstruction service did not return a runId.");
  onStarted?.(result.runId);
  if (result.modelUrl) onUpdate({ type: "model", modelUrl: result.modelUrl, modelId: result.modelId, status: "Model ready" });
  if (result.eventsUrl) {
    await new Promise<void>((resolve, reject) => {
      const events = new EventSource(new URL(result.eventsUrl!, base).toString());
      const finish = () => { events.close(); resolve(); };
      events.onmessage = message => {
        if (message.data === "[DONE]") { finish(); return; }
        try {
          const update = JSON.parse(message.data) as RunUpdate;
          onUpdate(update);
          // Keep listening after a model event: the search keeps improving it until [DONE].
          if (update.type === "error") { events.close(); reject(new Error(update.message || "Reconstruction failed.")); }
        } catch { /* Ignore non-JSON keepalives. */ }
      };
      events.onerror = () => { events.close(); reject(new Error("Lost the reconstruction event stream.")); };
      signal.addEventListener("abort", () => { events.close(); reject(new DOMException("Run cancelled", "AbortError")); }, { once: true });
    });
  }
  return result;
}

export async function refineSelection(runId: string, modelId: string, selection: unknown, signal?: AbortSignal) {
  const base = baseUrl();
  if (!base) throw new Error("Set NEXT_PUBLIC_REFRESH_API_URL to connect the refinement service.");
  const response = await fetch(`${base}/api/reconstructions/${encodeURIComponent(runId)}/refine`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ modelId, selection }), signal,
  });
  if (!response.ok) throw new Error(`Refinement service returned ${response.status}.`);
  return await response.json() as { modelUrl: string; modelId?: string };
}

// ── Search dashboard (Refresh API) ─────────────────────────────────────────

export type StructureCard = {
  structureHash: string; generation?: number; origin?: string; fitness?: number | null; samples: number;
  nodes: { nodeId: string; role: string; tools: string[] }[];
  edges: { from: string; to: string; type?: string }[];
  scores?: Record<string, number> | null; critiqueFixes: string[];
  render?: string; glb?: string; overview?: string; costUsd?: number; parents: string[];
  turntable?: string | null; frontMatch?: number | null; disqualified?: string[];
  solidity?: { solidity: number; closure: number; thickness: number } | null;
};
export type StructureDetail = StructureCard & {
  critique: string;
  modules?: { node_id: string; role: string; note?: string; calls: number; cost_usd?: number; output?: { summary?: string } }[];
  similar?: { structureHash: string; similarity: number; fitness?: number | null; roles: string[] }[];
};
export type RunSummary = {
  runId: string; status: string; running: boolean; reference: string; best: StructureCard | null;
  counts: Record<string, number>; generations: number; k: number; statusReason?: string | null;
  latestModel?: { url: string; role?: string; structureHash?: string; generation?: number; t?: number } | null;
  current?: { role?: string; event?: string; tool?: string; structureHash?: string; generation?: number; ts?: number } | null;
};
export type AgentEvent = {
  ts: number; event: string; node_id?: string; role?: string; tool?: string; ok?: boolean; error?: string | null;
  result?: string; overall?: number; top_feedback?: string[]; text?: string; args?: Record<string, unknown>;
  reason?: string; duration_ms?: number; model?: string; iteration?: number;
};

export type RenderItem = {
  url: string; kind: "preview" | "stage" | "turntable" | "image"; name: string; t: number;
  role?: string; structureHash?: string; nodeId?: string; generation?: number;
  overall?: number; front_match?: number; solidity?: number;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${baseUrl()}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Refresh API ${path} returned ${response.status}.`);
  return await response.json() as T;
}

export const fetchRun = (runId: string) => getJson<RunSummary>(`/api/runs/${encodeURIComponent(runId)}`);
export const fetchStructures = (runId: string) => getJson<StructureCard[]>(`/api/runs/${encodeURIComponent(runId)}/structures`);
export const fetchStructure = (runId: string, hash: string) => getJson<StructureDetail>(`/api/runs/${encodeURIComponent(runId)}/structures/${hash}`);
export const fetchAgentEvents = (runId: string, hash: string, nodeId?: string) =>
  getJson<AgentEvent[]>(`/api/runs/${encodeURIComponent(runId)}/agent-events?structure_hash=${hash}${nodeId ? `&node_id=${encodeURIComponent(nodeId)}` : ""}&limit=500`);

export const fetchRenders = (runId: string, structureHash?: string, limit = 60) =>
  getJson<RenderItem[]>(`/api/runs/${encodeURIComponent(runId)}/renders?limit=${limit}${structureHash ? `&structure_hash=${structureHash}` : ""}`);

export async function stopRun(runId: string) {
  await fetch(`${baseUrl()}/api/reconstructions/${encodeURIComponent(runId)}/stop`, { method: "POST" });
}
