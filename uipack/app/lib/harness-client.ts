export type RunUpdate = {
  type: "status" | "model" | "metric" | "error";
  status?: string;
  message?: string;
  modelUrl?: string;
  modelId?: string;
  similarity?: number;
};

function baseUrl() {
  return process.env.NEXT_PUBLIC_REFRESH_API_URL?.replace(/\/$/, "") || "";
}

export async function startReconstruction(projectId: string, files: File[], onUpdate: (update: RunUpdate) => void, signal: AbortSignal) {
  const base = baseUrl();
  if (!base) throw new Error("Set NEXT_PUBLIC_REFRESH_API_URL to connect the reconstruction service.");
  const body = new FormData();
  body.append("projectId", projectId);
  files.forEach(file => body.append("images", file, file.name));
  const response = await fetch(`${base}/api/reconstructions`, { method: "POST", body, signal });
  if (!response.ok) throw new Error(`Reconstruction service returned ${response.status}.`);
  const result = await response.json() as { runId: string; modelId?: string; modelUrl?: string; eventsUrl?: string };
  if (!result.runId) throw new Error("The reconstruction service did not return a runId.");
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
          if (update.type === "model" && update.modelUrl) finish();
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
