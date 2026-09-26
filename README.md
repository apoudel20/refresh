# Refresh

Long-term memory using ablation on agent DAGs.

Refresh is a Blender reconstruction frontend. Users create local projects, capture or upload reference images, request a base model from a reconstruction backend, inspect the GLB, then pinch-sculpt a region and send that edit to a refinement backend. Browser hand tracking and the pigeon interaction demo run locally.

## Run the UI

Requirements: Node.js 20+ and npm.

```sh
npm install
npm run dev
```

Open `http://localhost:3000`. Camera permission is required for capture and hand tracking. Browsers allow camera access on `localhost` or HTTPS. The standalone pinch sculpt demo is at `http://localhost:3000/pinch-demo`; it works without a reconstruction API.

Create a local environment file from the provided example:

```sh
cp .env.example .env.local
```

Set the API origin in `.env.local`:

```env
NEXT_PUBLIC_REFRESH_API_URL=http://localhost:8000
```

Restart Next.js after changing environment variables. Leave the value empty to run the UI and demo without a backend. Do not put secrets in a `NEXT_PUBLIC_*` variable: it is exposed to the browser.

## Integration guide

This README contains the full setup and backend contract, including request/response formats, SSE progress events, CORS, model hosting, authentication notes, and the code files to change.

## Backend contract

The UI adapter is `app/lib/harness-client.ts`. It uses the configured API origin and these endpoints. Keep the UI-facing contract stable even if your agent tree, MCP tools, Blender scripts, or job system change internally.

### 1. Start a reconstruction

On **Generate base model**, the browser posts at least two references as `multipart/form-data`:

```http
POST /api/reconstructions
Content-Type: multipart/form-data
```

Parts:

- `projectId`: browser-generated project ID
- `images`: one part per camera capture or uploaded image; this field repeats

The initial response can be `200` or `202` JSON:

```json
{
  "runId": "run_123",
  "modelId": "model_01",
  "modelUrl": "https://assets.example.com/model_01.glb",
  "eventsUrl": "http://localhost:8000/api/reconstructions/run_123/events"
}
```

`runId` is required. `modelId`, `modelUrl`, and `eventsUrl` may be omitted while work is running. If the model is not included in the initial response, the SSE stream must emit a `model` event with a `modelUrl` before it closes. The browser stores `runId`/model metadata on the project so later refinement calls can identify the run.

### 2. Stream progress (SSE)

If `eventsUrl` is returned, the browser opens it with native `EventSource` using GET. Use `Content-Type: text/event-stream`, disable response buffering, and send each event as a JSON `data:` line followed by a blank line:

```text
data: {"type":"status","status":"Segmenting reference views"}

data: {"type":"metric","similarity":82}

data: {"type":"status","status":"Generating geometry"}

data: {"type":"model","modelId":"model_01","modelUrl":"https://assets.example.com/model_01.glb"}

data: [DONE]

```

Supported payloads:

- `{"type":"status","status":"..."}` — lifecycle status; the UI maps this to a short activity label.
- `{"type":"metric","similarity":82}` — optional numeric similarity percentage.
- `{"type":"model","modelId":"...","modelUrl":"https://.../model.glb"}` — model ready or updated; `modelUrl` is required.
- `{"type":"error","message":"Short user-safe error"}` — stops the run and shows the message.
- `[DONE]` — optional stream terminator, after the model event.

Send high-level task status and metrics only. Never send chain-of-thought/private agent reasoning. If SSE is used, the current client waits until a model event, `[DONE]`, or an error; ensure the terminal model event occurs before `[DONE]` or return the ready `modelUrl` in the initial response.

### 3. Refine a pinch-sculpt selection

After the user releases a pinch stroke, the app sends this request when they choose **Refine selection**:

```http
POST /api/reconstructions/{runId}/refine
Content-Type: application/json
```

```json
{
  "modelId": "model_01",
  "selection": {
    "points": [[0.12, 0.4, -0.08]],
    "faces": [{"meshId":"mesh-uuid","faceIndex":12}],
    "geometryEdits": [{
      "meshId":"mesh-uuid",
      "meshName":"Body",
      "vertices":[{"index":12,"original":[0,0,0],"position":[0.01,0,0]}]
    }],
    "handDistanceMm": 500,
    "screenshot": "data:image/png;base64,…"
  }
}
```

`points` are model-world coordinates. `faces` refer to the loaded GLB mesh IDs and triangle indices. `geometryEdits` contains sparse vertex positions before and after the local sculpt, in mesh-local coordinates; these indices are valid only for the exact `modelId`/topology sent with the request. `handDistanceMm` is the browser's approximate palm-width estimate. `screenshot` is a rendered viewport PNG data URL and can be large.

Return the replacement model as JSON:

```json
{"modelId":"model_02","modelUrl":"https://assets.example.com/model_02.glb"}
```

The UI appends it as the next visible iteration. The response needs a new browser-loadable `modelUrl`.

## Browser access, CORS, and auth

- Allow the frontend origin (for local development: `http://localhost:3000`) on reconstruction, SSE, refinement, and GLB asset responses.
- The model viewer loads GLBs in the browser without custom auth headers. Use public assets, time-limited signed URLs, or proxy protected assets through the Next.js app.
- Native `EventSource` cannot add arbitrary bearer headers. For authenticated SSE, use same-site cookies (and update the client for credentials) or replace it with a fetch-based SSE reader in `app/lib/harness-client.ts`.
- `NEXT_PUBLIC_REFRESH_API_URL` is public configuration, not a secret. For private API keys, add a server-side Next.js route/proxy and read the secret only on the server.
- Validate image type/count/size and authenticate/authorize project ownership in the backend. Project IDs are generated in the browser and are not an authorization mechanism.

## What the backend needs to do

A minimal backend can expose only the endpoints above. Internally, it should validate/store reference images, create a run, dispatch coordinator and worker tasks, run Blender or another reconstruction pipeline, publish concise progress events, save a GLB revision, and apply the user's sculpt edits during refinement. Keep run state and model versions addressable by `runId` and `modelId`. Use the status stream to report steps such as reading views, segmentation, geometry generation, comparison, and refinement—never private reasoning.

Suggested first connection order:

1. Implement `POST /api/reconstructions` and return a known test GLB URL.
2. Verify GLB CORS and display in the workspace.
3. Add SSE updates and terminal model events.
4. Implement `/refine` to accept a selection and return a revised GLB.
5. Add auth, job persistence, retry/cancel behavior, and production storage.

## What already works without the backend

- Project names, source images, model metadata, and iteration history persist in this browser's IndexedDB. They do not sync to a server or another device.
- Camera capture and image upload are local until the user starts a reconstruction.
- Hand landmarks, pinch detection, approximate hand-distance estimation, viewport brush, and local mesh deformation run in the browser.
- The separate pigeon demo exercises hand tracking and local sculpting without the reconstruction API.
- MediaPipe runtime files are served locally from `public/mediapipe/wasm`; the hand-landmarker model is `public/assets/hand_landmarker.task`.

## Code map

- `app/page.tsx` — dashboard, project lifecycle, event timeline, and workflow calls.
- `app/lib/harness-client.ts` — backend base URL, run request, SSE handling, refinement request.
- `app/lib/project-store.ts` — browser IndexedDB persistence.
- `app/components/HandCamera.tsx` — webcam capture, MediaPipe landmarks, pinch state, distance estimate.
- `app/components/ModelViewport.tsx` — GLB rendering, pinch selection, local mesh deformation, selection packet.
- `app/pinch-demo/page.tsx` — standalone pigeon interaction demo.
- `app/globals.css` — application and integration-guide styling.
