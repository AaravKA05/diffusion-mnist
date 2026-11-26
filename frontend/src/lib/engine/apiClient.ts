/*
 * Hybrid API client.
 *
 * "server"  -> the laptop FastAPI backend is reachable: use it (fastest).
 * "browser" -> it is not: run the same request on this device with ONNX
 *              Runtime Web. The page works either way.
 *
 * apiFetch returns a normal Response in both cases, so existing code that
 * checks response.ok / response.json() / response.text() keeps working.
 */

import { estimateSteps, handleLocally, LocalApiError } from "./localApi";

export type EngineMode = "server" | "browser";

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

let mode: EngineMode | null = null;
const listeners = new Set<(mode: EngineMode) => void>();

export function getEngineMode() {
  return mode;
}

export function setEngineMode(next: EngineMode) {
  if (mode === next) return;
  mode = next;
  listeners.forEach((fn) => fn(next));
}

export function onEngineModeChange(fn: (mode: EngineMode) => void) {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

// ---------------------------------------------------------------------------
// Activity: what is running right now, for the progress bar
// ---------------------------------------------------------------------------

export type Activity = {
  mode: EngineMode;
  label: string;
  /** Finished denoising steps (browser mode only) */
  done: number;
  /** Expected steps, or null when unknown (server mode) */
  total: number | null;
};

let activity: Activity | null = null;
const activityListeners = new Set<(a: Activity | null) => void>();

function setActivity(next: Activity | null) {
  activity = next;
  activityListeners.forEach((fn) => fn(next));
}

export function onActivityChange(fn: (a: Activity | null) => void) {
  activityListeners.add(fn);
  fn(activity);
  return () => {
    activityListeners.delete(fn);
  };
}

const LABELS: Record<string, string> = {
  "/api/generate": "Generating images",
  "/api/compare": "Comparing guidance scales",
  "/api/experiment": "Running experiment",
  "/api/interpolate": "Interpolating noise",
  "/api/visualize": "Capturing denoising steps",
  "/api/representation": "Reading U-Net features",
  "/api/representation/evolution": "Tracking feature evolution",
  "/api/evaluate": "Evaluating the model",
  "/api/recognition": "Tracing recognition",
  "/api/sketch": "Cleaning up your sketch",
};

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function runLocally(path: string, init?: RequestInit) {
  let body: Record<string, unknown> = {};
  if (typeof init?.body === "string" && init.body) {
    body = JSON.parse(init.body);
  }
  const total = estimateSteps(path, body);
  let done = 0;
  const label = LABELS[path] ?? "Working";
  setActivity({ mode: "browser", label, done, total: total || null });
  try {
    return jsonResponse(
      await handleLocally(path, body, () => {
        done += 1;
        setActivity({ mode: "browser", label, done: Math.min(done, total || done), total: total || null });
      })
    );
  } catch (error) {
    if (error instanceof LocalApiError) {
      return jsonResponse({ detail: error.message }, error.status);
    }
    const message = error instanceof Error ? error.message : "Browser engine failed.";
    return jsonResponse({ detail: `Browser engine: ${message}` }, 500);
  } finally {
    setActivity(null);
  }
}

/** Drop-in replacement for fetch(`${API}${path}`, init). */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  if (mode === "server") {
    setActivity({ mode: "server", label: LABELS[path] ?? "Working", done: 0, total: null });
    try {
      return await fetch(`${API}${path}`, init);
    } catch {
      // Server vanished mid-session (laptop asleep, tunnel closed): fall back
      setEngineMode("browser");
    } finally {
      setActivity(null);
    }
  }
  return runLocally(path, init);
}
