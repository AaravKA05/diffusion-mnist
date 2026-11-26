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

import { handleLocally, LocalApiError } from "./localApi";

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
  try {
    return jsonResponse(await handleLocally(path, body));
  } catch (error) {
    if (error instanceof LocalApiError) {
      return jsonResponse({ detail: error.message }, error.status);
    }
    const message = error instanceof Error ? error.message : "Browser engine failed.";
    return jsonResponse({ detail: `Browser engine: ${message}` }, 500);
  }
}

/** Drop-in replacement for fetch(`${API}${path}`, init). */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  if (mode === "server") {
    try {
      return await fetch(`${API}${path}`, init);
    } catch {
      // Server vanished mid-session (laptop asleep, tunnel closed): fall back
      setEngineMode("browser");
    }
  }
  return runLocally(path, init);
}
