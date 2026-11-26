/*
 * Browser runtime for DiffusionLab.
 *
 * Loads ONNX Runtime Web from /ort (copied there by scripts/copy-ort.mjs) and
 * the exported models from /models (made by export_onnx.py). Everything runs
 * on the visitor's own device, inside a Web Worker so the page stays smooth.
 */

import type * as OrtTypes from "onnxruntime-web";

type Ort = typeof OrtTypes;

export type Schedule = {
  T: number;
  img_size: number;
  num_classes: number;
  alpha_bar: number[];
};

export type Runtime = {
  ort: Ort;
  unet: OrtTypes.InferenceSession;
  clf: OrtTypes.InferenceSession;
  schedule: Schedule;
  threads: number;
};

let runtimePromise: Promise<Runtime> | null = null;

async function loadOrt(): Promise<Ort> {
  // Loaded at run time from /public/ort, not bundled by Next.js
  const url = "/ort/ort.wasm.min.mjs";
  const ort: Ort = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ url);

  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.proxy = true;
  ort.env.wasm.numThreads =
    typeof self !== "undefined" && self.crossOriginIsolated
      ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))
      : 1;
  ort.env.logLevel = "error";

  return ort;
}

async function create(): Promise<Runtime> {
  const ort = await loadOrt();

  const [scheduleResponse, unet, clf] = await Promise.all([
    fetch("/models/schedule.json"),
    ort.InferenceSession.create("/models/unet.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("/models/classifier.onnx", { executionProviders: ["wasm"] }),
  ]);

  if (!scheduleResponse.ok) {
    throw new Error("Could not load the noise schedule.");
  }

  const schedule: Schedule = await scheduleResponse.json();

  return { ort, unet, clf, schedule, threads: ort.env.wasm.numThreads as number };
}

/** Load once, reuse for every request. A failed load can be retried. */
export function getRuntime(): Promise<Runtime> {
  if (!runtimePromise) {
    runtimePromise = create().catch((error) => {
      runtimePromise = null;
      throw error;
    });
  }
  return runtimePromise;
}

/** Start downloading the models in the background (about 25 MB, once). */
export function preloadRuntime() {
  getRuntime().catch(() => {
    // Errors are reported when a request actually needs the runtime
  });
}
