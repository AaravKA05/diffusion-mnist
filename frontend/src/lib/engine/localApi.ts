/*
 * The FastAPI endpoints, re-implemented to run in the browser.
 * Each handler takes the same JSON body as server2.py and returns the same
 * JSON shape, so the page code does not care where the answer came from.
 */

import { decodeSketch, greyToPng, resizeBilinear } from "./images";
import { getRuntime, type Runtime } from "./runtime";
import {
  argmax,
  classify,
  ddimStep,
  ddimTimesteps,
  guidedEps,
  imageAt,
  makeRng,
  partialTimesteps,
  randn,
  sampleDdim,
} from "./sampler";

const P = 32 * 32;

export class LocalApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

type Body = Record<string, unknown>;

// ---------------------------------------------------------------------------
// Small validation helpers (same limits as the pydantic models)
// ---------------------------------------------------------------------------

function int(body: Body, key: string, def: number, min: number, max: number) {
  const raw = body[key] ?? def;
  const v = Number(raw);
  if (!Number.isFinite(v) || !Number.isInteger(v) || v < min || v > max) {
    throw new LocalApiError(422, `${key} must be an integer between ${min} and ${max}.`);
  }
  return v;
}

function num(body: Body, key: string, def: number, min: number, max: number) {
  const raw = body[key] ?? def;
  const v = Number(raw);
  if (!Number.isFinite(v) || v < min || v > max) {
    throw new LocalApiError(422, `${key} must be between ${min} and ${max}.`);
  }
  return v;
}

function list(body: Body, key: string, def: number[], minLen: number, maxLen: number) {
  const raw = (body[key] ?? def) as unknown;
  if (!Array.isArray(raw) || raw.length < minLen || raw.length > maxLen) {
    throw new LocalApiError(422, `${key} must have ${minLen} to ${maxLen} values.`);
  }
  return raw.map((v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) throw new LocalApiError(422, `${key} must contain numbers.`);
    return n;
  });
}

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
const now = () => performance.now();
const secondsSince = (start: number, d = 2) => round((now() - start) / 1000, d);

// ---------------------------------------------------------------------------
// Shared generation (port of server2.run)
// ---------------------------------------------------------------------------

async function run(rt: Runtime, digit: number, count: number, w: number, steps: number, seed: number, size: number) {
  const rng = makeRng(seed);
  const noise = randn(count * P, rng);
  const { images } = await sampleDdim(rt, {
    n: count,
    labels: new Array(count).fill(digit),
    w,
    steps,
    noise,
  });
  const probs = await classify(rt, images, count);
  const items = probs.map((row, k) => {
    const pred = argmax(row);
    return {
      image: greyToPng(imageAt(images, k), 32, 32, size),
      pred,
      conf: round(row[pred], 3),
    };
  });
  const matches = items.filter((item) => item.pred === digit).length;
  return { items, matches, images, probs };
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

async function generate(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const count = int(body, "count", 8, 1, 16);
  const w = num(body, "w", 1.2, 0, 3);
  const steps = int(body, "steps", 20, 5, 100);
  const seed = int(body, "seed", 0, 0, 2 ** 31 - 1);
  const start = now();
  const { items, matches } = await run(rt, digit, count, w, steps, seed, 192);
  return { items, matches, count, seconds: secondsSince(start), params: { digit, count, w, steps, seed } };
}

async function compare(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const count = int(body, "count", 6, 1, 8);
  const ws = list(body, "ws", [0, 1.2, 2], 2, 4).map((w) => Math.min(Math.max(w, 0), 3));
  const steps = int(body, "steps", 20, 5, 100);
  const seed = int(body, "seed", 0, 0, 2 ** 31 - 1);
  const start = now();
  const results = [];
  for (const w of ws) {
    const { items, matches } = await run(rt, digit, count, w, steps, seed, 128);
    results.push({ w, matches, items });
  }
  return { digit, count, seconds: secondsSince(start), results };
}

async function experiment(rt: Runtime, body: Body) {
  const type = String(body.experiment_type ?? "guidance");
  if (!["guidance", "steps", "seeds"].includes(type)) {
    throw new LocalApiError(422, "experiment_type must be guidance, steps or seeds.");
  }
  const digit = int(body, "digit", 7, 0, 9);
  const count = int(body, "count", 4, 1, 8);
  const baseW = num(body, "w", 1.2, 0, 3);
  const baseSteps = int(body, "steps", 30, 5, 100);
  const baseSeed = int(body, "seed", 42, 0, 2 ** 31 - 1);
  let values = list(body, "values", [], 2, 4);
  if (type === "guidance") values = values.map((v) => Math.min(Math.max(v, 0), 3));
  else if (type === "steps") values = values.map((v) => Math.min(Math.max(Math.trunc(v), 5), 100));
  else values = values.map((v) => Math.min(Math.max(Math.trunc(v), 0), 2 ** 31 - 1));

  const totalStart = now();
  const results = [];
  for (const value of values) {
    const w = type === "guidance" ? value : baseW;
    const steps = type === "steps" ? value : baseSteps;
    const seed = type === "seeds" ? value : baseSeed;
    const start = now();
    const { items, matches } = await run(rt, digit, count, w, steps, seed, 128);
    const avg = items.reduce((s, it) => s + it.conf, 0) / items.length;
    results.push({
      parameter: type === "guidance" ? "guidance" : type === "steps" ? "steps" : "seed",
      value,
      matches,
      match_rate: round((matches / items.length) * 100, 2),
      avg_confidence: round(avg, 3),
      seconds: secondsSince(start),
      items,
    });
  }
  return { experiment_type: type, digit, count, total_seconds: secondsSince(totalStart), results };
}

async function interpolate(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const w = num(body, "w", 1.2, 0, 3);
  const steps = int(body, "steps", 30, 5, 100);
  const seedA = int(body, "seed_a", 42, 0, 2 ** 31 - 1);
  const seedB = int(body, "seed_b", 123, 0, 2 ** 31 - 1);
  const frames = int(body, "frames", 9, 3, 15);
  const start = now();
  const noiseA = randn(P, makeRng(seedA));
  const noiseB = randn(P, makeRng(seedB));
  const results = [];
  for (let f = 0; f < frames; f++) {
    const alpha = frames === 1 ? 0 : f / (frames - 1);
    const noise = new Float32Array(P);
    for (let j = 0; j < P; j++) noise[j] = (1 - alpha) * noiseA[j] + alpha * noiseB[j];
    const { images } = await sampleDdim(rt, { n: 1, labels: [digit], w, steps, noise });
    results.push({ alpha: round(alpha, 3), image: greyToPng(images, 32, 32) });
  }
  return { digit, frames: results, count: results.length, inference_time: secondsSince(start, 3) };
}

async function visualize(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const w = num(body, "w", 1.2, 0, 3);
  const steps = int(body, "steps", 20, 5, 100);
  const seed = int(body, "seed", 42, 0, 2 ** 31 - 1);
  const snapshotCount = int(body, "snapshot_count", 8, 4, 12);
  const start = now();
  const noise = randn(P, makeRng(seed));
  const ts = ddimTimesteps(rt.schedule.T, steps);
  const count = Math.min(snapshotCount, ts.length);
  const captureIdx = new Set<number>();
  for (let k = 0; k < count; k++) {
    captureIdx.add(Math.round(count === 1 ? 0 : (k * (ts.length - 1)) / (count - 1)));
  }
  const snapshots = [{ timestep: ts[0], image: greyToPng(clampCopy(noise), 32, 32, 192) }];
  await sampleDdim(rt, {
    n: 1,
    labels: [digit],
    w,
    steps,
    noise,
    timesteps: ts,
    onStep: ({ i, tPrev, x }) => {
      if (captureIdx.has(i)) snapshots.push({ timestep: tPrev, image: greyToPng(clampCopy(x), 32, 32, 192) });
    },
  });
  return { digit, steps, guidance: w, seed, seconds: secondsSince(start), snapshots };
}

function clampCopy(x: Float32Array) {
  const out = new Float32Array(x.length);
  for (let j = 0; j < x.length; j++) out[j] = x[j] < -1 ? -1 : x[j] > 1 ? 1 : x[j];
  return out;
}

function stats(values: ArrayLike<number>) {
  let mean = 0, min = Infinity, max = -Infinity;
  for (let j = 0; j < values.length; j++) {
    mean += values[j];
    if (values[j] < min) min = values[j];
    if (values[j] > max) max = values[j];
  }
  mean /= values.length;
  let v = 0;
  for (let j = 0; j < values.length; j++) v += (values[j] - mean) ** 2;
  const std = Math.sqrt(v / Math.max(1, values.length - 1)); // unbiased, like torch.std
  return { mean, std, min, max };
}

function nearest(ts: number[], target: number) {
  return ts.reduce((best, v) => (Math.abs(v - target) < Math.abs(best - target) ? v : best), ts[0]);
}

async function representation(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const timestep = int(body, "timestep", 200, 0, 399);
  const steps = int(body, "steps", 50, 5, 100);
  const w = num(body, "w", 1.2, 0, 3);
  const seed = int(body, "seed", 42, 0, 2 ** 31 - 1);
  const start = now();
  const ts = ddimTimesteps(rt.schedule.T, steps);
  const actual = nearest(ts, timestep);
  const { captured } = await sampleDdim(rt, {
    n: 1,
    labels: [digit],
    w,
    steps,
    noise: randn(P, makeRng(seed)),
    timesteps: ts,
    captureAt: new Set([actual]),
  });
  const feat = captured.get(actual);
  if (!feat) throw new LocalApiError(500, "Bottleneck feature capture failed.");
  const [, channels, h, wd] = feat.shape;
  const heatmaps = [];
  for (let c = 0; c < channels; c++) {
    const map = resizeBilinear(feat.data.subarray(c * h * wd, (c + 1) * h * wd), h, wd, 64, 64);
    const s = stats(map);
    const norm = new Float32Array(map.length);
    for (let j = 0; j < map.length; j++) norm[j] = ((map[j] - s.min) / (s.max - s.min + 1e-8)) * 2 - 1;
    heatmaps.push({ channel: c, image: greyToPng(norm, 64, 64, 192) });
  }
  const all = stats(feat.data.subarray(0, channels * h * wd));
  return {
    digit,
    requested_timestep: timestep,
    actual_timestep: actual,
    steps,
    channels,
    feature_shape: [channels, h, wd],
    statistics: { mean: all.mean, std: all.std, min: all.min, max: all.max },
    heatmaps,
    seconds: secondsSince(start, 3),
  };
}

async function representationEvolution(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const requested = list(body, "timesteps", [350, 250, 150, 50], 2, 6);
  const steps = int(body, "steps", 50, 5, 100);
  const w = num(body, "w", 1.2, 0, 3);
  const seed = int(body, "seed", 42, 0, 2 ** 31 - 1);
  const channel = int(body, "channel", 0, 0, 255);
  const T = rt.schedule.T;
  if (requested.some((t) => t < 0 || t >= T)) {
    throw new LocalApiError(400, `Timesteps must be between 0 and ${T - 1}.`);
  }
  const start = now();
  const ts = ddimTimesteps(T, steps);
  const actual = Array.from(new Set(requested.map((r) => nearest(ts, r)))).sort((a, b) => b - a);
  const { captured } = await sampleDdim(rt, {
    n: 1,
    labels: [digit],
    w,
    steps,
    noise: randn(P, makeRng(seed)),
    timesteps: ts,
    captureAt: new Set(actual),
  });
  if (actual.some((t) => !captured.has(t))) {
    throw new LocalApiError(500, "One or more timestep features could not be captured.");
  }
  const first = captured.get(actual[0])!;
  const [, channels, h, wd] = first.shape;
  if (channel >= channels) throw new LocalApiError(422, `channel must be below ${channels}.`);
  const maps = actual.map((t) => {
    const f = captured.get(t)!;
    return resizeBilinear(f.data.subarray(channel * h * wd, (channel + 1) * h * wd), h, wd, 64, 64);
  });
  let gMin = Infinity, gMax = -Infinity;
  for (const m of maps) {
    const s = stats(m);
    gMin = Math.min(gMin, s.min);
    gMax = Math.max(gMax, s.max);
  }
  const evolution = actual.map((t, k) => {
    const m = maps[k];
    const norm = new Float32Array(m.length);
    for (let j = 0; j < m.length; j++) norm[j] = ((m[j] - gMin) / (gMax - gMin + 1e-8)) * 2 - 1;
    const s = stats(m);
    return { timestep: t, image: greyToPng(norm, 64, 64, 192), mean: s.mean, std: s.std };
  });
  return {
    digit,
    channel,
    requested_timesteps: requested,
    actual_timesteps: actual,
    steps,
    seed,
    guidance: w,
    feature_shape: [channels, h, wd],
    activation_range: { min: gMin, max: gMax },
    evolution,
    seconds: secondsSince(start, 3),
  };
}

async function evaluate(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const count = int(body, "count", 32, 2, 64);
  const w = num(body, "w", 1.2, 0, 3);
  const steps = int(body, "steps", 20, 5, 100);
  const seed = int(body, "seed", 42, 0, 2 ** 31 - 1);
  const start = now();
  const { items, matches, images, probs } = await run(rt, digit, count, w, steps, seed, 128);

  // Mean pairwise L1 distance / pixels (torch.pdist(p=1).mean() / D)
  let pairSum = 0, pairs = 0;
  for (let a = 0; a < count; a++) {
    for (let b = a + 1; b < count; b++) {
      let d = 0;
      for (let j = 0; j < P; j++) d += Math.abs(images[a * P + j] - images[b * P + j]);
      pairSum += d;
      pairs++;
    }
  }
  const diversity = pairs ? pairSum / pairs / P : 0;

  // Per-pixel std across the batch (unbiased), averaged over pixels
  let stdSum = 0;
  for (let j = 0; j < P; j++) {
    let m = 0;
    for (let k = 0; k < count; k++) m += images[k * P + j];
    m /= count;
    let v = 0;
    for (let k = 0; k < count; k++) v += (images[k * P + j] - m) ** 2;
    stdSum += Math.sqrt(v / (count - 1));
  }
  const meanConf = probs.reduce((s, row) => s + Math.max(...row), 0) / count;
  const elapsed = (now() - start) / 1000;

  return {
    digit,
    count,
    matches,
    class_consistency: round((matches / count) * 100, 2),
    mean_confidence: round(meanConf, 4),
    diversity_mean_pairwise_l1: round(diversity, 4),
    mean_pixel_std: round(stdSum / P, 4),
    seconds: round(elapsed, 2),
    images_per_second: elapsed > 0 ? round(count / elapsed, 2) : 0,
    items,
    params: { digit, count, w, steps, seed },
  };
}

function lockInStep(points: { step: number; match_rate: number }[], threshold = 0.9) {
  let lock: number | null = null;
  for (let k = points.length - 1; k >= 0; k--) {
    if (points[k].match_rate >= threshold) lock = points[k].step;
    else break;
  }
  return lock;
}

async function recognition(rt: Runtime, body: Body) {
  const digit = int(body, "digit", 7, 0, 9);
  const ws = list(body, "ws", [0, 1.2, 3], 1, 4);
  if (ws.some((w) => w < 0 || w > 3)) throw new LocalApiError(422, "Each w must be between 0 and 3.");
  const steps = int(body, "steps", 30, 5, 100);
  const count = int(body, "count", 8, 1, 16);
  const seed = int(body, "seed", 42, 0, 2 ** 31 - 1);
  const start = now();
  const ts = partialTimesteps(rt.schedule.T - 1, steps);
  const previewCount = Math.min(6, ts.length);
  const previewIdx = new Set<number>();
  for (let k = 0; k < previewCount; k++) {
    previewIdx.add(Math.round(previewCount === 1 ? 0 : (k * (ts.length - 1)) / (previewCount - 1)));
  }

  const series = [];
  for (const w of ws) {
    const noise = randn(count * P, makeRng(seed));
    const points: { step: number; timestep: number; target_prob: number; match_rate: number }[] = [];
    const previews: { step: number; timestep: number; image: string; pred: number; conf: number }[] = [];
    const { images } = await sampleDdim(rt, {
      n: count,
      labels: new Array(count).fill(digit),
      w,
      steps,
      noise,
      timesteps: ts,
      onStep: async ({ i, t, x0 }) => {
        const probs = await classify(rt, x0, count);
        const preds = probs.map(argmax);
        points.push({
          step: i + 1,
          timestep: t,
          target_prob: round(probs.reduce((s, r) => s + r[digit], 0) / count, 4),
          match_rate: round(preds.filter((p) => p === digit).length / count, 4),
        });
        if (previewIdx.has(i)) {
          previews.push({
            step: i + 1,
            timestep: t,
            image: greyToPng(x0.subarray(0, P), 32, 32, 128),
            pred: preds[0],
            conf: round(Math.max(...probs[0]), 3),
          });
        }
      },
    });
    const finalProbs = await classify(rt, images, count);
    const finalMatches = finalProbs.filter((r) => argmax(r) === digit).length;
    series.push({ w, points, lock_in_step: lockInStep(points), final_matches: finalMatches, previews });
  }

  return { digit, steps: series[0].points.length, count, seed, seconds: secondsSince(start), series };
}

async function sketch(rt: Runtime, body: Body) {
  const image = String(body.image ?? "");
  if (image.length < 20) throw new LocalApiError(422, "image is required.");
  const digitRaw = body.digit;
  const digit = digitRaw === null || digitRaw === undefined ? null : int(body, "digit", 7, 0, 9);
  const strength = num(body, "strength", 0.5, 0.1, 1);
  const w = num(body, "w", 1.2, 0, 3);
  const steps = int(body, "steps", 50, 5, 100);
  const count = int(body, "count", 4, 1, 8);
  const seed = int(body, "seed", 42, 0, 2 ** 31 - 1);

  let sketchImg: Float32Array;
  try {
    sketchImg = await decodeSketch(image);
  } catch (error) {
    throw new LocalApiError(400, error instanceof Error ? error.message : "Could not read the sketch.");
  }

  const start = now();
  const T = rt.schedule.T;
  const tStart = Math.max(1, Math.round(strength * (T - 1)));
  const usedSteps = Math.max(2, Math.round(steps * strength));
  const ts = partialTimesteps(tStart, usedSteps);

  // q_sample: x_t = sqrt(ab) * x0 + sqrt(1 - ab) * noise
  const rng = makeRng(seed);
  const noise = randn(count * P, rng);
  const ab = rt.schedule.alpha_bar[tStart];
  const x = new Float32Array(count * P);
  for (let k = 0; k < count; k++) {
    for (let j = 0; j < P; j++) {
      x[k * P + j] = Math.sqrt(ab) * sketchImg[j] + Math.sqrt(1 - ab) * noise[k * P + j];
    }
  }
  const noised = clampCopy(x.subarray(0, P));

  const labels = digit === null ? null : new Array(count).fill(digit);
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i];
    const tPrev = i + 1 < ts.length ? ts[i + 1] : -1;
    const r = await guidedEps(rt, x, count, t, labels, w);
    ddimStep(x, r.eps, rt.schedule.alpha_bar, t, tPrev);
  }
  const out = clampCopy(x);
  const probs = await classify(rt, out, count);
  const items = probs.map((row, k) => {
    const pred = argmax(row);
    return { image: greyToPng(imageAt(out, k), 32, 32, 192), pred, conf: round(row[pred], 3) };
  });
  const sketchProbs = (await classify(rt, sketchImg, 1))[0];
  const sketchPred = argmax(sketchProbs);

  return {
    input: greyToPng(sketchImg, 32, 32, 192),
    noised: greyToPng(noised, 32, 32, 192),
    sketch_pred: sketchPred,
    sketch_conf: round(sketchProbs[sketchPred], 3),
    items,
    matches: digit === null ? null : items.filter((it) => it.pred === digit).length,
    t_start: tStart,
    steps_used: ts.length,
    seconds: secondsSince(start),
    params: { digit, strength, w, steps, count, seed },
  };
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const HANDLERS: Record<string, (rt: Runtime, body: Body) => Promise<unknown>> = {
  "/api/generate": generate,
  "/api/compare": compare,
  "/api/experiment": experiment,
  "/api/interpolate": interpolate,
  "/api/visualize": visualize,
  "/api/representation": representation,
  "/api/representation/evolution": representationEvolution,
  "/api/evaluate": evaluate,
  "/api/recognition": recognition,
  "/api/sketch": sketch,
};

// One request at a time, like the server's lock
let queue: Promise<unknown> = Promise.resolve();

export function handleLocally(path: string, body: Body): Promise<unknown> {
  const handler = HANDLERS[path];
  if (!handler) return Promise.reject(new LocalApiError(404, `Unknown endpoint ${path}`));
  const job = queue.then(async () => {
    const rt = await getRuntime();
    return handler(rt, body);
  });
  queue = job.catch(() => undefined);
  return job;
}

export function localInfo(threads: number) {
  return {
    status: "online",
    model: "Class-Conditional MNIST Diffusion",
    img_size: 32,
    device: `browser (${threads} thread${threads === 1 ? "" : "s"})`,
    classes: 10,
  };
}
