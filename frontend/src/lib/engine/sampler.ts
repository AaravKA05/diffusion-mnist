/*
 * The diffusion maths, ported from sample_cond.py and lab_features.py.
 * Images are Float32Arrays of size n * 32 * 32 with values in [-1, 1].
 */

import type { Runtime } from "./runtime";

export const NULL_LABEL = 10;

// ---------------------------------------------------------------------------
// Seeded random numbers (the browser cannot reproduce torch.manual_seed, so the
// same seed gives different images than the laptop server, but the same seed
// always gives the same images in the browser)
// ---------------------------------------------------------------------------

export function makeRng(seed: number) {
  let a = (seed >>> 0) ^ 0x9e3779b9;
  const uniform = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | null = null;
  const normal = () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    while (u === 0) u = uniform();
    const v = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return { uniform, normal };
}

export function randn(size: number, rng: ReturnType<typeof makeRng>) {
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) out[i] = rng.normal();
  return out;
}

// ---------------------------------------------------------------------------
// Timesteps
// ---------------------------------------------------------------------------

/** Same as list(reversed(torch.linspace(0, T - 1, steps).long())). */
export function ddimTimesteps(T: number, steps: number) {
  const ts: number[] = [];
  for (let i = 0; i < steps; i++) {
    ts.push(steps === 1 ? 0 : Math.floor((i * (T - 1)) / (steps - 1) + 1e-6));
  }
  return ts.reverse();
}

/** Same as lab_features.ddim_timesteps: evenly spaced, no duplicates. */
export function partialTimesteps(tStart: number, steps: number) {
  const set = new Set<number>();
  for (let i = 0; i < steps; i++) {
    set.add(steps === 1 ? 0 : Math.round((i * tStart) / (steps - 1)));
  }
  return Array.from(set).sort((a, b) => b - a);
}

// ---------------------------------------------------------------------------
// Model calls
// ---------------------------------------------------------------------------

const P = 32 * 32;

type EpsResult = { eps: Float32Array; bottleneck?: Float32Array; bottleneckShape?: number[] };

/**
 * Classifier-free guided noise prediction for a batch of n images.
 * labels === null means "no label" (the model picks the digit).
 * The conditional and unconditional passes go through the model as one batch.
 */
export async function guidedEps(
  rt: Runtime,
  x: Float32Array,
  n: number,
  t: number,
  labels: number[] | null,
  w: number,
  captureBottleneck = false
): Promise<EpsResult> {
  const { ort } = rt;
  const useNullOnly = labels === null;
  const twoPasses = !useNullOnly && w !== 1;
  const batch = twoPasses ? 2 * n : n;

  const xs = new Float32Array(batch * P);
  xs.set(x, 0);
  if (twoPasses) xs.set(x, n * P);

  const ts = new BigInt64Array(batch);
  const ys = new BigInt64Array(batch);
  for (let k = 0; k < batch; k++) {
    ts[k] = BigInt(t);
    const conditional = !useNullOnly && k < n;
    ys[k] = BigInt(conditional ? (labels as number[])[k] : NULL_LABEL);
  }

  const feeds = {
    x: new ort.Tensor("float32", xs, [batch, 1, 32, 32]),
    t: new ort.Tensor("int64", ts, [batch]),
    y: new ort.Tensor("int64", ys, [batch]),
  };

  const outputs = await rt.unet.run(feeds, captureBottleneck ? ["eps", "bottleneck"] : ["eps"]);
  const raw = outputs.eps.data as Float32Array;

  let eps: Float32Array;
  if (twoPasses) {
    eps = new Float32Array(n * P);
    for (let j = 0; j < n * P; j++) {
      const u = raw[n * P + j];
      eps[j] = u + w * (raw[j] - u);
    }
  } else {
    eps = raw.slice(0, n * P);
  }

  if (!captureBottleneck) return { eps };

  // Conditional pass comes first, so the first n feature maps belong to it
  const b = outputs.bottleneck;
  const [, c, h, wd] = b.dims as number[];
  const perImage = c * h * wd;
  return {
    eps,
    bottleneck: (b.data as Float32Array).slice(0, n * perImage),
    bottleneckShape: [n, c, h, wd],
  };
}

/** One deterministic DDIM update, in place. Returns the clean-image guess x0. */
export function ddimStep(
  x: Float32Array,
  eps: Float32Array,
  alphaBar: number[],
  t: number,
  tPrev: number
) {
  const ab = alphaBar[t];
  const abPrev = tPrev >= 0 ? alphaBar[tPrev] : 1;
  const sa = Math.sqrt(ab);
  const s1a = Math.sqrt(1 - ab);
  const sp = Math.sqrt(abPrev);
  const s1p = Math.sqrt(1 - abPrev);
  const x0 = new Float32Array(x.length);
  for (let j = 0; j < x.length; j++) {
    let v = (x[j] - s1a * eps[j]) / sa;
    v = v < -1 ? -1 : v > 1 ? 1 : v;
    x0[j] = v;
    x[j] = sp * v + s1p * eps[j];
  }
  return x0;
}

export function clamp(x: Float32Array) {
  const out = new Float32Array(x.length);
  for (let j = 0; j < x.length; j++) out[j] = x[j] < -1 ? -1 : x[j] > 1 ? 1 : x[j];
  return out;
}

/** Softmax probabilities, shape n x 10. */
export async function classify(rt: Runtime, images: Float32Array, n: number) {
  // Copy: the model runs in a worker and takes ownership of the buffer it is
  // given, which would leave the caller's array empty ("detached")
  const out = await rt.clf.run({
    x: new rt.ort.Tensor("float32", images.slice(0, n * P), [n, 1, 32, 32]),
  });
  const logits = out.logits.data as Float32Array;
  const probs: number[][] = [];
  for (let k = 0; k < n; k++) {
    const row = Array.from(logits.slice(k * 10, k * 10 + 10));
    const m = Math.max(...row);
    const e = row.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    probs.push(e.map((v) => v / s));
  }
  return probs;
}

export function argmax(row: number[]) {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

export type SampleOptions = {
  n: number;
  labels: number[] | null;
  w: number;
  steps: number;
  noise: Float32Array;
  /** Called after each step with the step index, timestep, x (after step) and x0 guess */
  onStep?: (info: { i: number; t: number; tPrev: number; x: Float32Array; x0: Float32Array }) => void | Promise<void>;
  /** Capture the conditional bottleneck at these timesteps */
  captureAt?: Set<number>;
  timesteps?: number[];
};

/** Port of cond_ddim_sample. Returns final images in [-1, 1] and any captured features. */
export async function sampleDdim(rt: Runtime, opts: SampleOptions) {
  const { n, labels, w, steps, onStep, captureAt } = opts;
  const ts = opts.timesteps ?? ddimTimesteps(rt.schedule.T, steps);
  const x = Float32Array.from(opts.noise);
  const captured = new Map<number, { data: Float32Array; shape: number[] }>();

  for (let i = 0; i < ts.length; i++) {
    const t = ts[i];
    const tPrev = i + 1 < ts.length ? ts[i + 1] : -1;
    const capture = captureAt?.has(t) ?? false;
    const r = await guidedEps(rt, x, n, t, labels, w, capture);
    if (capture && r.bottleneck && r.bottleneckShape) {
      captured.set(t, { data: r.bottleneck, shape: r.bottleneckShape });
    }
    const x0 = ddimStep(x, r.eps, rt.schedule.alpha_bar, t, tPrev);
    if (onStep) await onStep({ i, t, tPrev, x, x0 });
  }

  return { images: clamp(x), captured, timesteps: ts };
}

export function imageAt(images: Float32Array, index: number) {
  return images.subarray(index * P, (index + 1) * P);
}
