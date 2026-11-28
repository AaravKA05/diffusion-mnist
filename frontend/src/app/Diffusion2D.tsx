"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Pause, Play, Shuffle } from "lucide-react";

/**
 * 2D playground: the same DDPM recipe as the MNIST model, on toy point clouds.
 * Weights come from toy2d.py (public/models/toy2d/<shape>.json, loaded when a
 * shape is first picked). The MLP is tiny (3 -> 128 -> 128 -> 2), so it runs
 * in plain TypeScript on the main thread, a few steps per frame. No backend or
 * ONNX runtime needed.
 */

type Layer = { in: number; out: number; W: Float32Array; b: Float32Array };
type Quality = { sample_to_data: number; data_to_sample: number; modes_covered?: number; within_3std?: number };
type SamplerKey = "ddpm_100" | "ddim_20" | "ddim_10";
type Schedule = {
    name: string;
    alphaBarT: number;
    quality: Record<SamplerKey, Quality>;
    layers: Layer[];
    betas: Float64Array;
    alphas: Float64Array;
    alphaBar: Float64Array;
};
type Toy = {
    T: number;
    trainSteps: number;
    x0Clip: number;
    data: Float32Array;
    reference: Quality;
    schedules: Schedule[];
};

type RawLayer = { in: number; out: number; W: string; b: string };
type RawSchedule = { betas: string; alpha_bar_T: number; layers: RawLayer[] } & Record<SamplerKey, Quality>;

const N = 2000;
const VIEW = 3.2; // half-width of the plotted square
const STEPS_PER_FRAME = 4;

const DATASETS = [
    { id: "swiss_roll", label: "Swiss roll" },
    { id: "moons", label: "Two moons" },
    { id: "gaussians8", label: "8 Gaussians" },
] as const;
type DatasetId = (typeof DATASETS)[number]["id"];

const SAMPLERS: { id: SamplerKey; label: string; steps: number }[] = [
    { id: "ddpm_100", label: "DDPM · 100", steps: 100 },
    { id: "ddim_20", label: "DDIM · 20", steps: 20 },
    { id: "ddim_10", label: "DDIM · 10", steps: 10 },
];

const SCHEDULE_INFO: Record<string, { formula: string; note: string }> = {
    gentle: {
        formula: "linear, β_T = 0.02",
        note: "Never reaches pure noise: 60% of the signal is left at t = 100. Same β range as the MNIST model, which keeps 13% at t = 400.",
    },
    linear: { formula: "linear, β_T = 0.2", note: "Signal reaches about zero right at the last step." },
    aggressive: {
        formula: "linear, β_T = 0.5",
        note: "Signal is gone by about t = 50, so half the steps do nothing useful.",
    },
    cosine: {
        formula: "cosine, s = 0.008",
        note: "Signal falls off evenly across all steps and still reaches noise (Nichol & Dhariwal, 2021).",
    },
};

function decode(b64: string): Float32Array {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
}

function scheduleFromBetas(b32: Float32Array) {
    const T = b32.length;
    const betas = new Float64Array(T);
    const alphas = new Float64Array(T);
    const alphaBar = new Float64Array(T);
    let ab = 1;
    for (let i = 0; i < T; i++) {
        betas[i] = b32[i];
        alphas[i] = 1 - betas[i];
        ab *= alphas[i];
        alphaBar[i] = ab;
    }
    return { betas, alphas, alphaBar };
}

/** Frames DDIM visits with `steps` model calls, from T down to 0 (matches toy2d.ddim_frames). */
function ddimFrames(T: number, steps: number) {
    return Array.from({ length: steps + 1 }, (_, i) => Math.round(T - (i * T) / steps));
}

/** Seeded normal samples (mulberry32 + Box-Muller). */
function gaussian(seed: number) {
    let s = seed >>> 0;
    const uniform = () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let r = Math.imul(s ^ (s >>> 15), 1 | s);
        r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
    let spare: number | null = null;
    return () => {
        if (spare !== null) {
            const v = spare;
            spare = null;
            return v;
        }
        const u = Math.max(uniform(), 1e-12);
        const v = uniform();
        const m = Math.sqrt(-2 * Math.log(u));
        spare = m * Math.sin(2 * Math.PI * v);
        return m * Math.cos(2 * Math.PI * v);
    };
}

/** eps_theta for every point at timestep t (0-based). */
function predictNoise(layers: Layer[], x: Float32Array, t: number, T: number, out: Float32Array) {
    const hidden = layers[0].out;
    const h1 = new Float32Array(hidden);
    const h2 = new Float32Array(hidden);
    const [l1, l2, l3] = layers;
    const tt = t / T;
    for (let p = 0; p < N; p++) {
        const x0 = x[2 * p];
        const x1 = x[2 * p + 1];
        for (let j = 0; j < hidden; j++) {
            const v = l1.W[3 * j] * x0 + l1.W[3 * j + 1] * x1 + l1.W[3 * j + 2] * tt + l1.b[j];
            h1[j] = v > 0 ? v : 0;
        }
        for (let j = 0; j < hidden; j++) {
            let v = l2.b[j];
            const row = j * hidden;
            for (let k = 0; k < hidden; k++) v += l2.W[row + k] * h1[k];
            h2[j] = v > 0 ? v : 0;
        }
        for (let j = 0; j < 2; j++) {
            let v = l3.b[j];
            const row = j * hidden;
            for (let k = 0; k < hidden; k++) v += l3.W[row + k] * h2[k];
            out[2 * p + j] = v;
        }
    }
}

function Segmented<V extends string>({
    label,
    options,
    value,
    onChange,
}: {
    label: string;
    options: readonly { id: V; label: string }[];
    value: V;
    onChange: (v: V) => void;
}) {
    return (
        <div>
            <p className="mb-1.5 text-xs text-slate-400">{label}</p>
            <div role="radiogroup" aria-label={label} className="grid w-full grid-cols-3 rounded-xl border border-white/10 p-1 sm:inline-flex sm:w-auto">
                {options.map((o) => (
                    <button
                        key={o.id}
                        role="radio"
                        aria-checked={value === o.id}
                        onClick={() => onChange(o.id)}
                        className={`min-h-11 rounded-lg px-2 text-sm leading-tight transition sm:px-3 ${value === o.id ? "bg-white/10 font-semibold text-white" : "text-slate-300 hover:bg-white/5"}`}
                    >
                        {o.label}
                    </button>
                ))}
            </div>
        </div>
    );
}

export default function Diffusion2D() {
    const [toys, setToys] = useState<Partial<Record<DatasetId, Toy>>>({});
    const [loadError, setLoadError] = useState<string | null>(null);
    const [dataset, setDataset] = useState<DatasetId>("swiss_roll");
    const [scheduleName, setScheduleName] = useState("cosine");
    const [sampler, setSampler] = useState<SamplerKey>("ddpm_100");
    const [mode, setMode] = useState<"forward" | "reverse">("reverse");
    const [t, setT] = useState(100);
    const [seed, setSeed] = useState(1);
    const [showData, setShowData] = useState(false);
    const [playing, setPlaying] = useState(false);
    // frames[k] = points at timestep k (k = T is the starting noise). With
    // DDIM, skipped timesteps hold the last state reached at or above them.
    const [frames, setFrames] = useState<(Float32Array | null)[]>([]);
    const [calls, setCalls] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const toy = toys[dataset] ?? null;

    useEffect(() => {
        if (toys[dataset]) return;
        let cancelled = false;
        fetch(`/models/toy2d/${dataset}.json`)
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then((raw) => {
                if (cancelled) return;
                const schedules = Object.entries(raw.schedules as Record<string, RawSchedule>).map(([name, s]) => ({
                    name,
                    alphaBarT: s.alpha_bar_T,
                    quality: { ddpm_100: s.ddpm_100, ddim_20: s.ddim_20, ddim_10: s.ddim_10 },
                    layers: s.layers.map((l) => ({ in: l.in, out: l.out, W: decode(l.W), b: decode(l.b) })),
                    ...scheduleFromBetas(decode(s.betas)),
                }));
                setToys((prev) => ({
                    ...prev,
                    [dataset]: {
                        T: raw.T,
                        trainSteps: raw.trainSteps,
                        x0Clip: raw.x0Clip,
                        data: decode(raw.data),
                        reference: raw.reference,
                        schedules,
                    },
                }));
            })
            .catch((err) => {
                if (!cancelled)
                    setLoadError(`Could not load the 2D model: ${err instanceof Error ? err.message : "unknown error"}.`);
            });
        return () => {
            cancelled = true;
        };
    }, [dataset, toys]);

    const schedule = toy?.schedules.find((s) => s.name === scheduleName) ?? null;
    const T = toy?.T ?? 100;
    const samplerSteps = SAMPLERS.find((s) => s.id === sampler)!.steps;

    // Fixed noise for the forward view, so the slider only changes t.
    const forwardNoise = useMemo(() => {
        const g = gaussian(seed * 2654435761);
        return Float32Array.from({ length: 2 * N }, g);
    }, [seed]);

    // Reverse process: run the sampler a few steps per animation frame.
    useEffect(() => {
        if (!toy || !schedule) return;
        let cancelled = false;
        let raf = 0;
        const g = gaussian(seed);
        const all: (Float32Array | null)[] = Array(T + 1).fill(null);
        let x: Float32Array = Float32Array.from({ length: 2 * N }, g);
        all[T] = x;
        const eps = new Float32Array(2 * N);
        const { betas, alphas, alphaBar } = schedule;
        const path = sampler === "ddpm_100" ? ddimFrames(T, T) : ddimFrames(T, samplerSteps);
        let i = 0; // index into path
        let done = 0;

        const ddpmStep = (k: number) => {
            const step = k - 1;
            predictNoise(schedule.layers, x, step, T, eps);
            const coef = betas[step] / Math.sqrt(1 - alphaBar[step]);
            const sa = Math.sqrt(alphas[step]);
            const sb = Math.sqrt(betas[step]);
            const next = new Float32Array(2 * N);
            for (let j = 0; j < 2 * N; j++) next[j] = (x[j] - coef * eps[j]) / sa + (step > 0 ? sb * g() : 0);
            return next;
        };
        const ddimStep = (k: number, kNext: number) => {
            predictNoise(schedule.layers, x, k - 1, T, eps);
            const ab = alphaBar[k - 1];
            const abNext = kNext > 0 ? alphaBar[kNext - 1] : 1;
            const s = Math.sqrt(1 - ab);
            const sa = Math.sqrt(ab);
            const sn = Math.sqrt(abNext);
            const nn = Math.sqrt(1 - abNext);
            const clip = toy.x0Clip;
            const next = new Float32Array(2 * N);
            for (let j = 0; j < 2 * N; j++) {
                const x0 = Math.min(clip, Math.max(-clip, (x[j] - s * eps[j]) / sa));
                next[j] = sn * x0 + nn * eps[j];
            }
            return next;
        };

        const tick = () => {
            if (cancelled) return;
            for (let n = 0; n < STEPS_PER_FRAME && i < path.length - 1; n++, i++) {
                const k = path[i];
                const kNext = path[i + 1];
                const prev = x;
                x = sampler === "ddpm_100" ? ddpmStep(k) : ddimStep(k, kNext);
                for (let idx = kNext + 1; idx < k; idx++) all[idx] = prev;
                all[kNext] = x;
                done++;
            }
            setFrames(all.slice());
            setCalls(done);
            if (i < path.length - 1) raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => {
            cancelled = true;
            cancelAnimationFrame(raf);
        };
    }, [toy, schedule, seed, T, sampler, samplerSteps]);

    const points = useMemo(() => {
        if (!toy || !schedule) return null;
        if (mode === "reverse") return frames[t] ?? null;
        if (t === 0) return toy.data;
        const ab = schedule.alphaBar[t - 1];
        const s = Math.sqrt(ab);
        const n = Math.sqrt(1 - ab);
        const out = new Float32Array(2 * N);
        for (let i = 0; i < 2 * N; i++) out[i] = s * toy.data[i] + n * forwardNoise[i];
        return out;
    }, [toy, schedule, mode, t, frames, forwardNoise]);

    // Redraw on resize
    const [resizeTick, setResizeTick] = useState(0);
    useEffect(() => {
        const onResize = () => setResizeTick((n) => n + 1);
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

    // Draw
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const size = canvas.clientWidth;
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(size * dpr)) {
            canvas.width = Math.round(size * dpr);
            canvas.height = Math.round(size * dpr);
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, size, size);
        const px = (v: number) => ((v + VIEW) / (2 * VIEW)) * size;
        const py = (v: number) => size - ((v + VIEW) / (2 * VIEW)) * size;

        ctx.strokeStyle = "rgba(243,236,224,0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px(0), 0);
        ctx.lineTo(px(0), size);
        ctx.moveTo(0, py(0));
        ctx.lineTo(size, py(0));
        ctx.stroke();

        const r = Math.max(1.2, size / 420);
        const dots = (arr: Float32Array, color: string) => {
            ctx.fillStyle = color;
            for (let i = 0; i < N; i++) {
                ctx.fillRect(px(arr[2 * i]) - r / 2, py(arr[2 * i + 1]) - r / 2, r, r);
            }
        };
        if (showData && toy) dots(toy.data, "rgba(132,127,120,0.45)");
        if (points) dots(points, t === 0 ? "rgba(255,90,74,0.85)" : "rgba(243,236,224,0.7)");
    }, [points, showData, toy, t, resizeTick]);

    // Playback: reverse goes 100 -> 0, forward goes 0 -> 100
    useEffect(() => {
        if (!playing) return;
        const target = mode === "reverse" ? 0 : T;
        if (t === target) return;
        const nextT = mode === "reverse" ? t - 1 : t + 1;
        if (mode === "reverse" && !frames[nextT]) return; // wait for the sampler
        const id = window.setTimeout(() => {
            setT(nextT);
            if (nextT === target) setPlaying(false);
        }, 40);
        return () => window.clearTimeout(id);
    }, [playing, t, mode, T, frames]);

    const play = useCallback(() => {
        const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        const start = mode === "reverse" ? T : 0;
        const end = mode === "reverse" ? 0 : T;
        if (calm) {
            setT(end);
            return;
        }
        if (t === end) setT(start);
        setPlaying(true);
    }, [mode, T, t]);

    function restart() {
        setPlaying(false);
        setT(mode === "reverse" ? T : 0);
    }

    function switchMode(next: "forward" | "reverse") {
        setPlaying(false);
        setMode(next);
        setT(next === "reverse" ? T : 0);
    }

    const signal = schedule && t > 0 ? Math.sqrt(schedule.alphaBar[t - 1]) : 1;
    const noise = schedule && t > 0 ? Math.sqrt(1 - schedule.alphaBar[t - 1]) : 0;
    const finished = frames[0] != null;
    const computing = toy !== null && frames.length > 0 && !finished;
    const waiting = mode === "reverse" && !frames[t];
    const samplerLabel = sampler === "ddpm_100" ? "DDPM" : "DDIM";

    const curve = (s: Schedule) =>
        Array.from(s.alphaBar, (ab, i) => `${(i + 1) * (240 / T)},${(1 - Math.sqrt(ab)) * 80 + 4}`).join(" ");

    if (loadError) {
        return (
            <p className="rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
                {loadError} Reload the page to retry.
            </p>
        );
    }

    return (
        <section className="space-y-6">
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
                {/* Plot */}
                <div className="rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                    <div className="flex flex-wrap gap-x-6 gap-y-4">
                        <Segmented
                            label="Shape"
                            options={DATASETS}
                            value={dataset}
                            onChange={(v) => {
                                restart();
                                setDataset(v);
                            }}
                        />
                        <Segmented
                            label="Sampler (denoising only)"
                            options={SAMPLERS}
                            value={sampler}
                            onChange={(v) => {
                                restart();
                                setSampler(v);
                            }}
                        />
                    </div>

                    <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
                        <div role="tablist" aria-label="Process direction" className="inline-flex rounded-xl border border-white/10 p-1">
                            {(["reverse", "forward"] as const).map((m) => (
                                <button
                                    key={m}
                                    role="tab"
                                    aria-selected={mode === m}
                                    onClick={() => switchMode(m)}
                                    className={`min-h-11 rounded-lg px-4 text-sm transition ${mode === m ? "bg-violet-500 font-semibold text-black" : "text-slate-300 hover:bg-white/5"}`}
                                >
                                    {m === "reverse" ? "Denoise (reverse)" : "Add noise (forward)"}
                                </button>
                            ))}
                        </div>
                        <label className="flex min-h-11 cursor-pointer items-center gap-2 text-sm text-slate-300">
                            <input
                                type="checkbox"
                                checked={showData}
                                onChange={(e) => setShowData(e.target.checked)}
                                className="h-4 w-4 accent-violet-500"
                            />
                            Show training data
                        </label>
                    </div>

                    <div className="relative mt-5 overflow-hidden rounded-xl border border-white/10 bg-black/30">
                        <canvas
                            ref={canvasRef}
                            role="img"
                            aria-label={`${N} points at timestep ${t} of ${T}, ${mode === "reverse" ? `denoising with ${samplerLabel}` : "adding noise"}, ${scheduleName} schedule`}
                            className="block aspect-square w-full"
                        />
                        {(!toy || waiting) && (
                            <p className="absolute inset-0 flex items-center justify-center text-sm text-slate-400">
                                {!toy ? "Loading the 2D model…" : `Sampling… ${calls}/${samplerSteps} steps`}
                            </p>
                        )}
                    </div>

                    <div className="mt-5">
                        <div className="mb-2 flex justify-between text-sm">
                            <label htmlFor="toy-t">Timestep t</label>
                            <span className="tabular-nums text-violet-300">
                                {t} / {T}
                            </span>
                        </div>
                        <input
                            id="toy-t"
                            type="range"
                            min="0"
                            max={T}
                            value={t}
                            onChange={(e) => {
                                setPlaying(false);
                                setT(Number(e.target.value));
                            }}
                            className="w-full accent-violet-500"
                        />
                        <div className="mt-1 flex justify-between text-xs text-slate-400">
                            <span>t = 0: data</span>
                            <span>t = {T}: noise</span>
                        </div>
                    </div>

                    <div className="mt-5 flex flex-wrap items-center gap-3">
                        <button
                            onClick={() => (playing ? setPlaying(false) : play())}
                            disabled={!toy}
                            className="flex min-h-11 items-center gap-2 rounded-xl bg-violet-500 px-5 font-semibold text-black transition enabled:hover:bg-violet-400 disabled:opacity-50"
                        >
                            {playing ? (
                                <Pause className="h-4 w-4" strokeWidth={2} aria-hidden="true" />
                            ) : (
                                <Play className="h-4 w-4" strokeWidth={2} aria-hidden="true" />
                            )}
                            {playing ? "Pause" : mode === "reverse" ? "Play denoising" : "Play noising"}
                        </button>
                        <button
                            onClick={() => {
                                restart();
                                setSeed((s) => s + 1);
                            }}
                            disabled={!toy}
                            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/10 px-4 text-sm hover:bg-white/5 disabled:opacity-50"
                        >
                            <Shuffle className="h-4 w-4" strokeWidth={1.75} aria-hidden="true" />
                            New noise
                        </button>
                        <p className="ml-auto text-xs tabular-nums text-slate-400">
                            Seed {seed} · {computing ? `sampling ${calls}/${samplerSteps}` : `${samplerSteps} model calls, in your browser`}
                        </p>
                    </div>

                    <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
                        <div className="rounded-xl border border-white/10 p-3">
                            <dt className="text-xs text-slate-400">Signal kept, √ᾱ</dt>
                            <dd className="mt-1 text-lg tabular-nums">{signal.toFixed(3)}</dd>
                        </div>
                        <div className="rounded-xl border border-white/10 p-3">
                            <dt className="text-xs text-slate-400">Noise, √(1 − ᾱ)</dt>
                            <dd className="mt-1 text-lg tabular-nums">{noise.toFixed(3)}</dd>
                        </div>
                    </dl>
                    {sampler !== "ddpm_100" && mode === "reverse" && (
                        <p className="mt-3 text-xs leading-5 text-slate-400">
                            DDIM skips timesteps: each call predicts the clean points, then jumps straight to the
                            next visited step with no new noise. The plot holds still between jumps.
                        </p>
                    )}
                </div>

                {/* Schedules + recipe */}
                <div className="space-y-6">
                    <div className="rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                        <h3 className="text-xl font-semibold">Noise schedule</h3>
                        <p className="mt-2 text-sm leading-6 text-slate-400">
                            Four models per shape, same MLP and data. Scores are for the sampler picked on the left.
                        </p>

                        <div role="radiogroup" aria-label="Noise schedule" className="mt-5 space-y-2">
                            {toy?.schedules.map((s) => {
                                const active = s.name === scheduleName;
                                const q = s.quality[sampler];
                                return (
                                    <button
                                        key={s.name}
                                        role="radio"
                                        aria-checked={active}
                                        onClick={() => {
                                            restart();
                                            setScheduleName(s.name);
                                        }}
                                        className={`w-full rounded-xl border p-3 text-left transition ${active ? "border-violet-500 bg-violet-500/10" : "border-white/10 hover:bg-white/5"}`}
                                    >
                                        <div className="flex items-baseline justify-between gap-2">
                                            <span className="font-medium capitalize">{s.name}</span>
                                            <span className="text-xs tabular-nums text-slate-400">
                                                {SCHEDULE_INFO[s.name]?.formula} · ᾱ_T = {s.alphaBarT.toPrecision(2)}
                                            </span>
                                        </div>
                                        <p className="mt-1 text-xs text-slate-400">{SCHEDULE_INFO[s.name]?.note}</p>
                                        <p className="mt-1 text-xs tabular-nums text-slate-300">
                                            Sample to data {q.sample_to_data.toFixed(3)}
                                            {q.modes_covered !== undefined && ` · ${q.modes_covered}/8 modes`}
                                            <span className="text-slate-500">
                                                {" "}
                                                (real data: {toy.reference.sample_to_data.toFixed(3)})
                                            </span>
                                        </p>
                                    </button>
                                );
                            })}
                        </div>

                        {toy && (
                            <figure className="mt-5">
                                <svg viewBox="0 0 248 92" className="w-full" role="img" aria-label="Signal kept over timesteps for each schedule">
                                    <line x1="0" y1="84" x2="248" y2="84" stroke="rgba(243,236,224,0.15)" />
                                    {toy.schedules.map((s) => (
                                        <polyline
                                            key={s.name}
                                            points={curve(s)}
                                            fill="none"
                                            stroke={s.name === scheduleName ? "#ff5a4a" : "rgba(243,236,224,0.3)"}
                                            strokeWidth={s.name === scheduleName ? 2 : 1}
                                        />
                                    ))}
                                    {t > 0 && <line x1={t * (240 / T)} y1="0" x2={t * (240 / T)} y2="84" stroke="rgba(243,236,224,0.35)" strokeDasharray="2 2" />}
                                </svg>
                                <figcaption className="mt-1 flex justify-between text-xs text-slate-400">
                                    <span>Signal kept √ᾱ_t, from 1 (top) to 0</span>
                                    <span>t →</span>
                                </figcaption>
                            </figure>
                        )}
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                        <h3 className="text-xl font-semibold">The whole algorithm</h3>
                        <p className="mt-2 text-sm leading-6 text-slate-400">
                            Same lines as the MNIST model. Here ε_θ is an MLP (3 → 128 → 128 → 2) trained for{" "}
                            {toy ? toy.trainSteps.toLocaleString("en-US") : "…"} steps on {N.toLocaleString("en-US")} points.
                        </p>
                        <pre className="mt-4 overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 font-mono text-xs leading-6 text-slate-200">
{`# forward: jump straight to step t
xt = ab[t].sqrt() * x0 + (1 - ab[t]).sqrt() * eps
# train: predict the noise
loss = ((net(xt, t) - eps) ** 2).mean()
# DDPM: one step back, t -> t-1, plus fresh noise
x = (x - beta[t] / (1 - ab[t]).sqrt() * net(x, t)) / alpha[t].sqrt()
x = x + beta[t].sqrt() * randn()   # except at t = 0
# DDIM: guess x0, then jump to an earlier step s, no noise
x0 = (x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()
x = ab[s].sqrt() * x0 + (1 - ab[s]).sqrt() * eps`}
                        </pre>
                        <p className="mt-3 text-xs text-slate-400">
                            Train it yourself: <code className="text-slate-300">python toy2d.py</code> (about 15 minutes on a
                            CPU for all 12 models).
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
