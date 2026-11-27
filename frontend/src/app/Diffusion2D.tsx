"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Pause, Play, Shuffle } from "lucide-react";

/**
 * 2D playground: the same DDPM recipe as the MNIST model, on a swiss roll.
 * Weights come from toy2d.py (public/models/toy2d.json). The MLP is tiny
 * (3 -> 128 -> 128 -> 2), so it runs in plain TypeScript on the main thread,
 * a few steps per frame. No backend or ONNX runtime needed.
 */

type Layer = { in: number; out: number; W: Float32Array; b: Float32Array };
type Schedule = {
    name: string;
    betaEnd: number;
    alphaBarT: number;
    sampleToData: number;
    dataToSample: number;
    layers: Layer[];
    betas: Float64Array;
    alphas: Float64Array;
    alphaBar: Float64Array;
};
type Toy = { T: number; trainSteps: number; data: Float32Array; schedules: Schedule[] };

type RawLayer = { in: number; out: number; W: string; b: string };
type RawSchedule = {
    betaStart: number;
    betaEnd: number;
    alpha_bar_T: number;
    sample_to_data: number;
    data_to_sample: number;
    layers: RawLayer[];
};

const N = 2000;
const VIEW = 3.2; // half-width of the plotted square
const STEPS_PER_FRAME = 4;
const SCHEDULE_NOTES: Record<string, string> = {
    gentle: "Never reaches pure noise: 60% of the signal is left at t = 100.",
    linear: "Signal reaches about zero right at the last step.",
    aggressive: "Signal is gone by about t = 50, so half the steps do nothing useful.",
};

function decode(b64: string): Float32Array {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
}

function makeSchedule(T: number, start: number, end: number) {
    const betas = new Float64Array(T);
    const alphas = new Float64Array(T);
    const alphaBar = new Float64Array(T);
    let ab = 1;
    for (let i = 0; i < T; i++) {
        betas[i] = start + ((end - start) * i) / (T - 1);
        alphas[i] = 1 - betas[i];
        ab *= alphas[i];
        alphaBar[i] = ab;
    }
    return { betas, alphas, alphaBar };
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

export default function Diffusion2D() {
    const [toy, setToy] = useState<Toy | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [scheduleName, setScheduleName] = useState("linear");
    const [mode, setMode] = useState<"forward" | "reverse">("reverse");
    const [t, setT] = useState(100);
    const [seed, setSeed] = useState(1);
    const [showData, setShowData] = useState(false);
    const [playing, setPlaying] = useState(false);
    // frames[k] = points at timestep k (k = T is the starting noise)
    const [frames, setFrames] = useState<(Float32Array | null)[]>([]);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        let cancelled = false;
        fetch("/models/toy2d.json")
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then((raw) => {
                if (cancelled) return;
                const schedules = Object.entries(raw.schedules as Record<string, RawSchedule>).map(
                    ([name, s]) => ({
                        name,
                        betaEnd: s.betaEnd,
                        alphaBarT: s.alpha_bar_T,
                        sampleToData: s.sample_to_data,
                        dataToSample: s.data_to_sample,
                        layers: s.layers.map((l) => ({ in: l.in, out: l.out, W: decode(l.W), b: decode(l.b) })),
                        ...makeSchedule(raw.T, s.betaStart, s.betaEnd),
                    }),
                );
                setToy({ T: raw.T, trainSteps: raw.trainSteps, data: decode(raw.data), schedules });
            })
            .catch((err) => {
                if (!cancelled) setLoadError(`Could not load the 2D model: ${err instanceof Error ? err.message : "unknown error"}.`);
            });
        return () => {
            cancelled = true;
        };
    }, []);

    const schedule = toy?.schedules.find((s) => s.name === scheduleName) ?? null;
    const T = toy?.T ?? 100;

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
        let step = T - 1;

        const tick = () => {
            if (cancelled) return;
            for (let n = 0; n < STEPS_PER_FRAME && step >= 0; n++, step--) {
                predictNoise(schedule.layers, x, step, T, eps);
                const b = schedule.betas[step];
                const a = schedule.alphas[step];
                const coef = b / Math.sqrt(1 - schedule.alphaBar[step]);
                const sa = Math.sqrt(a);
                const sb = Math.sqrt(b);
                const next = new Float32Array(2 * N);
                for (let i = 0; i < 2 * N; i++) {
                    next[i] = (x[i] - coef * eps[i]) / sa + (step > 0 ? sb * g() : 0);
                }
                x = next;
                all[step] = x;
            }
            setFrames(all.slice());
            if (step >= 0) raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => {
            cancelled = true;
            cancelAnimationFrame(raf);
        };
    }, [toy, schedule, seed, T]);

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
    }, [points, showData, toy, t]);

    // Redraw on resize
    const [, setResizeTick] = useState(0);
    useEffect(() => {
        const onResize = () => setResizeTick((n) => n + 1);
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

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

    function switchMode(next: "forward" | "reverse") {
        setPlaying(false);
        setMode(next);
        setT(next === "reverse" ? T : 0);
    }

    const signal = schedule && t > 0 ? Math.sqrt(schedule.alphaBar[t - 1]) : 1;
    const noise = schedule && t > 0 ? Math.sqrt(1 - schedule.alphaBar[t - 1]) : 0;
    const ready = frames.filter(Boolean).length;
    const computing = toy !== null && ready > 0 && ready < T + 1;
    const waiting = mode === "reverse" && !frames[t];

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
                    <div className="flex flex-wrap items-center justify-between gap-3">
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
                            aria-label={`${N} points at timestep ${t} of ${T}, ${mode === "reverse" ? "denoising" : "adding noise"}, ${scheduleName} schedule`}
                            className="block aspect-square w-full"
                        />
                        {(!toy || waiting) && (
                            <p className="absolute inset-0 flex items-center justify-center text-sm text-slate-400">
                                {!toy ? "Loading the 2D model…" : `Sampling… ${ready - 1}/${T} steps`}
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
                                setPlaying(false);
                                setSeed((s) => s + 1);
                                setT(mode === "reverse" ? T : 0);
                            }}
                            disabled={!toy}
                            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/10 px-4 text-sm hover:bg-white/5 disabled:opacity-50"
                        >
                            <Shuffle className="h-4 w-4" strokeWidth={1.75} aria-hidden="true" />
                            New noise
                        </button>
                        <p className="ml-auto text-xs tabular-nums text-slate-400">
                            Seed {seed} · {computing ? `sampling ${ready - 1}/${T}` : "in your browser"}
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
                </div>

                {/* Schedules + recipe */}
                <div className="space-y-6">
                    <div className="rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                        <h3 className="text-xl font-semibold">Noise schedule</h3>
                        <p className="mt-2 text-sm leading-6 text-slate-400">
                            Three models with the same MLP and data. Only β at the last step changes
                            (all start at β₁ = 1e-4, linear).
                        </p>

                        <div role="radiogroup" aria-label="Noise schedule" className="mt-5 space-y-2">
                            {toy?.schedules.map((s) => {
                                const active = s.name === scheduleName;
                                return (
                                    <button
                                        key={s.name}
                                        role="radio"
                                        aria-checked={active}
                                        onClick={() => {
                                            setPlaying(false);
                                            setScheduleName(s.name);
                                            setT(mode === "reverse" ? T : 0);
                                        }}
                                        className={`w-full rounded-xl border p-3 text-left transition ${active ? "border-violet-500 bg-violet-500/10" : "border-white/10 hover:bg-white/5"}`}
                                    >
                                        <div className="flex items-baseline justify-between gap-2">
                                            <span className="font-medium capitalize">{s.name}</span>
                                            <span className="text-xs tabular-nums text-slate-400">
                                                β_T = {s.betaEnd} · ᾱ_T = {s.alphaBarT.toPrecision(2)}
                                            </span>
                                        </div>
                                        <p className="mt-1 text-xs text-slate-400">{SCHEDULE_NOTES[s.name]}</p>
                                        <p className="mt-1 text-xs tabular-nums text-slate-300">
                                            Sample to data distance {s.sampleToData.toFixed(3)} (real data: 0.011)
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
                            Same three lines as the MNIST model. Here ε_θ is an MLP (3 → 128 → 128 → 2)
                            trained for {toy ? toy.trainSteps.toLocaleString("en-US") : "…"} steps on {N.toLocaleString("en-US")} points.
                        </p>
                        <pre className="mt-4 overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 font-mono text-xs leading-6 text-slate-200">
{`# forward: jump straight to step t
xt = ab[t].sqrt() * x0 + (1 - ab[t]).sqrt() * eps
# train: predict the noise
loss = ((net(xt, t) - eps) ** 2).mean()
# reverse: one step back, t -> t-1
x = (x - beta[t] / (1 - ab[t]).sqrt() * net(x, t)) / alpha[t].sqrt()
x = x + beta[t].sqrt() * randn()   # except at t = 0`}
                        </pre>
                        <p className="mt-3 text-xs text-slate-400">
                            Train it yourself: <code className="text-slate-300">python toy2d.py</code> (about 3 minutes on a CPU).
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
