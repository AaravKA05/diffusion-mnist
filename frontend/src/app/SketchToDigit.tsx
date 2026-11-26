"use client";

import { useEffect, useRef, useState } from "react";

import { apiFetch } from "@/lib/engine/apiClient";

type SketchItem = {
    image: string;
    pred: number;
    conf: number;
};

type SketchResult = {
    input: string;
    noised: string;
    sketch_pred: number;
    sketch_conf: number;
    items: SketchItem[];
    matches: number | null;
    t_start: number;
    steps_used: number;
    seconds: number;
    params: { digit: number | null; strength: number; w: number; steps: number; seed: number };
};

const CANVAS_SIZE = 280;
const BRUSH = 18;

function strengthLabel(strength: number) {
    if (strength <= 0.3) return "Light clean-up: keeps your shape";
    if (strength <= 0.6) return "Balanced: tidies and reshapes";
    if (strength < 0.9) return "Strong: mostly redrawn by the model";
    return "Full: almost ignores the sketch";
}

export default function SketchToDigit({
    backendOnline,
}: {
    backendOnline: boolean | null;
}) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const drawing = useRef(false);
    const lastPoint = useRef<{ x: number; y: number } | null>(null);

    const [hasInk, setHasInk] = useState(false);
    const [digit, setDigit] = useState<number | null>(null);
    const [strength, setStrength] = useState(0.5);
    const [guidance, setGuidance] = useState(1.2);
    const [steps, setSteps] = useState(50);
    const [count, setCount] = useState(4);
    const [seed, setSeed] = useState(42);
    const [result, setResult] = useState<SketchResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    function resetCanvas() {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setHasInk(false);
    }

    useEffect(() => {
        resetCanvas();
    }, []);

    function pointFromEvent(event: React.PointerEvent<HTMLCanvasElement>) {
        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        return {
            x: ((event.clientX - rect.left) / rect.width) * canvas.width,
            y: ((event.clientY - rect.top) / rect.height) * canvas.height,
        };
    }

    function drawTo(point: { x: number; y: number }) {
        const ctx = canvasRef.current?.getContext("2d");
        if (!ctx) return;
        const from = lastPoint.current ?? point;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = BRUSH;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
        lastPoint.current = point;
    }

    function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
        event.currentTarget.setPointerCapture(event.pointerId);
        drawing.current = true;
        lastPoint.current = null;
        drawTo(pointFromEvent(event));
        setHasInk(true);
    }

    function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
        if (!drawing.current) return;
        drawTo(pointFromEvent(event));
    }

    function onPointerUp() {
        drawing.current = false;
        lastPoint.current = null;
    }

    async function run() {
        const canvas = canvasRef.current;
        if (!canvas) return;

        setLoading(true);
        setError("");

        try {
            const response = await apiFetch("/api/sketch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: canvas.toDataURL("image/png"),
                    digit,
                    strength,
                    w: guidance,
                    steps,
                    count,
                    seed,
                }),
            });

            if (!response.ok) {
                let message = await response.text();
                try {
                    message = JSON.parse(message).detail ?? message;
                } catch {
                    // keep the plain text message
                }
                throw new Error(`Request failed (${response.status}): ${message}`);
            }

            setResult(await response.json());
        } catch (err) {
            setError(err instanceof Error ? err.message : "Request failed.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <section className="space-y-6">
            <div className="rounded-2xl border border-violet-400/20 bg-violet-500/6 p-6 sm:p-8">
                <p className="text-xs tracking-widest text-violet-300">
                    GUIDED EDITING / SDEDIT
                </p>
                <h3 className="mt-2 text-2xl font-semibold">Sketch to Digit</h3>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                    Draw a rough digit. The model adds noise to your sketch, then removes
                    it again with the reverse diffusion process. With low strength it
                    only tidies your strokes; with high strength it redraws the digit
                    almost from scratch. No retraining is needed: this uses the same
                    model as the Studio.
                </p>
            </div>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[0.9fr_1.1fr]">
                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
                    <p className="text-xs tracking-widest text-violet-300">01 / DRAW</p>
                    <h3 className="mt-2 text-xl font-semibold">Your Sketch</h3>

                    <div className="mt-6 flex flex-col items-center gap-3 sm:flex-row sm:items-start">
                        <canvas
                            ref={canvasRef}
                            width={CANVAS_SIZE}
                            height={CANVAS_SIZE}
                            onPointerDown={onPointerDown}
                            onPointerMove={onPointerMove}
                            onPointerUp={onPointerUp}
                            onPointerCancel={onPointerUp}
                            aria-label="Drawing canvas. Draw a digit with the mouse, a finger or a pen."
                            className="aspect-square w-full max-w-70 cursor-crosshair touch-none rounded-xl border border-violet-400/30 bg-black shadow-inner"
                        />
                        <div className="flex w-full flex-row gap-2 sm:w-auto sm:flex-col">
                            <button
                                onClick={resetCanvas}
                                className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-slate-300 hover:bg-white/10 sm:flex-none"
                            >
                                ✕ Clear
                            </button>
                            <p className="hidden max-w-40 text-xs leading-5 text-slate-400 sm:block">
                                Draw big and centred, like a handwritten digit. Thick strokes work best.
                            </p>
                        </div>
                    </div>

                    <div className="mt-7">
                        <div className="mb-3 flex justify-between text-sm">
                            <label>Turn it into</label>
                            <span className="text-violet-300">
                                {digit === null ? "Model decides" : `Digit ${digit}`}
                            </span>
                        </div>
                        <div className="grid grid-cols-6 gap-2">
                            <button
                                onClick={() => setDigit(null)}
                                aria-pressed={digit === null}
                                className={`col-span-6 rounded-xl border py-2 text-sm font-medium transition sm:col-span-1 ${digit === null
                                    ? "border-violet-400 bg-violet-500/30 text-white"
                                    : "border-white/10 bg-white/3 text-slate-400 hover:border-violet-400/40 hover:bg-white/10 hover:text-white"
                                    }`}
                            >
                                Auto
                            </button>
                            {Array.from({ length: 10 }, (_, i) => (
                                <button
                                    key={i}
                                    onClick={() => setDigit(i)}
                                    aria-pressed={digit === i}
                                    aria-label={`Digit ${i}`}
                                    className={`rounded-xl border py-2 text-base font-semibold transition ${digit === i
                                        ? "border-violet-400 bg-violet-500/30 text-white"
                                        : "border-white/10 bg-white/3 text-slate-400 hover:border-violet-400/40 hover:bg-white/10 hover:text-white"
                                        }`}
                                >
                                    {i}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="mt-7 space-y-6">
                        <div>
                            <div className="mb-2 flex justify-between text-sm">
                                <label htmlFor="sk-strength">Edit strength</label>
                                <span className="text-violet-300 tabular-nums">{strength.toFixed(2)}</span>
                            </div>
                            <input
                                id="sk-strength"
                                type="range"
                                min="0.1"
                                max="1"
                                step="0.05"
                                value={strength}
                                onChange={(e) => setStrength(Number(e.target.value))}
                                className="w-full accent-violet-500"
                            />
                            <p className="mt-1 text-xs text-slate-400">{strengthLabel(strength)}</p>
                        </div>

                        {digit !== null && (
                            <div>
                                <div className="mb-2 flex justify-between text-sm">
                                    <label htmlFor="sk-w">Guidance scale</label>
                                    <span className="text-violet-300">{guidance.toFixed(1)}</span>
                                </div>
                                <input
                                    id="sk-w"
                                    type="range"
                                    min="0"
                                    max="3"
                                    step="0.1"
                                    value={guidance}
                                    onChange={(e) => setGuidance(Number(e.target.value))}
                                    className="w-full accent-violet-500"
                                />
                            </div>
                        )}

                        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                            <div>
                                <div className="mb-2 flex justify-between text-sm">
                                    <label htmlFor="sk-steps">DDIM steps (full)</label>
                                    <span className="text-violet-300">{steps}</span>
                                </div>
                                <input
                                    id="sk-steps"
                                    type="range"
                                    min="10"
                                    max="100"
                                    step="5"
                                    value={steps}
                                    onChange={(e) => setSteps(Number(e.target.value))}
                                    className="w-full accent-violet-500"
                                />
                            </div>
                            <div>
                                <div className="mb-2 flex justify-between text-sm">
                                    <label htmlFor="sk-count">Variations</label>
                                    <span className="text-violet-300">{count}</span>
                                </div>
                                <input
                                    id="sk-count"
                                    type="range"
                                    min="1"
                                    max="8"
                                    value={count}
                                    onChange={(e) => setCount(Number(e.target.value))}
                                    className="w-full accent-violet-500"
                                />
                            </div>
                        </div>

                        <div>
                            <label htmlFor="sk-seed" className="mb-2 block text-sm">
                                Random seed
                            </label>
                            <div className="flex gap-3">
                                <input
                                    id="sk-seed"
                                    type="number"
                                    min="0"
                                    max="2147483647"
                                    value={seed}
                                    onChange={(e) =>
                                        setSeed(Math.max(0, Math.min(2147483647, Number(e.target.value) || 0)))
                                    }
                                    className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none transition focus:border-violet-400/60 focus:ring-2 focus:ring-violet-500/30"
                                />
                                <button
                                    onClick={() => setSeed(Math.floor(Math.random() * 100000))}
                                    className="rounded-xl border border-white/10 bg-white/5 px-4 text-xs hover:bg-white/10"
                                >
                                    ↻ Random
                                </button>
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={run}
                        disabled={loading || !hasInk || backendOnline !== true}
                        className="mt-8 w-full rounded-xl bg-linear-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {loading
                            ? "◌ Denoising your sketch..."
                            : backendOnline !== true
                                ? "Backend Unavailable"
                                : hasInk
                                    ? "✦ Clean Up Sketch"
                                    : "Draw something first"}
                    </button>

                    {error && (
                        <p className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
                            {error}
                        </p>
                    )}
                </div>

                <div className="flex min-h-120 flex-col rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                    <p className="text-xs tracking-widest text-violet-300">02 / OUTPUT</p>
                    <h3 className="mt-2 text-xl font-semibold">From Sketch to Digit</h3>

                    {loading ? (
                        <div className="mt-6" aria-hidden="true">
                            <div className="grid grid-cols-2 gap-3">
                                {[0, 1].map((i) => (
                                    <div key={i} className="dl-skeleton aspect-square rounded-xl border border-white/10 bg-white/6" />
                                ))}
                            </div>
                            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                                {Array.from({ length: count }, (_, i) => (
                                    <div
                                        key={i}
                                        className="dl-skeleton aspect-square rounded-xl border border-white/10 bg-white/6"
                                        style={{ animationDelay: `${i * 80}ms` }}
                                    />
                                ))}
                            </div>
                        </div>
                    ) : !result ? (
                        <div className="mt-6 flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                            <div className="flex h-24 w-24 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-4xl text-violet-300">
                                ✎
                            </div>
                            <h4 className="mt-6 text-lg font-medium">Draw, then clean up</h4>
                            <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
                                Your processed sketch, the noisy version the model starts from,
                                and the cleaned-up results will appear here.
                            </p>
                        </div>
                    ) : (
                        <div className="dl-fade-in mt-6">
                            <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                                <figure className="rounded-xl border border-white/10 bg-black/40 p-2">
                                    <img
                                        src={`data:image/png;base64,${result.input}`}
                                        alt="Your sketch after MNIST-style centring"
                                        className="aspect-square w-full rounded-lg object-contain"
                                    />
                                    <figcaption className="mt-2 text-center text-xs text-slate-400">
                                        Your sketch · classifier reads {result.sketch_pred} ({Math.round(result.sketch_conf * 100)}%)
                                    </figcaption>
                                </figure>
                                <span className="text-xl text-violet-300" aria-hidden="true">→</span>
                                <figure className="rounded-xl border border-white/10 bg-black/40 p-2">
                                    <img
                                        src={`data:image/png;base64,${result.noised}`}
                                        alt={`Sketch with noise added up to timestep ${result.t_start}`}
                                        className="aspect-square w-full rounded-lg object-contain"
                                    />
                                    <figcaption className="mt-2 text-center text-xs text-slate-400">
                                        Noise added to t = {result.t_start}
                                    </figcaption>
                                </figure>
                            </div>

                            <p className="mt-6 mb-3 text-sm text-slate-300">
                                Cleaned up in {result.steps_used} DDIM steps
                                {result.params.digit === null
                                    ? " · model chose the digit"
                                    : ` · target ${result.params.digit} · ${result.matches}/${result.items.length} read as ${result.params.digit}`}
                            </p>

                            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                                {result.items.map((item, index) => {
                                    const target = result.params.digit;
                                    const border =
                                        target === null
                                            ? "border-white/10"
                                            : item.pred === target
                                                ? "border-emerald-500/70"
                                                : "border-red-500/70";
                                    return (
                                        <figure
                                            key={index}
                                            className={`dl-fade-in rounded-xl border-2 bg-black/40 p-2 transition duration-200 hover:-translate-y-0.5 ${border}`}
                                        >
                                            <img
                                                src={`data:image/png;base64,${item.image}`}
                                                alt={`Cleaned-up variation ${index + 1}, read as ${item.pred}`}
                                                className="aspect-square w-full rounded-lg object-contain"
                                            />
                                            <figcaption className="mt-2 text-center text-xs text-slate-400">
                                                Read as {item.pred} · {(item.conf * 100).toFixed(1)}%
                                            </figcaption>
                                        </figure>
                                    );
                                })}
                            </div>

                            <p className="mt-4 text-xs text-slate-400 tabular-nums">
                                Strength {result.params.strength.toFixed(2)} · seed {result.params.seed} · {result.seconds}s
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}