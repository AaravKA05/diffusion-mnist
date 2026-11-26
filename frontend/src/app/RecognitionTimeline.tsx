"use client";

import { useState } from "react";

type TimelinePoint = {
    step: number;
    timestep: number;
    target_prob: number;
    match_rate: number;
};

type TimelinePreview = {
    step: number;
    timestep: number;
    image: string;
    pred: number;
    conf: number;
};

type TimelineSeries = {
    w: number;
    points: TimelinePoint[];
    lock_in_step: number | null;
    final_matches: number;
    previews: TimelinePreview[];
};

type TimelineResult = {
    digit: number;
    steps: number;
    count: number;
    seed: number;
    seconds: number;
    series: TimelineSeries[];
};

// Validated categorical palette for the dark surface (#111522):
// all four pass lightness, chroma, colour-blind separation and contrast checks.
const SERIES_COLORS = ["#338ced", "#e76624", "#00a888", "#d28e05"];
const W_OPTIONS = [0, 0.5, 1, 1.2, 2, 3];

const CHART = { width: 720, height: 300, left: 48, right: 64, top: 16, bottom: 40 };

function TimelineChart({ result }: { result: TimelineResult }) {
    const [hoverStep, setHoverStep] = useState<number | null>(null);

    const steps = result.steps;
    const plotW = CHART.width - CHART.left - CHART.right;
    const plotH = CHART.height - CHART.top - CHART.bottom;
    const x = (step: number) =>
        CHART.left + (steps === 1 ? 0 : ((step - 1) / (steps - 1)) * plotW);
    const y = (value: number) => CHART.top + (1 - value) * plotH;

    const xTicks = Array.from(
        new Set([1, Math.round(steps * 0.25), Math.round(steps * 0.5), Math.round(steps * 0.75), steps])
    ).filter((s) => s >= 1);

    // End-of-line labels: keep them at least 13px apart so they never overlap
    const labelY: number[] = (() => {
        const order = result.series
            .map((s, i) => ({ i, y: y(s.points[s.points.length - 1].target_prob) + 4 }))
            .sort((a, b) => a.y - b.y);
        for (let k = 1; k < order.length; k++) {
            order[k].y = Math.max(order[k].y, order[k - 1].y + 13);
        }
        const out: number[] = [];
        order.forEach((o) => (out[o.i] = o.y));
        return out;
    })();

    function handleMove(event: React.PointerEvent<SVGRectElement>) {
        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = (event.clientX - rect.left) / rect.width;
        const step = Math.min(steps, Math.max(1, Math.round(ratio * (steps - 1)) + 1));
        setHoverStep(step);
    }

    const hoverPoints =
        hoverStep === null
            ? []
            : result.series.map((s, i) => ({
                w: s.w,
                color: SERIES_COLORS[i],
                point: s.points[hoverStep - 1],
            }));

    return (
        <div className="relative">
            <svg
                viewBox={`0 0 ${CHART.width} ${CHART.height}`}
                className="h-auto w-full"
                role="img"
                aria-label={`Classifier probability of digit ${result.digit} at each sampling step, one line per guidance scale`}
            >
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                    <g key={v}>
                        <line
                            x1={CHART.left}
                            x2={CHART.width - CHART.right}
                            y1={y(v)}
                            y2={y(v)}
                            stroke="rgb(255 255 255 / 0.08)"
                            strokeWidth={1}
                        />
                        <text
                            x={CHART.left - 10}
                            y={y(v) + 4}
                            textAnchor="end"
                            fontSize={11}
                            fill="rgb(148 163 184)"
                        >
                            {Math.round(v * 100)}%
                        </text>
                    </g>
                ))}

                {/* 90% line used for the "recognisable from" step */}
                <line
                    x1={CHART.left}
                    x2={CHART.width - CHART.right}
                    y1={y(0.9)}
                    y2={y(0.9)}
                    stroke="rgb(167 139 250 / 0.45)"
                    strokeDasharray="4 4"
                    strokeWidth={1}
                />

                {xTicks.map((s) => (
                    <text
                        key={s}
                        x={x(s)}
                        y={CHART.height - CHART.bottom + 18}
                        textAnchor="middle"
                        fontSize={11}
                        fill="rgb(148 163 184)"
                    >
                        {s}
                    </text>
                ))}
                <text
                    x={CHART.left + plotW / 2}
                    y={CHART.height - 4}
                    textAnchor="middle"
                    fontSize={11}
                    fill="rgb(148 163 184)"
                >
                    DDIM step (noise → image)
                </text>

                {result.series.map((s, i) => {
                    const d = s.points
                        .map((p, k) => `${k === 0 ? "M" : "L"}${x(p.step).toFixed(1)},${y(p.target_prob).toFixed(1)}`)
                        .join(" ");
                    const last = s.points[s.points.length - 1];
                    const lock = s.lock_in_step ? s.points[s.lock_in_step - 1] : null;

                    return (
                        <g key={s.w}>
                            <path
                                d={d}
                                fill="none"
                                stroke={SERIES_COLORS[i]}
                                strokeWidth={2}
                                strokeLinejoin="round"
                                strokeLinecap="round"
                            />
                            {lock && (
                                <circle
                                    cx={x(lock.step)}
                                    cy={y(lock.target_prob)}
                                    r={5}
                                    fill={SERIES_COLORS[i]}
                                    stroke="#111522"
                                    strokeWidth={2}
                                />
                            )}
                            <text
                                x={x(last.step) + 8}
                                y={labelY[i]}
                                fontSize={11}
                                fill="rgb(226 232 240)"
                            >
                                w={s.w}
                            </text>
                        </g>
                    );
                })}

                {hoverStep !== null && (
                    <line
                        x1={x(hoverStep)}
                        x2={x(hoverStep)}
                        y1={CHART.top}
                        y2={CHART.top + plotH}
                        stroke="rgb(255 255 255 / 0.35)"
                        strokeWidth={1}
                    />
                )}
                {hoverPoints.map(
                    (h) =>
                        h.point && (
                            <circle
                                key={h.w}
                                cx={x(h.point.step)}
                                cy={y(h.point.target_prob)}
                                r={4}
                                fill={h.color}
                                stroke="#111522"
                                strokeWidth={2}
                            />
                        )
                )}

                <rect
                    x={CHART.left}
                    y={CHART.top}
                    width={plotW}
                    height={plotH}
                    fill="transparent"
                    onPointerMove={handleMove}
                    onPointerLeave={() => setHoverStep(null)}
                />
            </svg>

            {hoverStep !== null && (
                <div
                    className="pointer-events-none absolute top-2 rounded-lg border border-white/10 bg-canvas/95 px-3 py-2 text-xs shadow-xl"
                    style={{
                        left: `${(x(hoverStep) / CHART.width) * 100}%`,
                        transform: x(hoverStep) > CHART.width / 2 ? "translateX(calc(-100% - 12px))" : "translateX(12px)",
                    }}
                >
                    <p className="mb-1 text-slate-400">
                        Step {hoverStep} · t = {result.series[0].points[hoverStep - 1]?.timestep}
                    </p>
                    {hoverPoints.map(
                        (h) =>
                            h.point && (
                                <p key={h.w} className="flex items-center gap-2 tabular-nums text-slate-100">
                                    <span className="h-2 w-2 rounded-full" style={{ background: h.color }} />
                                    w={h.w}: {(h.point.target_prob * 100).toFixed(1)}%
                                </p>
                            )
                    )}
                </div>
            )}
        </div>
    );
}

export default function RecognitionTimeline({
    api,
    backendOnline,
}: {
    api: string;
    backendOnline: boolean | null;
}) {
    const [digit, setDigit] = useState(7);
    const [ws, setWs] = useState<number[]>([0, 1.2, 3]);
    const [steps, setSteps] = useState(30);
    const [count, setCount] = useState(8);
    const [seed, setSeed] = useState(42);
    const [result, setResult] = useState<TimelineResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [showTable, setShowTable] = useState(false);

    function toggleW(w: number) {
        setWs((current) => {
            if (current.includes(w)) {
                return current.length === 1 ? current : current.filter((v) => v !== w);
            }
            if (current.length >= 4) return current;
            return [...current, w].sort((a, b) => a - b);
        });
    }

    async function run() {
        setLoading(true);
        setError("");

        try {
            const response = await fetch(`${api}/api/recognition`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ digit, ws, steps, count, seed }),
            });

            if (!response.ok) {
                const message = await response.text();
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
                    ANALYSIS / WHEN DOES THE DIGIT APPEAR?
                </p>
                <h3 className="mt-2 text-2xl font-semibold">Recognition Timeline</h3>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                    At every DDIM step the model makes a guess of the final clean image.
                    The CNN classifier reads that guess and reports how likely it is to be
                    the requested digit. The step where the line crosses 90% and stays
                    there is where the digit becomes recognisable. Compare guidance scales
                    to see how guidance speeds this up.
                </p>
            </div>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[0.75fr_1.25fr]">
                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
                    <p className="text-xs tracking-widest text-violet-300">01 / CONFIGURATION</p>
                    <h3 className="mt-2 text-xl font-semibold">Timeline Settings</h3>

                    <div className="mt-7">
                        <div className="mb-3 flex justify-between text-sm">
                            <label>Target digit</label>
                            <span className="text-violet-300">{digit}</span>
                        </div>
                        <div className="grid grid-cols-5 gap-2">
                            {Array.from({ length: 10 }, (_, i) => (
                                <button
                                    key={i}
                                    onClick={() => setDigit(i)}
                                    aria-pressed={digit === i}
                                    aria-label={`Digit ${i}`}
                                    className={`aspect-square rounded-xl border text-lg font-semibold transition ${digit === i
                                        ? "border-violet-400 bg-violet-500/30 text-white"
                                        : "border-white/10 bg-white/3 text-slate-400 hover:border-violet-400/40 hover:bg-white/10 hover:text-white"
                                        }`}
                                >
                                    {i}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="mt-7">
                        <div className="mb-3 flex justify-between text-sm">
                            <label>Guidance scales to compare</label>
                            <span className="text-slate-400">{ws.length}/4 selected</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {W_OPTIONS.map((w) => {
                                const index = ws.indexOf(w);
                                const active = index !== -1;
                                return (
                                    <button
                                        key={w}
                                        onClick={() => toggleW(w)}
                                        aria-pressed={active}
                                        className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition ${active
                                            ? "border-violet-400/60 bg-violet-500/15 text-white"
                                            : "border-white/10 bg-white/3 text-slate-400 hover:bg-white/10 hover:text-white"
                                            }`}
                                    >
                                        {active && (
                                            <span
                                                className="h-2 w-2 rounded-full"
                                                style={{ background: SERIES_COLORS[index] }}
                                                aria-hidden="true"
                                            />
                                        )}
                                        w = {w}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    <div className="mt-7 space-y-6">
                        <div>
                            <div className="mb-2 flex justify-between text-sm">
                                <label htmlFor="rt-steps">DDIM sampling steps</label>
                                <span className="text-violet-300">{steps}</span>
                            </div>
                            <input
                                id="rt-steps"
                                type="range"
                                min="5"
                                max="100"
                                step="5"
                                value={steps}
                                onChange={(e) => setSteps(Number(e.target.value))}
                                className="w-full accent-violet-500"
                            />
                        </div>

                        <div>
                            <div className="mb-2 flex justify-between text-sm">
                                <label htmlFor="rt-count">Samples averaged per line</label>
                                <span className="text-violet-300">{count}</span>
                            </div>
                            <input
                                id="rt-count"
                                type="range"
                                min="1"
                                max="16"
                                value={count}
                                onChange={(e) => setCount(Number(e.target.value))}
                                className="w-full accent-violet-500"
                            />
                        </div>

                        <div>
                            <label htmlFor="rt-seed" className="mb-2 block text-sm">
                                Random seed
                            </label>
                            <div className="flex gap-3">
                                <input
                                    id="rt-seed"
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
                        disabled={loading || backendOnline !== true}
                        className="mt-8 w-full rounded-xl bg-linear-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {loading
                            ? "◌ Tracing..."
                            : backendOnline === true
                                ? "✦ Trace Recognition"
                                : "Backend Unavailable"}
                    </button>

                    {error && (
                        <p className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
                            {error}
                        </p>
                    )}
                </div>

                <div className="flex min-h-100 flex-col rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                    <p className="text-xs tracking-widest text-violet-300">02 / RESULT</p>
                    <h3 className="mt-2 text-xl font-semibold">
                        Probability of digit {result ? result.digit : digit} at each step
                    </h3>

                    {loading ? (
                        <div className="dl-skeleton mt-6 flex flex-1 items-center justify-center rounded-xl border border-white/10 bg-black/20 p-8">
                            <p role="status" aria-live="polite" className="flex items-center gap-2 text-sm text-violet-300">
                                <span className="dl-spinner" aria-hidden="true" />
                                Running {ws.length} sampler{ws.length === 1 ? "" : "s"} and classifying every step...
                            </p>
                        </div>
                    ) : !result ? (
                        <div className="mt-6 flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                            <div className="flex h-24 w-24 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-4xl text-violet-300">
                                ⟿
                            </div>
                            <h4 className="mt-6 text-lg font-medium">Ready to trace</h4>
                            <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
                                Pick a digit and up to four guidance scales, then press Trace
                                Recognition to see when each one becomes readable.
                            </p>
                        </div>
                    ) : (
                        <div className="dl-fade-in mt-5">
                            <div className="mb-3 flex flex-wrap items-center gap-4 text-xs text-slate-300">
                                {result.series.map((s, i) => (
                                    <span key={s.w} className="flex items-center gap-2">
                                        <span className="h-0.5 w-5 rounded-full" style={{ background: SERIES_COLORS[i] }} />
                                        w = {s.w}
                                    </span>
                                ))}
                                <span className="flex items-center gap-2 text-slate-400">
                                    <span className="h-0 w-5 border-t border-dashed border-violet-400/70" />
                                    90% threshold
                                </span>
                                <span className="flex items-center gap-2 text-slate-400">
                                    <span className="h-2.5 w-2.5 rounded-full border-2 border-surface bg-slate-300" />
                                    recognisable from here
                                </span>
                            </div>

                            <TimelineChart result={result} />

                            <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
                                {result.series.map((s, i) => (
                                    <div key={s.w} className="rounded-xl border border-white/10 bg-white/3 p-4">
                                        <p className="flex items-center gap-2 text-xs font-semibold tracking-widest text-slate-400">
                                            <span className="h-2 w-2 rounded-full" style={{ background: SERIES_COLORS[i] }} />
                                            GUIDANCE w = {s.w}
                                        </p>
                                        <p className="mt-2 text-lg font-semibold">
                                            {s.lock_in_step
                                                ? `Recognisable from step ${s.lock_in_step} of ${result.steps}`
                                                : "Never settled on the digit"}
                                        </p>
                                        <p className="mt-1 text-xs text-slate-400">
                                            {s.lock_in_step
                                                ? `${Math.round((s.lock_in_step / result.steps) * 100)}% of the way through sampling · t = ${s.points[s.lock_in_step - 1].timestep}`
                                                : "Fewer than 90% of samples read as the digit at the end"}
                                            {" · "}final {s.final_matches}/{result.count} correct
                                        </p>
                                    </div>
                                ))}
                            </div>

                            <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
                                <span className="tabular-nums">
                                    {result.count} samples per line · seed {result.seed} · {result.seconds}s
                                </span>
                                <button
                                    onClick={() => setShowTable((v) => !v)}
                                    aria-expanded={showTable}
                                    className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-slate-300 hover:bg-white/10"
                                >
                                    {showTable ? "Hide table" : "Show as table"}
                                </button>
                            </div>

                            {showTable && (
                                <div className="mt-3 max-h-72 overflow-auto rounded-xl border border-white/10">
                                    <table className="w-full text-left text-xs tabular-nums">
                                        <thead className="sticky top-0 bg-surface-2 text-slate-300">
                                            <tr>
                                                <th className="px-3 py-2 font-medium">Step</th>
                                                <th className="px-3 py-2 font-medium">t</th>
                                                {result.series.map((s) => (
                                                    <th key={s.w} className="px-3 py-2 font-medium">
                                                        w={s.w} prob · match
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {result.series[0].points.map((p, k) => (
                                                <tr key={p.step} className="border-t border-white/5 text-slate-300">
                                                    <td className="px-3 py-1.5">{p.step}</td>
                                                    <td className="px-3 py-1.5">{p.timestep}</td>
                                                    {result.series.map((s) => (
                                                        <td key={s.w} className="px-3 py-1.5">
                                                            {(s.points[k].target_prob * 100).toFixed(1)}% ·{" "}
                                                            {Math.round(s.points[k].match_rate * 100)}%
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {result && !loading && (
                <div className="dl-fade-in rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
                    <p className="text-xs tracking-widest text-violet-300">03 / WHAT THE MODEL IS GUESSING</p>
                    <h3 className="mt-2 text-xl font-semibold">Clean-image guess at each stage</h3>
                    <p className="mt-2 text-sm text-slate-400">
                        First sample of each run. The label under each image is what the
                        classifier reads in the guess at that step.
                    </p>

                    <div className="mt-6 space-y-6">
                        {result.series.map((s, i) => (
                            <div key={s.w}>
                                <p className="mb-3 flex items-center gap-2 text-sm text-slate-300">
                                    <span className="h-2 w-2 rounded-full" style={{ background: SERIES_COLORS[i] }} />
                                    w = {s.w}
                                </p>
                                <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
                                    {s.previews.map((pv) => (
                                        <figure
                                            key={pv.step}
                                            className={`rounded-xl border-2 bg-black/40 p-2 ${pv.pred === result.digit ? "border-emerald-500/60" : "border-white/10"
                                                }`}
                                        >
                                            <img
                                                src={`data:image/png;base64,${pv.image}`}
                                                alt={`Model guess at step ${pv.step}, read as ${pv.pred}`}
                                                className="aspect-square w-full rounded-lg object-contain"
                                            />
                                            <figcaption className="mt-2 text-center text-xs text-slate-400">
                                                Step {pv.step} · reads {pv.pred} ({Math.round(pv.conf * 100)}%)
                                            </figcaption>
                                        </figure>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </section>
    );
}