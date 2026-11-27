"use client";

import { useEffect, useRef, useState } from "react";

import { apiFetch, onActivityChange, type Activity } from "@/lib/engine/apiClient";
import { claimInlineProgress } from "@/lib/inlineProgress";

type Frame = { timestep: number; image: string };

const STEPS = 20;
const SNAPSHOTS = 8; // plus the starting noise = 9 frames
const DIGITS = Array.from({ length: 10 }, (_, i) => i);

/**
 * Launch-page demo: pick a digit and the real model draws it.
 * Uses the same /api/visualize call as the Denoising visualizer, so it runs
 * on the laptop server when it is up and in the browser when it is not.
 */
export default function LiveDenoise() {
    const [digit, setDigit] = useState<number | null>(null);
    const [frames, setFrames] = useState<Frame[]>([]);
    const [shown, setShown] = useState(0);
    const [running, setRunning] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [activity, setActivity] = useState<Activity | null>(null);
    const runId = useRef(0);

    useEffect(() => onActivityChange(setActivity), []);

    // Our own progress line is on screen while running: hide the floating toast
    useEffect(() => (running ? claimInlineProgress() : undefined), [running]);

    // Reveal the frames one after another, noise first
    useEffect(() => {
        if (shown >= frames.length) return;
        const id = window.setTimeout(() => setShown((n) => n + 1), 110);
        return () => window.clearTimeout(id);
    }, [shown, frames]);

    async function run(next: number) {
        const id = ++runId.current;
        setDigit(next);
        setRunning(true);
        setError(null);
        setFrames([]);
        setShown(0);
        try {
            const response = await apiFetch("/api/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    digit: next,
                    w: 2,
                    steps: STEPS,
                    seed: 1000 + id * 7919, // a different start each run
                    snapshot_count: SNAPSHOTS,
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data?.detail ?? `Request failed (${response.status})`);
            if (id === runId.current) {
                const snapshots = data.snapshots as Frame[];
                const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
                setFrames(snapshots);
                setShown(calm ? snapshots.length : 0);
            }
        } catch (err) {
            if (id === runId.current) {
                setError(
                    `The model could not run: ${err instanceof Error ? err.message : "unknown error"}. Pick a digit to try again.`,
                );
            }
        } finally {
            if (id === runId.current) setRunning(false);
        }
    }

    const mine = running && activity !== null;
    let status = "Pick a digit to start.";
    if (running) {
        if (mine && activity.mode === "server") status = "Running on our server…";
        else if (mine && activity.total && activity.done > 0)
            status = `Step ${activity.done} of ${activity.total}`;
        else status = "Loading the model. The first run downloads about 25 MB.";
    } else if (frames.length > 0 && digit !== null) {
        status = `Done. ${STEPS} denoising steps from random noise to a ${digit}.`;
    }
    const percent = mine && activity.total ? Math.round((activity.done / activity.total) * 100) : 0;

    return (
        <section aria-labelledby="live-heading" className="border-t border-slate-700 py-12" data-lightbox>
            <div className="flex flex-wrap items-end justify-between gap-x-10 gap-y-5">
                <div className="max-w-xl">
                    <h2 id="live-heading" className="text-3xl font-semibold tracking-[-0.02em]">
                        {digit === null ? "Watch the model draw a digit" : `Watch the model draw a ${digit}`}
                    </h2>
                    <p className="mt-3 text-slate-300">
                        Pick a digit. The model starts from random noise and removes a little of it at each of{" "}
                        {STEPS} steps. This is the real model running, not a recording.
                    </p>
                </div>
                <div role="group" aria-label="Digit to draw" className="flex flex-wrap gap-1.5">
                    {DIGITS.map((d) => (
                        <button
                            key={d}
                            type="button"
                            onClick={() => run(d)}
                            disabled={running}
                            aria-pressed={digit === d}
                            className={`h-11 w-11 rounded-md border text-base font-semibold tabular-nums disabled:cursor-wait ${
                                digit === d
                                    ? "border-violet-500 bg-violet-500 text-black"
                                    : "border-slate-700 text-slate-200 enabled:hover:border-slate-400"
                            }`}
                        >
                            {d}
                        </button>
                    ))}
                </div>
            </div>

            <ol className="mt-8 grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-9">
                {Array.from({ length: SNAPSHOTS + 1 }, (_, k) => {
                    const frame = k < shown ? frames[k] : undefined;
                    return (
                        <li key={k} className="flex flex-col gap-2">
                            <div
                                className={`aspect-square overflow-hidden rounded-md ${
                                    frame ? "bg-black" : "border border-dashed border-slate-700"
                                }`}
                            >
                                {frame && (
                                    <img
                                        src={`data:image/png;base64,${frame.image}`}
                                        alt={`Step ${k + 1} of the denoising path, timestep ${frame.timestep}`}
                                        className="h-full w-full object-contain"
                                    />
                                )}
                            </div>
                            <span className="text-center text-sm tabular-nums text-slate-400">
                                {frame ? (frame.timestep < 0 ? "Clean" : `t = ${frame.timestep}`) : " "}
                            </span>
                        </li>
                    );
                })}
            </ol>

            <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2">
                <p role="status" aria-live="polite" className="text-sm tabular-nums text-slate-300">
                    {status}
                </p>
                {running && mine && activity.mode === "browser" && activity.done > 0 && (
                    <div
                        className="h-1 w-40 overflow-hidden rounded-full bg-slate-700"
                        role="progressbar"
                        aria-label="Denoising progress"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={percent}
                    >
                        <div className="h-full rounded-full bg-violet-500" style={{ width: `${percent}%` }} />
                    </div>
                )}
            </div>
            {error && (
                <p role="alert" className="mt-3 text-sm text-violet-200">
                    {error}
                </p>
            )}
        </section>
    );
}
