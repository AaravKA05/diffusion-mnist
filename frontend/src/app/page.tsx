
"use client";

import { useState } from "react";

const API = "http://127.0.0.1:8000";

type GeneratedItem = {
  image: string;
  pred: number;
  conf: number;
};

type CompareResult = {
  w: number;
  matches: number;
  items: GeneratedItem[];
};

export default function Home() {
  const [digit, setDigit] = useState(7);
  const [count, setCount] = useState(8);
  const [guidance, setGuidance] = useState(1.2);
  const [steps, setSteps] = useState(50);
  const [seed, setSeed] = useState(42);

  const [activeTab, setActiveTab] = useState("Studio");

  const [images, setImages] = useState<GeneratedItem[]>([]);
  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const [elapsed, setElapsed] = useState<number | null>(null);
  const [matches, setMatches] = useState<number | null>(null);

  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [compareTime, setCompareTime] = useState<number | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  async function generateImages() {
    setLoading(true);
    setError("");
    setStatus("Connecting to diffusion model...");

    try {
      const response = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit,
          count,
          w: guidance,
          steps,
          seed,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const data = await response.json();

      setImages(data.items);
      setElapsed(data.seconds);
      setMatches(data.matches);
      setStatus("Generation completed successfully.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the backend."
      );
      setStatus("");
    } finally {
      setLoading(false);
    }
  }

  async function compareGuidance() {
    setCompareLoading(true);
    setError("");
    setCompareResults([]);
    setCompareTime(null);

    try {
      const response = await fetch(`${API}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit,
          count: Math.min(count, 8),
          ws: [0.0, 1.2, 2.0],
          steps,
          seed,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const data = await response.json();

      setCompareResults(data.results);
      setCompareTime(data.seconds);
      setStatus("Guidance comparison completed successfully.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to compare guidance scales."
      );
    } finally {
      setCompareLoading(false);
    }
  }

  function ImageCard({
    item,
    index,
    selectedDigit,
  }: {
    item: GeneratedItem;
    index: number;
    selectedDigit: number;
  }) {
    const correct = item.pred === selectedDigit;

    return (
      <div
        className={`overflow-hidden rounded-xl border-2 bg-black/40 p-2 ${correct
          ? "border-emerald-500/70"
          : "border-red-500/70"
          }`}
      >
        <img
          src={`data:image/png;base64,${item.image}`}
          alt={`Generated digit ${selectedDigit}, sample ${index + 1}`}
          className="aspect-square w-full rounded-lg object-contain"
        />

        <p className="mt-2 text-center text-[10px] text-slate-400">
          Read as {item.pred} · {(item.conf * 100).toFixed(1)}%
        </p>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-[#080b14] text-white selection:bg-violet-500/30">

      {/* BACKGROUND */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-40 top-20 h-96 w-96 rounded-full bg-blue-600/10 blur-[130px]" />
        <div className="absolute right-0 top-40 h-96 w-96 rounded-full bg-violet-600/10 blur-[130px]" />
        <div className="absolute bottom-0 left-1/2 h-80 w-80 rounded-full bg-cyan-500/5 blur-[130px]" />
      </div>

      <div className="relative mx-auto flex min-h-screen max-w-[1600px]">

        {/* SIDEBAR */}
        <aside className="hidden w-64 shrink-0 border-r border-white/[0.07] bg-white/[0.015] p-6 lg:flex lg:flex-col">

          <div className="mb-12 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 text-xl font-bold">
              D
            </div>

            <div>
              <h1 className="text-lg font-semibold">DiffusionLab</h1>
              <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500">
                Generative Studio
              </p>
            </div>
          </div>

          <p className="mb-4 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
            Workspace
          </p>

          <nav className="space-y-2">
            {[
              ["◈", "Studio"],
              ["▦", "My Generations"],
              ["◉", "Model Insights"],
            ].map(([icon, name]) => (
              <button
                key={name}
                onClick={() => setActiveTab(name)}
                className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${activeTab === name
                  ? "border border-violet-400/20 bg-violet-500/10 text-white"
                  : "text-slate-500 hover:bg-white/5 hover:text-white"
                  }`}
              >
                <span className="text-lg">{icon}</span>
                {name}
              </button>
            ))}
          </nav>

          <div className="mt-auto rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <div className="mb-3 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="text-xs text-emerald-300">
                API configured
              </span>
            </div>

            <p className="text-sm font-medium">MNIST Model</p>
            <p className="mt-1 text-xs text-slate-500">
              Conditional DDPM · EMA
            </p>
            <p className="mt-3 text-[10px] text-slate-500">
              FastAPI · PyTorch · DDIM
            </p>
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <section className="min-w-0 flex-1 px-5 py-7 sm:px-8 lg:px-10">

          {/* HEADER */}
          <header className="mb-10 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="mb-2 text-xs tracking-widest text-violet-300">
                YOUR GENERATIVE WORKSPACE
              </p>

              <h2 className="text-3xl font-semibold sm:text-4xl">
                Creative Studio<span className="text-violet-400">.</span>
              </h2>

              <p className="mt-2 text-sm text-slate-500">
                Explore handwritten digit generation with diffusion models.
              </p>
            </div>

            <div className="flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.04] px-4 py-2.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="text-xs text-slate-300">
                Backend configured
              </span>
            </div>
          </header>

          {/* MODEL STATS */}
          <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[
              ["MODEL", "Conditional DDPM", "PyTorch architecture"],
              ["SAMPLING", "DDIM", "Configurable inference"],
              ["CLASSES", "10 Digits", "Labels 0 through 9"],
            ].map(([label, value, description]) => (
              <div
                key={label}
                className="rounded-2xl border border-white/[0.08] bg-white/[0.035] p-5 backdrop-blur-xl"
              >
                <p className="text-[10px] font-semibold tracking-[0.2em] text-slate-500">
                  {label}
                </p>

                <h3 className="mt-3 text-xl font-semibold">{value}</h3>

                <p className="mt-1 text-xs text-slate-500">
                  {description}
                </p>
              </div>
            ))}
          </div>

          {/* GENERATION STUDIO */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_0.85fr]">

            {/* CONTROLS */}
            <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 backdrop-blur-2xl sm:p-8">

              <div className="mb-7">
                <p className="text-xs tracking-widest text-violet-300">
                  01 / CONFIGURATION
                </p>

                <h3 className="mt-2 text-xl font-semibold">
                  Generation Controls
                </h3>
              </div>

              {/* DIGIT SELECTOR */}
              <div className="mb-8">
                <div className="mb-4 flex justify-between text-sm">
                  <label className="text-slate-300">
                    Select a digit
                  </label>

                  <span className="text-violet-300">
                    Selected: {digit}
                  </span>
                </div>

                <div className="grid grid-cols-5 gap-3 sm:grid-cols-10">
                  {Array.from({ length: 10 }, (_, i) => (
                    <button
                      key={i}
                      onClick={() => setDigit(i)}
                      className={`aspect-square rounded-xl border text-lg font-semibold transition hover:-translate-y-1 ${digit === i
                        ? "border-violet-400 bg-gradient-to-br from-violet-500 to-blue-600 text-white shadow-lg shadow-violet-500/20"
                        : "border-white/10 bg-white/[0.035] text-slate-400 hover:border-violet-400/50"
                        }`}
                    >
                      {i}
                    </button>
                  ))}
                </div>
              </div>

              {/* SLIDERS */}
              <div className="space-y-7">

                <div>
                  <div className="mb-3 flex justify-between text-sm">
                    <label className="text-slate-300">
                      Number of samples
                    </label>
                    <span>{count}</span>
                  </div>

                  <input
                    type="range"
                    min="1"
                    max="16"
                    value={count}
                    onChange={(e) => setCount(Number(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </div>

                <div>
                  <div className="mb-3 flex justify-between text-sm">
                    <label className="text-slate-300">
                      Guidance scale
                    </label>
                    <span className="text-violet-300">
                      {guidance.toFixed(1)}
                    </span>
                  </div>

                  <input
                    type="range"
                    min="0"
                    max="3"
                    step="0.1"
                    value={guidance}
                    onChange={(e) => setGuidance(Number(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </div>

                <div>
                  <div className="mb-3 flex justify-between text-sm">
                    <label className="text-slate-300">
                      DDIM sampling steps
                    </label>
                    <span className="text-violet-300">{steps}</span>
                  </div>

                  <input
                    type="range"
                    min="5"
                    max="100"
                    step="5"
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </div>

                {/* SEED */}
                <div>
                  <label className="mb-3 block text-sm text-slate-300">
                    Random seed
                  </label>

                  <div className="flex gap-3">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      value={seed}
                      onChange={(e) =>
                        setSeed(
                          Math.max(
                            0,
                            Math.min(
                              2147483647,
                              Number(e.target.value) || 0
                            )
                          )
                        )
                      }
                      className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none focus:border-violet-400"
                    />

                    <button
                      onClick={() =>
                        setSeed(Math.floor(Math.random() * 100000))
                      }
                      className="rounded-xl border border-white/10 bg-white/5 px-4 text-xs text-slate-300 hover:bg-white/10"
                    >
                      ↻ Random
                    </button>
                  </div>
                </div>
              </div>

              {/* GENERATE BUTTON */}
              <button
                onClick={generateImages}
                disabled={loading || compareLoading}
                className="mt-9 w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:brightness-110 disabled:cursor-wait disabled:opacity-60"
              >
                {loading ? "◌ Generating..." : "✦ Generate Images"}
              </button>

              <div className="mt-4 text-center text-xs text-slate-400">
                {status}
              </div>

              {error && (
                <div className="mt-3 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
                  {error}
                </div>
              )}
            </div>

            {/* GENERATION OUTPUT */}
            <div className="flex min-h-[480px] flex-col rounded-3xl border border-white/10 bg-[#111522]/70 p-6 backdrop-blur-2xl sm:p-8">

              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-xs tracking-widest text-cyan-300">
                    02 / OUTPUT
                  </p>

                  <h3 className="mt-2 text-xl font-semibold">
                    Generation Preview
                  </h3>
                </div>

                <span className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-400">
                  {images.length ? "Generated" : "Awaiting generation"}
                </span>
              </div>

              {images.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                  <div className="mb-6 flex h-24 w-24 items-center justify-center rounded-3xl border border-violet-400/20 bg-violet-500/10 text-5xl text-violet-300">
                    {digit}
                  </div>

                  <h4 className="text-lg font-medium">
                    Your canvas is ready
                  </h4>

                  <p className="mt-2 max-w-xs text-sm leading-6 text-slate-500">
                    Generate handwritten samples using your trained diffusion model.
                  </p>
                </div>
              ) : (
                <>
                  <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {images.map((item, index) => (
                      <ImageCard
                        key={`${index}-${item.pred}`}
                        item={item}
                        index={index}
                        selectedDigit={digit}
                      />
                    ))}
                  </div>

                  <div className="mt-auto grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-500">
                        Classifier matches
                      </p>

                      <p className="mt-1 text-xl font-semibold text-emerald-400">
                        {matches}/{images.length}
                      </p>
                    </div>

                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-500">
                        Inference time
                      </p>

                      <p className="mt-1 text-xl font-semibold">
                        {elapsed} s
                      </p>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* GUIDANCE COMPARISON */}
          <section className="mt-8 rounded-3xl border border-white/10 bg-[#111522]/80 p-6 backdrop-blur-2xl sm:p-8">

            <div className="mb-6">
              <p className="text-xs tracking-widest text-cyan-300">
                03 / MODEL EXPERIMENTATION
              </p>

              <h3 className="mt-2 text-2xl font-semibold">
                Guidance Comparison
                <span className="text-violet-400">.</span>
              </h3>

              <p className="mt-2 text-sm text-slate-400">
                Compare how classifier-free guidance changes the generated digits.
              </p>
            </div>

            {/* COMPARISON DETAILS */}
            <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {[
                ["GUIDANCE SCALES", "0.0 · 1.2 · 2.0"],
                ["SAME STARTING NOISE", `Seed ${seed}`],
                ["DDIM STEPS", `${steps} steps`],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-4"
                >
                  <p className="text-[10px] tracking-widest text-slate-500">
                    {label}
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-200">
                    {value}
                  </p>
                </div>
              ))}
            </div>

            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-slate-400">
                Digit {digit} · {Math.min(count, 8)} samples per guidance scale
              </p>

              <span className="text-xs text-slate-500">
                3 guidance settings
              </span>
            </div>

            {/* COMPARE BUTTON */}
            <button
              onClick={compareGuidance}
              disabled={compareLoading || loading}
              className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:brightness-110 disabled:cursor-wait disabled:opacity-60"
            >
              {compareLoading
                ? "◌ Comparing guidance scales..."
                : "✦ Compare Guidance — 0.0 vs 1.2 vs 2.0"}
            </button>

            {/* LOADING */}
            {compareLoading && (
              <div className="mt-6 rounded-xl border border-violet-400/20 bg-violet-500/5 p-5 text-center">
                <p className="text-sm text-violet-300">
                  Running three sampling experiments...
                </p>

                <p className="mt-2 text-xs text-slate-500">
                  This may take a little time on CPU.
                </p>
              </div>
            )}

            {/* COMPARISON RESULTS */}
            {compareResults.length > 0 && (
              <div className="mt-8">

                <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h4 className="text-lg font-semibold">
                      Experiment Results
                    </h4>

                    <p className="mt-1 text-xs text-slate-500">
                      Same digit, seed and DDIM steps across all experiments.
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2">
                    <p className="text-[10px] text-slate-500">
                      TOTAL INFERENCE TIME
                    </p>

                    <p className="mt-1 text-lg font-semibold text-cyan-300">
                      {compareTime} s
                    </p>
                  </div>
                </div>

                <div className="space-y-8">
                  {compareResults.map((result) => {
                    const accuracy =
                      result.items.length > 0
                        ? (result.matches / result.items.length) * 100
                        : 0;

                    return (
                      <div
                        key={result.w}
                        className="rounded-2xl border border-white/[0.08] bg-black/20 p-4 sm:p-5"
                      >
                        {/* RESULT HEADER */}
                        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="text-xs tracking-widest text-violet-300">
                              GUIDANCE SCALE
                            </p>

                            <h5 className="mt-1 text-2xl font-bold">
                              w = {result.w}
                            </h5>
                          </div>

                          <div className="text-right">
                            <p className="text-sm font-semibold text-emerald-400">
                              {result.matches}/{result.items.length} matches
                            </p>

                            <p className="mt-1 text-xs text-slate-500">
                              {accuracy.toFixed(1)}% classifier match rate
                            </p>
                          </div>
                        </div>

                        {/* ACCURACY BAR */}
                        <div className="mb-5 h-2 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-500"
                            style={{ width: `${accuracy}%` }}
                          />
                        </div>

                        {/* IMAGE GRID */}
                        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8">
                          {result.items.map((item, index) => (
                            <ImageCard
                              key={`${result.w}-${index}`}
                              item={item}
                              index={index}
                              selectedDigit={digit}
                            />
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* COMPARISON SUMMARY */}
                <div className="mt-6 rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-5">
                  <h4 className="font-semibold">
                    Comparison Summary
                  </h4>

                  <p className="mt-2 text-sm leading-6 text-slate-400">
                    Classifier match rate indicates how often the separate CNN
                    classifier predicts the requested digit. It is a useful
                    consistency check, not a complete measure of generated
                    image quality.
                  </p>

                  <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                    {compareResults.map((result) => (
                      <div
                        key={result.w}
                        className="rounded-xl border border-white/10 bg-black/20 p-3"
                      >
                        <p className="text-xs text-slate-500">
                          Guidance {result.w}
                        </p>

                        <p className="mt-1 text-xl font-semibold">
                          {result.matches}/{result.items.length}
                        </p>

                        <p className="text-xs text-slate-500">
                          Correct classifier predictions
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {!compareLoading && compareResults.length === 0 && (
              <div className="mt-6 rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-2xl text-violet-300">
                  ◈
                </div>

                <h4 className="font-medium">
                  Ready to compare guidance
                </h4>

                <p className="mt-2 text-sm text-slate-500">
                  Run the experiment to see how guidance strength affects
                  handwritten digit generation.
                </p>
              </div>
            )}
          </section>

          {/* FOOTER */}
          <footer className="mt-8 border-t border-white/[0.07] py-6 text-xs text-slate-600">
            DiffusionLab · PyTorch + FastAPI + Next.js
          </footer>

        </section>
      </div>
    </main>
  );
}
