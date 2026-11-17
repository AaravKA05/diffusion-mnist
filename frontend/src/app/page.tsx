
"use client";

import { useEffect, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const DB_NAME = "DiffusionLabDB";
const STORE_NAME = "generations";

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

type GenerationRecord = {
  id: string;
  createdAt: string;
  digit: number;
  count: number;
  guidance: number;
  steps: number;
  seed: number;
  elapsed: number;
  matches: number;
  items: GeneratedItem[];
};

type EvaluationResult = {
  matches: number;
  class_consistency: number;
  mean_confidence: number;
  diversity_mean_pairwise_l1: number;
  mean_pixel_std: number;
  seconds: number;
  images_per_second: number;
  items: GeneratedItem[];
  params: {
    digit: number;
    count: number;
    w: number;
    steps: number;
    seed: number;
  };
};

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveGeneration(record: GenerationRecord) {
  const db = await openDatabase();

  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put(record);

    transaction.oncomplete = () => {
      db.close();
      resolve();
    };

    transaction.onerror = () => {
      db.close();
      reject(transaction.error);
    };
  });
}

async function getGenerations(): Promise<GenerationRecord[]> {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readonly");
    const request = transaction.objectStore(STORE_NAME).getAll();

    request.onsuccess = () => {
      const records = request.result as GenerationRecord[];
      records.sort(
        (a, b) =>
          new Date(b.createdAt).getTime() -
          new Date(a.createdAt).getTime()
      );
      db.close();
      resolve(records);
    };

    request.onerror = () => {
      db.close();
      reject(request.error);
    };
  });
}

async function deleteGeneration(id: string) {
  const db = await openDatabase();

  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).delete(id);

    transaction.oncomplete = () => {
      db.close();
      resolve();
    };

    transaction.onerror = () => {
      db.close();
      reject(transaction.error);
    };
  });
}

async function clearGenerations() {
  const db = await openDatabase();

  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).clear();

    transaction.oncomplete = () => {
      db.close();
      resolve();
    };

    transaction.onerror = () => {
      db.close();
      reject(transaction.error);
    };
  });
}

function downloadImage(image: string, filename: string) {
  const binary = atob(image);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  const blob = new Blob([bytes], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ImageCard({
  item,
  index,
  selectedDigit,
  onDownload,
}: {
  item: GeneratedItem;
  index: number;
  selectedDigit: number;
  onDownload?: () => void;
}) {
  const correct = item.pred === selectedDigit;

  return (
    <div
      className={`overflow-hidden rounded-xl border-2 bg-black/40 p-2 ${correct ? "border-emerald-500/70" : "border-red-500/70"
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

      {onDownload && (
        <button
          onClick={onDownload}
          className="mt-2 w-full rounded-lg border border-white/10 bg-white/5 py-2 text-xs text-slate-300 hover:bg-white/10"
        >
          ↓ Download
        </button>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-white/[0.08] bg-white/[0.035] p-5">
      <p className="text-[10px] font-semibold tracking-[0.2em] text-slate-500">
        {label}
      </p>
      <h3 className="mt-3 text-2xl font-semibold">{value}</h3>
      <p className="mt-1 text-xs text-slate-500">{description}</p>
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("Studio");

  const [digit, setDigit] = useState(7);
  const [count, setCount] = useState(8);
  const [guidance, setGuidance] = useState(1.2);
  const [steps, setSteps] = useState(50);
  const [seed, setSeed] = useState(42);

  const [images, setImages] = useState<GeneratedItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [matches, setMatches] = useState<number | null>(null);

  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [compareTime, setCompareTime] = useState<number | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");

  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendDevice, setBackendDevice] = useState("Checking...");
  const [imageSize, setImageSize] = useState<number | null>(null);

  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  // EVALUATION STATES
  const [evalDigit, setEvalDigit] = useState(7);
  const [evalCount, setEvalCount] = useState(8);
  const [evalGuidance, setEvalGuidance] = useState(1.2);
  const [evalSteps, setEvalSteps] = useState(20);
  const [evalSeed, setEvalSeed] = useState(42);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");

  // LOAD HISTORY
  useEffect(() => {
    let mounted = true;

    async function loadHistory() {
      try {
        const records = await getGenerations();
        if (mounted) setHistory(records);
      } catch {
        if (mounted) {
          setHistoryError("Could not load saved generations.");
        }
      } finally {
        if (mounted) setHistoryLoading(false);
      }
    }

    loadHistory();

    return () => {
      mounted = false;
    };
  }, []);

  // BACKEND HEALTH MONITOR
  useEffect(() => {
    let mounted = true;

    async function checkBackend() {
      try {
        const response = await fetch(`${API}/api/info`);

        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (mounted) {
          setBackendOnline(true);
          setBackendDevice(data.device || "Unknown");
          setImageSize(data.img_size ?? null);
        }
      } catch {
        if (mounted) {
          setBackendOnline(false);
          setBackendDevice("Unavailable");
          setImageSize(null);
        }
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 10000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // GENERATE IMAGES
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

      const record: GenerationRecord = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        createdAt: new Date().toISOString(),
        digit,
        count,
        guidance,
        steps,
        seed,
        elapsed: data.seconds,
        matches: data.matches,
        items: data.items,
      };

      try {
        await saveGeneration(record);
        setHistory((previous) => [record, ...previous]);
      } catch {
        setStatus(
          "Generation completed, but browser storage could not save the result."
        );
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to backend."
      );
      setStatus("");
    } finally {
      setLoading(false);
    }
  }

  // GUIDANCE COMPARISON
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
      setStatus("Guidance comparison completed.");
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

  // RUN EVALUATION
  async function runEvaluation() {
    setEvaluationLoading(true);
    setEvaluationError("");
    setEvaluation(null);

    try {
      const response = await fetch(`${API}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit: evalDigit,
          count: evalCount,
          w: evalGuidance,
          steps: evalSteps,
          seed: evalSeed,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(`Evaluation failed (${response.status}): ${message}`);
      }

      const data: EvaluationResult = await response.json();
      setEvaluation(data);
    } catch (err) {
      setEvaluationError(
        err instanceof Error ? err.message : "Evaluation failed."
      );
    } finally {
      setEvaluationLoading(false);
    }
  }

  // RESTORE SETTINGS
  function reuseSettings(record: GenerationRecord) {
    setDigit(record.digit);
    setCount(record.count);
    setGuidance(record.guidance);
    setSteps(record.steps);
    setSeed(record.seed);
    setActiveTab("Studio");
    setStatus("Previous generation settings restored.");
  }

  // DELETE ONE RECORD
  async function removeRecord(id: string) {
    try {
      await deleteGeneration(id);
      setHistory((previous) => previous.filter((item) => item.id !== id));
      setHistoryError("");
    } catch {
      setHistoryError("Could not delete this generation.");
    }
  }

  // DELETE ALL RECORDS
  async function removeAllRecords() {
    if (!window.confirm("Delete all saved generations from this browser?")) {
      return;
    }

    try {
      await clearGenerations();
      setHistory([]);
      setHistoryError("");
    } catch {
      setHistoryError("Could not clear generation history.");
    }
  }

  // ACTIVITY STATISTICS
  const totalImages = history.reduce(
    (total, record) => total + record.items.length,
    0
  );

  const averageTime =
    history.length > 0
      ? history.reduce((total, record) => total + record.elapsed, 0) /
      history.length
      : 0;

  const totalMatches = history.reduce(
    (total, record) => total + record.matches,
    0
  );

  const matchRate =
    totalImages > 0 ? (totalMatches / totalImages) * 100 : 0;

  const tabs = [
    ["◈", "Studio"],
    ["▦", "My Generations"],
    ["◉", "Model Insights"],
    ["◎", "Evaluation"],
  ];

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
            {tabs.map(([icon, name]) => (
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
              <span
                className={`h-2 w-2 rounded-full ${backendOnline === true
                  ? "bg-emerald-400"
                  : backendOnline === false
                    ? "bg-red-400"
                    : "bg-amber-400"
                  }`}
              />
              <span className="text-xs text-slate-300">
                {backendOnline === true
                  ? "API Connected"
                  : backendOnline === false
                    ? "API Offline"
                    : "Checking API..."}
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
          {/* MOBILE NAVIGATION */}
          <div className="mb-6 flex gap-2 overflow-x-auto lg:hidden">
            {tabs.map(([icon, name]) => (
              <button
                key={name}
                onClick={() => setActiveTab(name)}
                className={`shrink-0 rounded-xl border px-3 py-2 text-xs ${activeTab === name
                  ? "border-violet-400/30 bg-violet-500/15 text-white"
                  : "border-white/10 bg-white/5 text-slate-400"
                  }`}
              >
                {icon} {name}
              </button>
            ))}
          </div>

          {/* HEADER */}
          <header className="mb-10 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="mb-2 text-xs tracking-widest text-violet-300">
                YOUR GENERATIVE WORKSPACE
              </p>
              <h2 className="text-3xl font-semibold sm:text-4xl">
                {activeTab === "Studio"
                  ? "Creative Studio"
                  : activeTab === "My Generations"
                    ? "My Generations"
                    : activeTab === "Evaluation"
                      ? "Model Evaluation"
                      : "Model Insights"}
                <span className="text-violet-400">.</span>
              </h2>
              <p className="mt-2 text-sm text-slate-500">
                {activeTab === "Studio"
                  ? "Explore handwritten digit generation with diffusion models."
                  : activeTab === "My Generations"
                    ? "Your previously generated samples, saved in this browser."
                    : activeTab === "Evaluation"
                      ? "Evaluate class consistency, confidence, diversity and inference speed."
                      : "A summary of your generation activity and model configuration."}
              </p>
            </div>

            <div className="flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.04] px-4 py-2.5">
              <span
                className={`h-2 w-2 rounded-full ${backendOnline === true
                  ? "bg-emerald-400"
                  : backendOnline === false
                    ? "bg-red-400"
                    : "bg-amber-400"
                  }`}
              />
              <span className="text-xs text-slate-300">
                {backendOnline === true
                  ? `Backend Online · ${backendDevice.toUpperCase()}`
                  : backendOnline === false
                    ? "Backend Offline"
                    : "Checking Backend..."}
              </span>
            </div>
          </header>

          {/* STUDIO */}
          {activeTab === "Studio" && (
            <>
              <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
                <StatCard
                  label="MODEL"
                  value="Conditional DDPM"
                  description="PyTorch architecture"
                />
                <StatCard
                  label="SAMPLING"
                  value="DDIM"
                  description="Configurable inference"
                />
                <StatCard
                  label="CLASSES"
                  value="10 Digits"
                  description="Labels 0 through 9"
                />
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_0.85fr]">
                {/* CONTROLS */}
                <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-violet-300">
                    01 / CONFIGURATION
                  </p>
                  <h3 className="mt-2 text-xl font-semibold">
                    Generation Controls
                  </h3>

                  <div className="mb-8 mt-7">
                    <div className="mb-4 flex justify-between text-sm">
                      <label className="text-slate-300">Select a digit</label>
                      <span className="text-violet-300">Selected: {digit}</span>
                    </div>

                    <div className="grid grid-cols-5 gap-3 sm:grid-cols-10">
                      {Array.from({ length: 10 }, (_, i) => (
                        <button
                          key={i}
                          onClick={() => setDigit(i)}
                          className={`aspect-square rounded-xl border text-lg font-semibold ${digit === i
                            ? "border-violet-400 bg-gradient-to-br from-violet-500 to-blue-600 text-white"
                            : "border-white/10 bg-white/[0.035] text-slate-400"
                            }`}
                        >
                          {i}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-7">
                    <div>
                      <div className="mb-3 flex justify-between text-sm">
                        <label>Number of samples</label>
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
                        <label>Guidance scale</label>
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
                        <label>DDIM sampling steps</label>
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

                    <div>
                      <label className="mb-3 block text-sm">Random seed</label>
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
                                Math.min(2147483647, Number(e.target.value) || 0)
                              )
                            )
                          }
                          className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none"
                        />
                        <button
                          onClick={() =>
                            setSeed(Math.floor(Math.random() * 100000))
                          }
                          className="rounded-xl border border-white/10 bg-white/5 px-4 text-xs"
                        >
                          ↻ Random
                        </button>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={generateImages}
                    disabled={loading || compareLoading || backendOnline !== true}
                    className="mt-9 w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold disabled:opacity-50"
                  >
                    {loading
                      ? "◌ Generating..."
                      : backendOnline === true
                        ? "✦ Generate Images"
                        : "Backend Unavailable"}
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

                {/* OUTPUT */}
                <div className="flex min-h-[480px] flex-col rounded-3xl border border-white/10 bg-[#111522]/70 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-cyan-300">
                    02 / OUTPUT
                  </p>
                  <h3 className="mb-6 mt-2 text-xl font-semibold">
                    Generation Preview
                  </h3>

                  {images.length === 0 ? (
                    <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                      <div className="mb-6 flex h-24 w-24 items-center justify-center rounded-3xl border border-violet-400/20 bg-violet-500/10 text-5xl text-violet-300">
                        {digit}
                      </div>
                      <h4 className="text-lg font-medium">
                        Your canvas is ready
                      </h4>
                      <p className="mt-2 max-w-xs text-sm text-slate-500">
                        Generate handwritten samples using your trained
                        diffusion model.
                      </p>
                    </div>
                  ) : (
                    <>
                      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {images.map((item, index) => (
                          <ImageCard
                            key={index}
                            item={item}
                            index={index}
                            selectedDigit={digit}
                            onDownload={() =>
                              downloadImage(
                                item.image,
                                `digit-${digit}-sample-${index + 1}.png`
                              )
                            }
                          />
                        ))}
                      </div>

                      <div className="mb-4 grid grid-cols-2 gap-3">
                        <StatCard
                          label="CLASSIFIER MATCHES"
                          value={`${matches}/${images.length}`}
                          description="Predictions matching requested digit"
                        />
                        <StatCard
                          label="INFERENCE TIME"
                          value={`${elapsed} s`}
                          description="Generation time"
                        />
                      </div>

                      <button
                        onClick={() =>
                          images.forEach((item, index) =>
                            setTimeout(
                              () =>
                                downloadImage(
                                  item.image,
                                  `digit-${digit}-sample-${index + 1}.png`
                                ),
                              index * 250
                            )
                          )
                        }
                        className="mt-auto w-full rounded-xl border border-violet-400/20 bg-violet-500/10 py-3 text-sm text-violet-200"
                      >
                        ↓ Download All Images
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* GUIDANCE COMPARISON */}
              <section className="mt-8 rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                <p className="text-xs tracking-widest text-cyan-300">
                  03 / MODEL EXPERIMENTATION
                </p>
                <h3 className="mt-2 text-2xl font-semibold">
                  Guidance Comparison
                </h3>
                <p className="mt-2 text-sm text-slate-400">
                  Compare how classifier-free guidance changes generated
                  digits.
                </p>

                <div className="my-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <StatCard
                    label="GUIDANCE SCALES"
                    value="0.0 · 1.2 · 2.0"
                    description="Three configurations"
                  />
                  <StatCard
                    label="STARTING NOISE"
                    value={`Seed ${seed}`}
                    description="Same seed across experiments"
                  />
                  <StatCard
                    label="DDIM STEPS"
                    value={String(steps)}
                    description="Sampling steps"
                  />
                </div>

                <p className="mb-5 text-sm text-slate-400">
                  Digit {digit} · {Math.min(count, 8)} samples per guidance
                  scale
                </p>

                <button
                  onClick={compareGuidance}
                  disabled={compareLoading || loading || backendOnline !== true}
                  className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold disabled:opacity-50"
                >
                  {compareLoading
                    ? "◌ Comparing..."
                    : "✦ Compare Guidance — 0.0 vs 1.2 vs 2.0"}
                </button>

                {compareLoading && (
                  <p className="mt-5 text-center text-sm text-violet-300">
                    Running three sampling experiments...
                  </p>
                )}

                {compareResults.length > 0 && (
                  <div className="mt-8 space-y-8">
                    <p className="text-sm text-slate-400">
                      Total inference time: {compareTime} seconds
                    </p>

                    {compareResults.map((result) => {
                      const accuracy =
                        result.items.length > 0
                          ? (result.matches / result.items.length) * 100
                          : 0;

                      return (
                        <div
                          key={result.w}
                          className="rounded-2xl border border-white/10 bg-black/20 p-5"
                        >
                          <div className="mb-4 flex justify-between">
                            <h4 className="text-xl font-bold">
                              Guidance w = {result.w}
                            </h4>
                            <span className="text-emerald-400">
                              {result.matches}/{result.items.length} matches
                            </span>
                          </div>

                          <div className="mb-5 h-2 overflow-hidden rounded-full bg-white/10">
                            <div
                              className="h-full bg-gradient-to-r from-blue-500 to-violet-500"
                              style={{ width: `${accuracy}%` }}
                            />
                          </div>

                          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
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

                    <p className="text-sm leading-6 text-slate-400">
                      Classifier match rate measures how often the separate
                      CNN classifier predicts the requested digit. It is a
                      consistency indicator, not a complete image quality
                      metric.
                    </p>
                  </div>
                )}
              </section>
            </>
          )}

          {/* MY GENERATIONS */}
          {activeTab === "My Generations" && (
            <section>
              <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-xl font-semibold">Generation History</h3>
                  <p className="mt-1 text-sm text-slate-500">
                    {history.length} saved generations · {totalImages} images
                  </p>
                </div>

                {history.length > 0 && (
                  <button
                    onClick={removeAllRecords}
                    className="rounded-xl border border-red-400/20 bg-red-500/10 px-4 py-2.5 text-xs text-red-300"
                  >
                    Delete All History
                  </button>
                )}
              </div>

              {historyError && (
                <p className="mb-4 rounded-xl bg-red-500/10 p-3 text-sm text-red-300">
                  {historyError}
                </p>
              )}

              {historyLoading ? (
                <div className="rounded-2xl border border-white/10 p-10 text-center text-slate-400">
                  Loading saved generations...
                </div>
              ) : history.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-white/10 p-12 text-center">
                  <h3 className="text-lg font-semibold">
                    No generations saved yet
                  </h3>
                  <p className="mt-2 text-sm text-slate-500">
                    Generate samples in Studio to start building your history.
                  </p>
                  <button
                    onClick={() => setActiveTab("Studio")}
                    className="mt-6 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-3 text-sm font-semibold"
                  >
                    Open Studio
                  </button>
                </div>
              ) : (
                <div className="space-y-5">
                  {history.map((record) => (
                    <div
                      key={record.id}
                      className="rounded-3xl border border-white/10 bg-[#111522]/80 p-5 sm:p-6"
                    >
                      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
                        <div>
                          <h4 className="font-semibold">
                            Digit {record.digit} · {record.items.length} samples
                          </h4>
                          <p className="mt-1 text-xs text-slate-500">
                            {new Date(record.createdAt).toLocaleString()}
                          </p>
                        </div>

                        <div className="flex flex-wrap gap-2">
                          <button
                            onClick={() => reuseSettings(record)}
                            className="rounded-lg border border-violet-400/20 bg-violet-500/10 px-3 py-2 text-xs text-violet-200"
                          >
                            ↻ Reuse Settings
                          </button>

                          <button
                            onClick={() =>
                              record.items.forEach((item, index) =>
                                setTimeout(
                                  () =>
                                    downloadImage(
                                      item.image,
                                      `digit-${record.digit}-sample-${index + 1}.png`
                                    ),
                                  index * 250
                                )
                              )
                            }
                            className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs"
                          >
                            ↓ Download All
                          </button>

                          <button
                            onClick={() => removeRecord(record.id)}
                            className="rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-xs text-red-300"
                          >
                            Delete
                          </button>
                        </div>
                      </div>

                      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <StatCard
                          label="GUIDANCE"
                          value={record.guidance.toFixed(1)}
                          description="Guidance scale"
                        />
                        <StatCard
                          label="DDIM STEPS"
                          value={String(record.steps)}
                          description="Sampling steps"
                        />
                        <StatCard
                          label="SEED"
                          value={String(record.seed)}
                          description="Random seed"
                        />
                        <StatCard
                          label="INFERENCE"
                          value={`${record.elapsed} s`}
                          description="Generation time"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
                        {record.items.map((item, index) => (
                          <ImageCard
                            key={`${record.id}-${index}`}
                            item={item}
                            index={index}
                            selectedDigit={record.digit}
                            onDownload={() =>
                              downloadImage(
                                item.image,
                                `digit-${record.digit}-sample-${index + 1}.png`
                              )
                            }
                          />
                        ))}
                      </div>

                      <p className="mt-4 text-xs text-slate-500">
                        Classifier matches: {record.matches}/
                        {record.items.length}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* MODEL INSIGHTS */}
          {activeTab === "Model Insights" && (
            <section>
              <div className="mb-6">
                <h3 className="text-xl font-semibold">Your Activity</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Statistics calculated from your saved generation history.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  label="TOTAL GENERATIONS"
                  value={String(history.length)}
                  description="Saved sessions"
                />
                <StatCard
                  label="IMAGES GENERATED"
                  value={String(totalImages)}
                  description="Total saved samples"
                />
                <StatCard
                  label="AVG. INFERENCE TIME"
                  value={`${averageTime.toFixed(2)} s`}
                  description="Per generation session"
                />
                <StatCard
                  label="CLASSIFIER MATCH RATE"
                  value={`${matchRate.toFixed(1)}%`}
                  description={`${totalMatches}/${totalImages} predictions matched`}
                />
              </div>

              <div className="mt-8 grid grid-cols-1 gap-6 xl:grid-cols-2">
                <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-violet-300">
                    MODEL CONFIGURATION
                  </p>
                  <h3 className="mt-2 text-xl font-semibold">
                    Architecture Overview
                  </h3>

                  <div className="mt-6 space-y-4">
                    {[
                      ["Model", "Conditional DDPM"],
                      ["EMA checkpoint", "cond_ema.pt"],
                      ["Sampling method", "DDIM"],
                      ["Dataset", "MNIST"],
                      [
                        "Image resolution",
                        imageSize ? `${imageSize} × ${imageSize}` : "32 × 32",
                      ],
                      ["Output classes", "0–9"],
                      ["Inference device", backendDevice],
                      ["Backend", "FastAPI + PyTorch"],
                      ["Frontend", "Next.js + React"],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        className="flex justify-between gap-3 border-b border-white/[0.06] pb-3"
                      >
                        <span className="text-sm text-slate-500">{label}</span>
                        <span className="text-right text-sm font-medium">
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-cyan-300">
                    HOW TO INTERPRET
                  </p>
                  <h3 className="mt-2 text-xl font-semibold">
                    Performance Metrics
                  </h3>

                  <div className="mt-6 space-y-5">
                    <div>
                      <h4 className="font-medium text-emerald-300">
                        Classifier Match Rate
                      </h4>
                      <p className="mt-2 text-sm leading-6 text-slate-400">
                        The percentage of generated samples that the separate
                        CNN classifier identifies as the requested digit. This
                        is a consistency indicator, not a complete image
                        quality metric.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-medium text-violet-300">
                        Inference Time
                      </h4>
                      <p className="mt-2 text-sm leading-6 text-slate-400">
                        Time reported by the backend for generation. It can
                        vary with sample count, DDIM steps, hardware and system
                        load.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-medium text-cyan-300">
                        Guidance Scale
                      </h4>
                      <p className="mt-2 text-sm leading-6 text-slate-400">
                        Controls how strongly the conditional label influences
                        the sampling process. Its effect should be evaluated
                        experimentally.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-8 rounded-3xl border border-violet-400/20 bg-violet-500/[0.06] p-6">
                <h3 className="font-semibold">Your Experiment Activity</h3>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  {history.length === 0
                    ? "No generation activity is available yet. Generate samples in Studio to start building your activity statistics."
                    : `You have saved ${history.length} generation sessions containing ${totalImages} images. Your average classifier match rate across these saved samples is ${matchRate.toFixed(1)}%.`}
                </p>
              </div>
            </section>
          )}

          {/* EVALUATION */}
          {activeTab === "Evaluation" && (
            <section>
              <div className="mb-6">
                <h3 className="text-xl font-semibold">
                  Diffusion Model Evaluation
                </h3>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  Generate a batch of samples and evaluate class consistency,
                  classifier confidence, diversity and inference performance.
                </p>
              </div>

              {/* EVALUATION CONTROLS */}
              <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                <p className="text-xs tracking-widest text-violet-300">
                  01 / EXPERIMENT CONFIGURATION
                </p>
                <h3 className="mt-2 text-xl font-semibold">
                  Evaluation Parameters
                </h3>

                <div className="mt-7 grid grid-cols-1 gap-6 sm:grid-cols-2">
                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Target digit
                    </label>
                    <select
                      value={evalDigit}
                      onChange={(e) => setEvalDigit(Number(e.target.value))}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] px-4 py-3 text-sm"
                    >
                      {Array.from({ length: 10 }, (_, i) => (
                        <option key={i} value={i}>
                          Digit {i}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Number of samples: {evalCount}
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="64"
                      step="1"
                      value={evalCount}
                      onChange={(e) => setEvalCount(Number(e.target.value))}
                      className="w-full accent-violet-500"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Guidance scale: {evalGuidance.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="3"
                      step="0.1"
                      value={evalGuidance}
                      onChange={(e) =>
                        setEvalGuidance(Number(e.target.value))
                      }
                      className="w-full accent-violet-500"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      DDIM steps: {evalSteps}
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      step="5"
                      value={evalSteps}
                      onChange={(e) => setEvalSteps(Number(e.target.value))}
                      className="w-full accent-violet-500"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Random seed
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      value={evalSeed}
                      onChange={(e) =>
                        setEvalSeed(
                          Math.max(
                            0,
                            Math.min(2147483647, Number(e.target.value) || 0)
                          )
                        )
                      }
                      className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm"
                    />
                  </div>
                </div>

                <button
                  onClick={runEvaluation}
                  disabled={evaluationLoading || backendOnline !== true}
                  className="mt-8 w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {evaluationLoading
                    ? "◌ Evaluating model..."
                    : backendOnline !== true
                      ? "Backend Unavailable"
                      : "✦ Run Model Evaluation"}
                </button>

                {evaluationError && (
                  <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
                    {evaluationError}
                  </div>
                )}
              </div>

              {/* EVALUATION RESULTS */}
              {evaluation && (
                <div className="mt-8 space-y-6">
                  <div>
                    <p className="text-xs tracking-widest text-cyan-300">
                      02 / EVALUATION RESULTS
                    </p>
                    <h3 className="mt-2 text-2xl font-semibold">
                      Performance Overview
                    </h3>
                    <p className="mt-2 text-sm text-slate-400">
                      Evaluation for digit {evaluation.params.digit} using{" "}
                      {evaluation.params.count} generated samples.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    <StatCard
                      label="CLASS CONSISTENCY"
                      value={`${evaluation.class_consistency.toFixed(1)}%`}
                      description={`${evaluation.matches}/${evaluation.items.length} classifier matches`}
                    />
                    <StatCard
                      label="MEAN CONFIDENCE"
                      value={`${(evaluation.mean_confidence * 100).toFixed(2)}%`}
                      description="Average classifier confidence"
                    />
                    <StatCard
                      label="DIVERSITY SCORE"
                      value={evaluation.diversity_mean_pairwise_l1.toFixed(4)}
                      description="Mean pairwise pixel L1 distance"
                    />
                    <StatCard
                      label="PIXEL VARIATION"
                      value={evaluation.mean_pixel_std.toFixed(4)}
                      description="Mean pixel standard deviation"
                    />
                    <StatCard
                      label="INFERENCE TIME"
                      value={`${evaluation.seconds.toFixed(2)} s`}
                      description="Total evaluation time"
                    />
                    <StatCard
                      label="THROUGHPUT"
                      value={`${evaluation.images_per_second.toFixed(2)}`}
                      description="Images generated per second"
                    />
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-[#111522]/80 p-5">
                    <h4 className="font-semibold">Evaluation Configuration</h4>
                    <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-5">
                      {[
                        ["Digit", evaluation.params.digit],
                        ["Samples", evaluation.params.count],
                        ["Guidance", evaluation.params.w],
                        ["DDIM Steps", evaluation.params.steps],
                        ["Seed", evaluation.params.seed],
                      ].map(([label, value]) => (
                        <div
                          key={String(label)}
                          className="rounded-xl border border-white/[0.07] bg-black/20 p-3"
                        >
                          <p className="text-xs text-slate-500">{label}</p>
                          <p className="mt-1 font-semibold">{value}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6">
                    <h4 className="text-lg font-semibold">
                      Generated Evaluation Samples
                    </h4>
                    <p className="mt-1 text-sm text-slate-500">
                      Green border indicates the classifier predicted the
                      requested digit. Red border indicates a mismatch.
                    </p>

                    <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
                      {evaluation.items.map((item, index) => (
                        <ImageCard
                          key={`evaluation-${index}`}
                          item={item}
                          index={index}
                          selectedDigit={evaluation.params.digit}
                          onDownload={() =>
                            downloadImage(
                              item.image,
                              `evaluation-digit-${evaluation.params.digit}-${index + 1}.png`
                            )
                          }
                        />
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-5">
                    <h4 className="font-semibold">How to interpret results</h4>
                    <div className="mt-3 space-y-3 text-sm leading-6 text-slate-400">
                      <p>
                        <strong className="text-emerald-300">
                          Class consistency:
                        </strong>{" "}
                        Percentage of samples classified as the requested
                        digit by the separate CNN classifier.
                      </p>
                      <p>
                        <strong className="text-violet-300">
                          Mean confidence:
                        </strong>{" "}
                        Average confidence reported by that classifier. It
                        does not directly measure generator accuracy.
                      </p>
                      <p>
                        <strong className="text-cyan-300">
                          Diversity:
                        </strong>{" "}
                        Mean pairwise pixel-space L1 distance. It is a simple
                        diversity indicator, not a perceptual metric or FID.
                      </p>
                      <p>
                        <strong className="text-violet-300">
                          Throughput:
                        </strong>{" "}
                        Number of images generated per second during this
                        evaluation run.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* FOOTER */}
          <footer className="mt-10 border-t border-white/[0.07] py-6 text-xs text-slate-600">
            DiffusionLab · PyTorch + FastAPI + Next.js
          </footer>
        </section>
      </div>
    </main>
  );
}
