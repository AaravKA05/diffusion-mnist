
"use client";

import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";
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

function downloadGeneration(record: GenerationRecord) {
  record.items.forEach((item, index) => {
    setTimeout(() => {
      downloadImage(
        item.image,
        `digit-${record.digit}-sample-${index + 1}.png`
      );
    }, index * 250);
  });
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
          className="mt-2 w-full rounded-lg border border-white/10 bg-white/5 py-2 text-xs text-slate-300 transition hover:bg-white/10"
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

  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");

  // BACKEND HEALTH MONITORING
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendDevice, setBackendDevice] = useState("Checking...");
  const [imageSize, setImageSize] = useState<number | null>(null);

  // LOAD SAVED HISTORY
  useEffect(() => {
    let mounted = true;

    async function loadHistory() {
      try {
        const records = await getGenerations();

        if (mounted) {
          setHistory(records);
        }
      } catch {
        if (mounted) {
          setHistoryError(
            "Could not load saved generations from this browser."
          );
        }
      } finally {
        if (mounted) {
          setHistoryLoading(false);
        }
      }
    }

    loadHistory();

    return () => {
      mounted = false;
    };
  }, []);

  // CHECK BACKEND EVERY 10 SECONDS
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
          "Generation completed, but browser storage could not save this result."
        );
      }
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

  // COMPARE GUIDANCE
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
      setHistory((previous) =>
        previous.filter((item) => item.id !== id)
      );
      setHistoryError("");
    } catch {
      setHistoryError("Could not delete this generation.");
    }
  }

  // DELETE ALL RECORDS
  async function removeAllRecords() {
    if (
      !window.confirm(
        "Delete all saved generations from this browser?"
      )
    ) {
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

  // INSIGHTS CALCULATIONS
  const totalImages = history.reduce(
    (total, record) => total + record.items.length,
    0
  );

  const averageTime =
    history.length > 0
      ? history.reduce((total, record) => total + record.elapsed, 0) /
      history.length
      : 0;

  const totalSamples = history.reduce(
    (total, record) => total + record.items.length,
    0
  );

  const totalMatches = history.reduce(
    (total, record) => total + record.matches,
    0
  );

  const matchRate =
    totalSamples > 0 ? (totalMatches / totalSamples) * 100 : 0;

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
              <span
                className={`h-2 w-2 rounded-full ${backendOnline === true
                  ? "bg-emerald-400"
                  : backendOnline === false
                    ? "bg-red-400"
                    : "bg-amber-400"
                  }`}
              />

              <span
                className={`text-xs ${backendOnline === true
                  ? "text-emerald-300"
                  : backendOnline === false
                    ? "text-red-300"
                    : "text-amber-300"
                  }`}
              >
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
                    : "Model Insights"}
                <span className="text-violet-400">.</span>
              </h2>

              <p className="mt-2 text-sm text-slate-500">
                {activeTab === "Studio"
                  ? "Explore handwritten digit generation with diffusion models."
                  : activeTab === "My Generations"
                    ? "Your previously generated samples, saved in this browser."
                    : "A summary of your generation activity and model configuration."}
              </p>
            </div>

            {/* LIVE BACKEND STATUS */}
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
              {/* MODEL STATS */}
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
                        onChange={(e) =>
                          setGuidance(Number(e.target.value))
                        }
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
                    disabled={
                      loading || compareLoading || backendOnline !== true
                    }
                    className="mt-9 w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {loading
                      ? "◌ Generating..."
                      : backendOnline === false
                        ? "Backend Offline"
                        : backendOnline === null
                          ? "Checking Backend..."
                          : "✦ Generate Images"}
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
                        Generate handwritten samples using your trained
                        diffusion model.
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
                        className="mt-auto w-full rounded-xl border border-violet-400/20 bg-violet-500/10 py-3 text-sm font-medium text-violet-200 transition hover:bg-violet-500/20"
                      >
                        ↓ Download All Images
                      </button>
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
                    Compare how classifier-free guidance changes the generated
                    digits.
                  </p>
                </div>

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
                    Digit {digit} · {Math.min(count, 8)} samples per guidance
                    scale
                  </p>

                  <span className="text-xs text-slate-500">
                    3 guidance settings
                  </span>
                </div>

                <button
                  onClick={compareGuidance}
                  disabled={
                    compareLoading || loading || backendOnline !== true
                  }
                  className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {compareLoading
                    ? "◌ Comparing guidance scales..."
                    : "✦ Compare Guidance — 0.0 vs 1.2 vs 2.0"}
                </button>

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

                {compareResults.length > 0 && (
                  <div className="mt-8">
                    <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h4 className="text-lg font-semibold">
                          Experiment Results
                        </h4>
                        <p className="mt-1 text-xs text-slate-500">
                          Same digit, seed and DDIM steps across all
                          experiments.
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

                            <div className="mb-5 h-2 overflow-hidden rounded-full bg-white/10">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-500"
                                style={{ width: `${accuracy}%` }}
                              />
                            </div>

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

                    <div className="mt-6 rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-5">
                      <h4 className="font-semibold">Comparison Summary</h4>
                      <p className="mt-2 text-sm leading-6 text-slate-400">
                        Classifier match rate indicates how often the separate
                        CNN classifier predicts the requested digit. It is a
                        useful consistency check, not a complete measure of
                        generated image quality.
                      </p>
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
            </>
          )}

          {/* MY GENERATIONS */}
          {activeTab === "My Generations" && (
            <section>
              <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-xl font-semibold">
                    Generation History
                  </h3>
                  <p className="mt-1 text-sm text-slate-500">
                    {history.length} saved generations · {totalImages} images
                  </p>
                </div>

                {history.length > 0 && (
                  <button
                    onClick={removeAllRecords}
                    className="rounded-xl border border-red-400/20 bg-red-500/10 px-4 py-2.5 text-xs text-red-300 transition hover:bg-red-500/20"
                  >
                    Delete All History
                  </button>
                )}
              </div>

              {historyError && (
                <p className="mb-4 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
                  {historyError}
                </p>
              )}

              {historyLoading ? (
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-10 text-center text-slate-400">
                  Loading saved generations...
                </div>
              ) : history.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.025] p-12 text-center">
                  <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-3xl text-violet-300">
                    ▦
                  </div>
                  <h3 className="text-lg font-semibold">
                    No generations saved yet
                  </h3>
                  <p className="mx-auto mt-2 max-w-sm text-sm text-slate-500">
                    Your generated samples will automatically appear here.
                    Head to Studio and generate your first batch.
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
                        <div className="flex items-center gap-3">
                          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/15 text-xl font-bold text-violet-300">
                            {record.digit}
                          </span>

                          <div>
                            <h4 className="font-semibold">
                              Digit {record.digit} · {record.items.length}{" "}
                              samples
                            </h4>
                            <p className="mt-1 text-xs text-slate-500">
                              {new Date(record.createdAt).toLocaleString()}
                            </p>
                          </div>
                        </div>

                        <div className="flex flex-wrap gap-2">
                          <button
                            onClick={() => reuseSettings(record)}
                            className="rounded-lg border border-violet-400/20 bg-violet-500/10 px-3 py-2 text-xs text-violet-200 hover:bg-violet-500/20"
                          >
                            ↻ Reuse Settings
                          </button>

                          <button
                            onClick={() => downloadGeneration(record)}
                            className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300 hover:bg-white/10"
                          >
                            ↓ Download All
                          </button>

                          <button
                            onClick={() => removeRecord(record.id)}
                            className="rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-xs text-red-300 hover:bg-red-500/20"
                          >
                            Delete
                          </button>
                        </div>
                      </div>

                      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                        {[
                          ["Guidance", record.guidance.toFixed(1)],
                          ["DDIM Steps", String(record.steps)],
                          ["Seed", String(record.seed)],
                          ["Inference", `${record.elapsed} s`],
                        ].map(([label, value]) => (
                          <div
                            key={label}
                            className="rounded-xl border border-white/[0.07] bg-black/20 p-3"
                          >
                            <p className="text-[10px] text-slate-500">
                              {label}
                            </p>
                            <p className="mt-1 text-sm font-semibold">
                              {value}
                            </p>
                          </div>
                        ))}
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
                  description="Saved generation sessions"
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
                  description={`${totalMatches}/${totalSamples} predictions matched`}
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
                        className="flex items-center justify-between gap-3 border-b border-white/[0.06] pb-3"
                      >
                        <span className="text-sm text-slate-500">
                          {label}
                        </span>
                        <span className="text-right text-sm font-medium text-slate-200">
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
                        The time reported by the backend for a generation
                        request. It can vary with sample count, DDIM steps,
                        hardware and system load.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-medium text-cyan-300">
                        Guidance Scale
                      </h4>
                      <p className="mt-2 text-sm leading-6 text-slate-400">
                        Controls how strongly the conditional label influences
                        the sampling process. Its effect should be evaluated
                        experimentally rather than assuming higher is always
                        better.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-8 rounded-3xl border border-violet-400/20 bg-violet-500/[0.06] p-6">
                <h3 className="font-semibold">Your Experiment Activity</h3>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  {history.length === 0
                    ? "No generation activity is available yet. Generate some samples in Studio to start building your activity statistics."
                    : `You have saved ${history.length} generation sessions containing ${totalImages} images. Your average classifier match rate across these saved samples is ${matchRate.toFixed(1)}%.`}
                </p>
              </div>
            </section>
          )}

          {/* FOOTER */}
          <footer className="mt-8 border-t border-white/[0.07] py-6 text-xs text-slate-600">
            DiffusionLab · PyTorch + FastAPI + Next.js
          </footer>
        </section>
      </div>
    </main>
  );
}
