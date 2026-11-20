"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
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

type ExperimentRecord = {
  id: string;
  createdAt: string;
  experimentType: string;
  digit: number;
  count: number;
  values: number[];
  guidance: number;
  steps: number;
  seed: number;
  totalSeconds: number;
  results: any[];
};

type VisualizationFrame = {
  timestep: number;
  image: string;
};

type VisualizationResult = {
  digit: number;
  steps: number;
  guidance: number;
  seed: number;
  seconds: number;
  snapshots: VisualizationFrame[];
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
    const request = indexedDB.open(DB_NAME, 2);

    request.onupgradeneeded = () => {
      const db = request.result;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }

      if (!db.objectStoreNames.contains("experiments")) {
        db.createObjectStore("experiments", { keyPath: "id" });
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

async function saveExperiment(record: ExperimentRecord) {
  const db = await openDatabase();

  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction("experiments", "readwrite");
    transaction.objectStore("experiments").put(record);

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

async function getExperiments(): Promise<ExperimentRecord[]> {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction("experiments", "readonly");
    const request = transaction.objectStore("experiments").getAll();

    request.onsuccess = () => {
      db.close();
      resolve(
        (request.result as ExperimentRecord[]).sort(
          (a, b) =>
            new Date(b.createdAt).getTime() -
            new Date(a.createdAt).getTime()
        )
      );
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

  const [experimentType, setExperimentType] = useState("guidance");
  const [savedExperiments, setSavedExperiments] = useState<ExperimentRecord[]>([]);
  const [experimentValues, setExperimentValues] = useState("0, 1.2, 2");
  const [experimentResults, setExperimentResults] = useState<any[]>([]);
  const [experimentTime, setExperimentTime] = useState<number | null>(null);
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [experimentError, setExperimentError] = useState("");

  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");

  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendDevice, setBackendDevice] = useState("Checking...");
  const [imageSize, setImageSize] = useState<number | null>(null);

  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const [evalDigit, setEvalDigit] = useState(7);
  const [evalCount, setEvalCount] = useState(8);
  const [evalGuidance, setEvalGuidance] = useState(1.2);
  const [evalSteps, setEvalSteps] = useState(20);
  const [evalSeed, setEvalSeed] = useState(42);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");

  const [visualDigit, setVisualDigit] = useState(7);
  const [visualGuidance, setVisualGuidance] = useState(1.2);
  const [visualSteps, setVisualSteps] = useState(20);
  const [visualSeed, setVisualSeed] = useState(42);
  const [snapshotCount, setSnapshotCount] = useState(8);
  const [visualization, setVisualization] =
    useState<VisualizationResult | null>(null);
  const [visualizationLoading, setVisualizationLoading] = useState(false);
  const [visualizationError, setVisualizationError] = useState("");
  const [activeFrame, setActiveFrame] = useState(0);

  const [interpDigit, setInterpDigit] = useState(7);
  const [interpGuidance, setInterpGuidance] = useState(1.2);
  const [interpSteps, setInterpSteps] = useState(30);
  const [seedA, setSeedA] = useState(42);
  const [seedB, setSeedB] = useState(123);
  const [interpFrameCount, setInterpFrameCount] = useState(9);
  const [interpolation, setInterpolation] = useState<any>(null);
  const [interpolationLoading, setInterpolationLoading] = useState(false);
  const [interpolationError, setInterpolationError] = useState("");
  const [selectedInterpFrame, setSelectedInterpFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!isPlaying || !interpolation?.frames?.length) return;

    const timer = setInterval(() => {
      setSelectedInterpFrame((current) => {
        if (current >= interpolation.frames.length - 1) {
          setIsPlaying(false);
          return 0;
        }

        return current + 1;
      });
    }, 700);

    return () => clearInterval(timer);
  }, [isPlaying, interpolation]);

  useEffect(() => {
    let mounted = true;

    async function loadHistory() {
      try {
        const records = await getGenerations();
        const experiments = await getExperiments();

        if (mounted) {
          setHistory(records);
          setSavedExperiments(experiments);
        }
      } catch {
        if (mounted) {
          setHistoryError("Could not load saved history.");
          setHistoryError("Could not load saved generations and experiments.");
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

  async function runExperiment() {
    setExperimentLoading(true);
    setExperimentError("");
    setExperimentResults([]);
    setExperimentTime(null);

    try {
      const values = experimentValues
        .split(",")
        .map((value) => Number(value.trim()));

      if (
        values.length < 2 ||
        values.length > 4 ||
        values.some((value) => !Number.isFinite(value))
      ) {
        throw new Error("Enter 2 to 4 valid comma-separated values.");
      }

      const response = await fetch(`${API}/api/experiment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_type: experimentType,
          digit,
          count: Math.min(count, 8),
          values,
          w: guidance,
          steps,
          seed,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Experiment failed.");
      }

      setExperimentResults(data.results);
      setExperimentTime(data.total_seconds);
      setStatus("Experiment completed successfully.");
    } catch (err) {
      setExperimentError(
        err instanceof Error ? err.message : "Unable to run experiment."
      );
    } finally {
      setExperimentLoading(false);
    }
  }

  async function saveCurrentExperiment() {
    if (experimentResults.length === 0) return;

    try {
      const record: ExperimentRecord = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        experimentType,
        digit,
        count: Math.min(count, 8),
        values: experimentValues
          .split(",")
          .map((value) => Number(value.trim())),
        guidance,
        steps,
        seed,
        totalSeconds: experimentTime ?? 0,
        results: experimentResults,
      };

      await saveExperiment(record);

      setSavedExperiments((current) => [record, ...current]);
      setStatus("Experiment saved to history.");
    } catch (err) {
      setExperimentError(
        err instanceof Error ? err.message : "Could not save experiment."
      );
    }
  }

  function exportExperimentCSV() {
    if (experimentResults.length === 0) return;

    const headers = [
      "Experiment Type",
      "Target Digit",
      "Parameter",
      "Value",
      "Matches",
      "Match Rate (%)",
      "Average Confidence",
      "Inference Time (s)",
    ];

    const rows = experimentResults.map((result: any) => [
      experimentType,
      digit,
      result.parameter,
      result.value,
      result.matches,
      result.match_rate,
      result.avg_confidence,
      result.seconds,
    ]);

    const csv = [headers, ...rows]
      .map((row) => row.join(","))
      .join("\n");

    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `experiment-${experimentType}-results.csv`;
    link.click();

    URL.revokeObjectURL(url);
  }

  async function runVisualization() {
    setVisualizationLoading(true);
    setVisualizationError("");
    setVisualization(null);
    setActiveFrame(0);

    try {
      const response = await fetch(`${API}/api/visualize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit: visualDigit,
          w: visualGuidance,
          steps: visualSteps,
          seed: visualSeed,
          snapshot_count: snapshotCount,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Visualization failed (${response.status}): ${message}`
        );
      }

      const data: VisualizationResult = await response.json();

      if (!data.snapshots || data.snapshots.length === 0) {
        throw new Error("The backend returned no visualization frames.");
      }

      setVisualization(data);
      setActiveFrame(0);
    } catch (err) {
      setVisualizationError(
        err instanceof Error ? err.message : "Unable to run visualization."
      );
    } finally {
      setVisualizationLoading(false);
    }
  }

  async function runInterpolation() {
    setInterpolationLoading(true);
    setInterpolationError("");
    setInterpolation(null);
    setSelectedInterpFrame(0);
    setIsPlaying(false);

    try {
      const response = await fetch(`${API}/api/interpolate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit: interpDigit,
          w: interpGuidance,
          steps: interpSteps,
          seed_a: seedA,
          seed_b: seedB,
          frames: interpFrameCount,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Interpolation failed (${response.status}): ${message}`
        );
      }

      const data = await response.json();

      if (!data.frames || data.frames.length === 0) {
        throw new Error("The backend returned no interpolation frames.");
      }

      setInterpolation(data);
      setSelectedInterpFrame(0);
    } catch (err) {
      setInterpolationError(
        err instanceof Error
          ? err.message
          : "Unable to run noise interpolation."
      );
    } finally {
      setInterpolationLoading(false);
    }
  }

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

  function reuseSettings(record: GenerationRecord) {
    setDigit(record.digit);
    setCount(record.count);
    setGuidance(record.guidance);
    setSteps(record.steps);
    setSeed(record.seed);
    setActiveTab("Studio");
    setStatus("Previous generation settings restored.");
  }

  async function removeRecord(id: string) {
    try {
      await deleteGeneration(id);
      setHistory((previous) => previous.filter((item) => item.id !== id));
      setHistoryError("");
    } catch {
      setHistoryError("Could not delete this generation.");
    }
  }

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
    ["◌", "Denoising Visualizer"],
    ["◈", "Noise Interpolation"],
    ["⚗", "Experiment Lab"],
  ];

  return (
    <main className="min-h-screen bg-[#080b14] text-white selection:bg-violet-500/30">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-40 top-20 h-96 w-96 rounded-full bg-blue-600/10 blur-[130px]" />
        <div className="absolute right-0 top-40 h-96 w-96 rounded-full bg-violet-600/10 blur-[130px]" />
        <div className="absolute bottom-0 left-1/2 h-80 w-80 rounded-full bg-cyan-500/5 blur-[130px]" />
      </div>

      <div className="relative mx-auto flex min-h-screen max-w-[1600px]">
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

        <section className="min-w-0 flex-1 px-5 py-7 sm:px-8 lg:px-10">
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
                      : activeTab === "Denoising Visualizer"
                        ? "Denoising Visualizer"
                        : activeTab === "Noise Interpolation"
                          ? "Noise Interpolation"
                          : activeTab === "Experiment Lab"
                            ? "Experiment Lab"
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
                      : activeTab === "Denoising Visualizer"
                        ? "Watch a noisy image progressively become a generated digit."
                        : activeTab === "Noise Interpolation"
                          ? "Explore smooth transitions between two random noise inputs."
                          : activeTab === "Experiment Lab"
                            ? "Run controlled experiments to understand how diffusion parameters affect generated images."
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

          {activeTab === "Denoising Visualizer" && (
            <section className="space-y-6">
              <div className="rounded-3xl border border-violet-400/20 bg-violet-500/[0.06] p-6 sm:p-8">
                <p className="text-xs tracking-widest text-violet-300">
                  DIFFUSION PROCESS / STEP BY STEP
                </p>
                <h3 className="mt-2 text-2xl font-semibold">
                  From Noise to a Handwritten Digit
                </h3>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                  This visualizer captures intermediate states from the actual
                  DDIM sampling process. Move through the frames to see how the
                  model gradually transforms random noise into the selected
                  digit.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[0.8fr_1.2fr]">
                <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-violet-300">
                    01 / CONFIGURATION
                  </p>
                  <h3 className="mt-2 text-xl font-semibold">
                    Visualizer Settings
                  </h3>

                  <div className="mt-7">
                    <div className="mb-3 flex justify-between text-sm">
                      <label>Target digit</label>
                      <span className="text-violet-300">{visualDigit}</span>
                    </div>
                    <div className="grid grid-cols-5 gap-2">
                      {Array.from({ length: 10 }, (_, i) => (
                        <button
                          key={i}
                          onClick={() => setVisualDigit(i)}
                          className={`aspect-square rounded-xl border text-lg font-semibold ${visualDigit === i
                            ? "border-violet-400 bg-violet-500/30 text-white"
                            : "border-white/10 bg-white/[0.035] text-slate-400 hover:bg-white/10"
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
                        <label>Guidance scale</label>
                        <span className="text-violet-300">
                          {visualGuidance.toFixed(1)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="3"
                        step="0.1"
                        value={visualGuidance}
                        onChange={(e) =>
                          setVisualGuidance(Number(e.target.value))
                        }
                        className="w-full accent-violet-500"
                      />
                    </div>

                    <div>
                      <div className="mb-2 flex justify-between text-sm">
                        <label>DDIM sampling steps</label>
                        <span className="text-violet-300">{visualSteps}</span>
                      </div>
                      <input
                        type="range"
                        min="5"
                        max="100"
                        step="5"
                        value={visualSteps}
                        onChange={(e) =>
                          setVisualSteps(Number(e.target.value))
                        }
                        className="w-full accent-violet-500"
                      />
                    </div>

                    <div>
                      <div className="mb-2 flex justify-between text-sm">
                        <label>Snapshot count</label>
                        <span className="text-violet-300">{snapshotCount}</span>
                      </div>
                      <input
                        type="range"
                        min="4"
                        max="12"
                        step="1"
                        value={snapshotCount}
                        onChange={(e) =>
                          setSnapshotCount(Number(e.target.value))
                        }
                        className="w-full accent-violet-500"
                      />
                      <p className="mt-1 text-xs text-slate-500">
                        Number of intermediate sampling states to capture.
                      </p>
                    </div>

                    <div>
                      <label className="mb-2 block text-sm">Random seed</label>
                      <div className="flex gap-2">
                        <input
                          type="number"
                          min="0"
                          max="2147483647"
                          value={visualSeed}
                          onChange={(e) =>
                            setVisualSeed(
                              Math.max(
                                0,
                                Math.min(
                                  2147483647,
                                  Number(e.target.value) || 0
                                )
                              )
                            )
                          }
                          className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none"
                        />
                        <button
                          onClick={() =>
                            setVisualSeed(Math.floor(Math.random() * 100000))
                          }
                          className="rounded-xl border border-white/10 bg-white/5 px-3 text-xs hover:bg-white/10"
                        >
                          Random
                        </button>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={runVisualization}
                    disabled={visualizationLoading || backendOnline !== true}
                    className="mt-8 w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {visualizationLoading
                      ? "Generating sampling frames..."
                      : backendOnline === true
                        ? "Run Denoising Visualization"
                        : "Backend Unavailable"}
                  </button>

                  {visualizationError && (
                    <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
                      {visualizationError}
                    </div>
                  )}
                </div>

                <div className="rounded-3xl border border-white/10 bg-[#111522]/80 p-6 sm:p-8">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs tracking-widest text-cyan-300">
                        02 / SAMPLING TIMELINE
                      </p>
                      <h3 className="mt-2 text-xl font-semibold">
                        Denoising Frames
                      </h3>
                    </div>
                    {visualization && (
                      <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300">
                        {visualization.snapshots.length} frames captured
                      </span>
                    )}
                  </div>

                  {!visualization && !visualizationLoading && (
                    <div className="mt-6 flex min-h-[380px] flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                      <div className="flex h-28 w-28 items-center justify-center rounded-3xl border border-violet-400/20 bg-violet-500/10 text-6xl text-violet-300">
                        {visualDigit}
                      </div>
                      <h4 className="mt-6 text-lg font-medium">
                        Ready to visualize
                      </h4>
                      <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
                        Configure the sampling parameters and run the
                        visualizer. The first frame represents noise and the
                        final frame is the generated sample.
                      </p>
                    </div>
                  )}

                  {visualizationLoading && (
                    <div className="mt-6 flex min-h-[380px] items-center justify-center rounded-2xl border border-white/10 bg-black/20 text-sm text-violet-300">
                      Running the DDIM sampler and capturing intermediate
                      states...
                    </div>
                  )}

                  {visualization && visualization.snapshots.length > 0 && (
                    <div className="mt-6">
                      <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
                        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <p className="text-xs text-slate-500">
                              CURRENT FRAME
                            </p>
                            <h4 className="mt-1 text-lg font-semibold">
                              {activeFrame === 0
                                ? "Initial Noise"
                                : activeFrame ===
                                  visualization.snapshots.length - 1
                                  ? "Final Generated Digit"
                                  : `Denoising Stage ${activeFrame}`}
                            </h4>
                          </div>
                          <span className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-300">
                            Timestep:{" "}
                            {visualization.snapshots[activeFrame]?.timestep ??
                              "—"}
                          </span>
                        </div>

                        <div className="mx-auto max-w-[420px] rounded-2xl border border-white/10 bg-[#080b14] p-4">
                          <img
                            src={`data:image/png;base64,${visualization.snapshots[activeFrame]?.image ?? ""
                              }`}
                            alt={`Denoising frame ${activeFrame + 1}`}
                            className="aspect-square w-full object-contain"
                          />
                        </div>

                        <div className="mt-5 flex items-center justify-between gap-3">
                          <button
                            onClick={() =>
                              setActiveFrame((current) =>
                                Math.max(0, current - 1)
                              )
                            }
                            disabled={activeFrame === 0}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm disabled:opacity-30"
                          >
                            Previous
                          </button>

                          <span className="text-xs tabular-nums text-slate-400">
                            {activeFrame + 1} /{" "}
                            {visualization.snapshots.length}
                          </span>

                          <button
                            onClick={() =>
                              setActiveFrame((current) =>
                                Math.min(
                                  visualization.snapshots.length - 1,
                                  current + 1
                                )
                              )
                            }
                            disabled={
                              activeFrame ===
                              visualization.snapshots.length - 1
                            }
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm disabled:opacity-30"
                          >
                            Next
                          </button>
                        </div>

                        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500"
                            style={{
                              width: `${((activeFrame + 1) /
                                visualization.snapshots.length) *
                                100
                                }%`,
                            }}
                          />
                        </div>
                      </div>

                      <div className="mt-5 grid grid-cols-3 gap-3">
                        <StatCard
                          label="DIGIT"
                          value={String(visualization.digit)}
                          description="Condition label"
                        />
                        <StatCard
                          label="DDIM STEPS"
                          value={String(visualization.steps)}
                          description="Sampling steps"
                        />
                        <StatCard
                          label="TIME"
                          value={`${visualization.seconds.toFixed(2)} s`}
                          description="Backend runtime"
                        />
                      </div>

                      <div className="mt-5">
                        <p className="mb-3 text-xs font-semibold tracking-widest text-slate-500">
                          ALL CAPTURED FRAMES · SELECT TO INSPECT
                        </p>
                        <div className="grid grid-cols-3 gap-3 sm:grid-cols-4">
                          {visualization.snapshots.map((frame, index) => (
                            <button
                              key={`${frame.timestep}-${index}`}
                              onClick={() => setActiveFrame(index)}
                              className={`rounded-xl border p-2 text-left transition ${activeFrame === index
                                ? "border-violet-400 bg-violet-500/10"
                                : "border-white/10 bg-black/20 hover:border-white/30"
                                }`}
                            >
                              <img
                                src={`data:image/png;base64,${frame.image}`}
                                alt={`Snapshot ${index + 1}`}
                                className="aspect-square w-full rounded-lg bg-[#080b14] object-contain"
                              />
                              <p className="mt-2 text-[10px] text-slate-400">
                                {index === 0
                                  ? "Initial noise"
                                  : index ===
                                    visualization.snapshots.length - 1
                                    ? "Final image"
                                    : `Frame ${index + 1}`}
                              </p>
                              <p className="text-[10px] text-slate-600">
                                t = {frame.timestep}
                              </p>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}


          {activeTab === "Noise Interpolation" && (
            <section className="space-y-6">
              <div className="rounded-3xl border border-violet-400/20 bg-violet-500/[0.06] p-6 sm:p-8">
                <p className="text-xs font-semibold tracking-[0.2em] text-violet-300">
                  LATENT SPACE EXPLORATION
                </p>
                <h3 className="mt-3 text-2xl font-semibold">
                  Noise Interpolation
                </h3>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                  Explore how the diffusion model transforms different
                  random noise inputs into the same conditioned digit.
                  Move through intermediate frames to observe the transition.
                </p>
              </div>

              <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
                <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6">
                  <h3 className="text-lg font-semibold">Interpolation Settings</h3>

                  <div className="mt-6 space-y-5">
                    <div>
                      <label className="text-sm text-slate-400">Target Digit</label>
                      <select
                        value={interpDigit}
                        onChange={(e) => setInterpDigit(Number(e.target.value))}
                        className="mt-2 w-full rounded-xl border border-white/10 bg-[#111827] p-3 text-white"
                      >
                        {Array.from({ length: 10 }, (_, i) => (
                          <option key={i} value={i}>{i}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="text-sm text-slate-400">
                        CFG Guidance: {interpGuidance.toFixed(1)}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="3"
                        step="0.1"
                        value={interpGuidance}
                        onChange={(e) => setInterpGuidance(Number(e.target.value))}
                        className="mt-3 w-full accent-violet-500"
                      />
                    </div>

                    <div>
                      <label className="text-sm text-slate-400">
                        Sampling Steps: {interpSteps}
                      </label>
                      <input
                        type="range"
                        min="5"
                        max="100"
                        step="5"
                        value={interpSteps}
                        onChange={(e) => setInterpSteps(Number(e.target.value))}
                        className="mt-3 w-full accent-violet-500"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-slate-400">Seed A</label>
                        <input
                          type="number"
                          min="0"
                          max="2147483647"
                          value={seedA}
                          onChange={(e) => setSeedA(Math.max(0, Math.min(2147483647, Number(e.target.value))))}
                          className="mt-2 w-full rounded-xl border border-white/10 bg-[#111827] p-3 text-sm text-white"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400">Seed B</label>
                        <input
                          type="number"
                          min="0"
                          max="2147483647"
                          value={seedB}
                          onChange={(e) => setSeedB(Math.max(0, Math.min(2147483647, Number(e.target.value))))}
                          className="mt-2 w-full rounded-xl border border-white/10 bg-[#111827] p-3 text-sm text-white"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="text-sm text-slate-400">
                        Number of Frames: {interpFrameCount}
                      </label>
                      <input
                        type="range"
                        min="3"
                        max="15"
                        step="1"
                        value={interpFrameCount}
                        onChange={(e) => setInterpFrameCount(Number(e.target.value))}
                        className="mt-3 w-full accent-violet-500"
                      />
                    </div>

                    <button
                      onClick={runInterpolation}
                      disabled={interpolationLoading}
                      className="w-full rounded-xl bg-violet-600 px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {interpolationLoading ? "Generating Frames..." : "Generate Interpolation"}
                    </button>

                    {interpolationError && (
                      <p className="rounded-xl border border-red-400/20 bg-red-400/5 p-3 text-sm text-red-300">
                        {interpolationError}
                      </p>
                    )}
                  </div>
                </div>

                <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6">
                  {!interpolation ? (
                    <div className="flex min-h-[400px] flex-col items-center justify-center text-center">
                      <div className="text-5xl text-violet-400">◈</div>
                      <h3 className="mt-5 text-xl font-semibold">
                        Ready to interpolate
                      </h3>
                      <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
                        Generate frames to explore the transition between
                        two random noise inputs.
                      </p>
                    </div>
                  ) : (
                    <div>
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <h3 className="text-xl font-semibold">
                            Interpolation Results
                          </h3>
                          <p className="mt-1 text-xs text-slate-500">
                            Target digit: {interpolation.digit}
                          </p>
                        </div>
                        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-300">
                          {interpolation.count} frames generated
                        </span>
                      </div>

                      <div className="mt-7 grid grid-cols-1 gap-5 sm:grid-cols-2">
                        <div className="rounded-2xl border border-white/10 bg-[#080b14] p-4">
                          <p className="mb-3 text-xs text-slate-400">Starting Noise · Seed {seedA}</p>
                          <img
                            src={`data:image/png;base64,${interpolation.frames[0].image}`}
                            alt="Starting interpolation frame"
                            className="aspect-square w-full rounded-xl object-contain"
                          />
                        </div>

                        <div className="rounded-2xl border border-white/10 bg-[#080b14] p-4">
                          <p className="mb-3 text-xs text-slate-400">Ending Noise · Seed {seedB}</p>
                          <img
                            src={`data:image/png;base64,${interpolation.frames[interpolation.frames.length - 1].image}`}
                            alt="Ending interpolation frame"
                            className="aspect-square w-full rounded-xl object-contain"
                          />
                        </div>
                      </div>

                      <div className="mt-8 rounded-2xl border border-violet-400/20 bg-violet-500/[0.05] p-5">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <h4 className="font-semibold">Explore Transition</h4>
                          <span className="text-sm text-violet-300">
                            α = {interpolation.frames[selectedInterpFrame]?.alpha.toFixed(3)}
                          </span>
                        </div>

                        <div className="mx-auto mt-5 max-w-sm rounded-2xl border border-white/10 bg-[#080b14] p-4">
                          <img
                            src={`data:image/png;base64,${interpolation.frames[selectedInterpFrame]?.image}`}
                            alt={`Interpolation frame ${selectedInterpFrame + 1}`}
                            className="aspect-square w-full rounded-xl object-contain"
                          />
                        </div>

                        <div className="mt-5 flex items-center justify-between text-xs text-slate-400">
                          <span>Frame {selectedInterpFrame + 1}</span>
                          <span>of {interpolation.frames.length}</span>
                        </div>

                        <input
                          type="range"
                          min="0"
                          max={interpolation.frames.length - 1}
                          step="1"
                          value={selectedInterpFrame}
                          onChange={(e) => setSelectedInterpFrame(Number(e.target.value))}
                          className="mt-3 w-full accent-violet-500"
                        />

                        <div className="mt-5 flex justify-center gap-3">
                          <button
                            onClick={() => {
                              if (selectedInterpFrame >= interpolation.frames.length - 1) {
                                setSelectedInterpFrame(0);
                              }
                              setIsPlaying(!isPlaying);
                            }}
                            className="rounded-xl bg-violet-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-violet-500"
                          >
                            {isPlaying ? "Pause" : "Play"}
                          </button>

                          <button
                            onClick={() => {
                              setIsPlaying(false);
                              setSelectedInterpFrame(0);
                            }}
                            className="rounded-xl border border-white/10 px-6 py-2.5 text-sm text-slate-300 hover:bg-white/5"
                          >
                            Reset
                          </button>
                        </div>

                        <div className="mt-4 flex justify-between gap-3">
                          <button
                            onClick={() => setSelectedInterpFrame(0)}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 hover:bg-white/5"
                          >
                            First
                          </button>
                          <button
                            onClick={() => setSelectedInterpFrame(Math.max(0, selectedInterpFrame - 1))}
                            disabled={selectedInterpFrame === 0}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 disabled:opacity-30"
                          >
                            Previous
                          </button>
                          <button
                            onClick={() => setSelectedInterpFrame(Math.min(interpolation.frames.length - 1, selectedInterpFrame + 1))}
                            disabled={selectedInterpFrame === interpolation.frames.length - 1}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 disabled:opacity-30"
                          >
                            Next
                          </button>
                          <button
                            onClick={() => setSelectedInterpFrame(interpolation.frames.length - 1)}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 hover:bg-white/5"
                          >
                            Last
                          </button>
                        </div>
                      </div>

                      <div className="mt-6">
                        <h4 className="mb-4 font-semibold">All Interpolation Frames</h4>
                        <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-5">
                          {interpolation.frames.map((frame: any, index: number) => (
                            <button
                              key={index}
                              onClick={() => setSelectedInterpFrame(index)}
                              className={`rounded-xl border p-2 transition ${selectedInterpFrame === index
                                ? "border-violet-400 bg-violet-500/10"
                                : "border-white/10 bg-white/[0.02] hover:border-white/30"
                                }`}
                            >
                              <img
                                src={`data:image/png;base64,${frame.image}`}
                                alt={`Interpolation frame ${index + 1}`}
                                className="aspect-square w-full rounded-lg object-contain"
                              />
                              <p className="mt-2 text-xs text-slate-400">
                                α = {frame.alpha.toFixed(2)}
                              </p>
                            </button>
                          ))}
                        </div>
                      </div>

                      <p className="mt-5 text-xs text-slate-500">
                        Inference time: {interpolation.inference_time}s
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}


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
                      <span className="text-violet-300">
                        Selected: {digit}
                      </span>
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
                    disabled={
                      loading || compareLoading || backendOnline !== true
                    }
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
                  disabled={
                    compareLoading || loading || backendOnline !== true
                  }
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

          {activeTab === "Experiment Lab" && (
            <section className="space-y-6">
              <div className="rounded-3xl border border-violet-400/20 bg-[#111522]/80 p-6 sm:p-8">
                <p className="text-xs tracking-widest text-violet-300">
                  CONTROLLED DIFFUSION EXPERIMENTS
                </p>
                <h3 className="mt-2 text-2xl font-semibold">
                  Experiment Configuration
                </h3>
                <p className="mt-2 text-sm text-slate-400">
                  Compare how different diffusion parameters affect generated digits.
                </p>

                <div className="mt-6 grid gap-5 sm:grid-cols-2">
                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Experiment Type
                    </label>
                    <select
                      value={experimentType}
                      onChange={(e) => {
                        setExperimentType(e.target.value);
                        setExperimentValues(
                          e.target.value === "guidance"
                            ? "0, 1.2, 2"
                            : e.target.value === "steps"
                              ? "10, 20, 50"
                              : "42, 123, 456"
                        );
                        setExperimentResults([]);
                      }}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white"
                    >
                      <option value="guidance">CFG Guidance Scale</option>
                      <option value="steps">DDIM Sampling Steps</option>
                      <option value="seeds">Random Seeds</option>
                    </select>
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Target Digit (0–9)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="9"
                      value={digit}
                      onChange={(e) => setDigit(Number(e.target.value))}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Samples per Configuration (1–8)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="8"
                      value={count}
                      onChange={(e) => {
                        const value = e.target.value;
                        setCount(value === "" ? ("" as any) : Number(value));
                      }}
                      onBlur={() => {
                        setCount((current) =>
                          Math.min(8, Math.max(1, Number(current) || 1))
                        );
                      }}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Comparison Values (comma separated)
                    </label>
                    <input
                      type="text"
                      value={experimentValues}
                      onChange={(e) => setExperimentValues(e.target.value)}
                      placeholder="0, 1.2, 2"
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Guidance Scale
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="3"
                      step="0.1"
                      value={guidance}
                      onChange={(e) => setGuidance(Number(e.target.value))}
                      disabled={experimentType === "guidance"}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white disabled:opacity-40"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Sampling Steps
                    </label>
                    <input
                      type="number"
                      min="5"
                      max="100"
                      value={steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                      disabled={experimentType === "steps"}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white disabled:opacity-40"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Starting Seed
                    </label>
                    <input
                      type="number"
                      min="0"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                      disabled={experimentType === "seeds"}
                      className="w-full rounded-xl border border-white/10 bg-[#080b14] p-3 text-white disabled:opacity-40"
                    />
                  </div>
                </div>

                <button
                  onClick={runExperiment}
                  disabled={experimentLoading || backendOnline !== true}
                  className="mt-7 w-full rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-4 font-semibold disabled:opacity-50"
                >
                  {experimentLoading
                    ? "Running Experiment..."
                    : "Run Experiment"}
                </button>

                {experimentError && (
                  <p className="mt-4 rounded-xl bg-red-500/10 p-3 text-sm text-red-300">
                    {experimentError}
                  </p>
                )}

                {experimentLoading && (
                  <p className="mt-4 text-center text-sm text-violet-300">
                    Generating comparison samples...
                  </p>
                )}
              </div>

              {experimentResults.length > 0 && (
                <div className="space-y-6">

                  <div className="flex justify-end">
                    <button
                      onClick={exportExperimentCSV}
                      className="rounded-xl border border-violet-400/30 bg-violet-500/10 px-5 py-3 text-sm font-medium text-violet-200 hover:bg-violet-500/20"
                    >
                      Download Results CSV
                    </button>
                    <button
                      onClick={saveCurrentExperiment}
                      disabled={experimentResults.length === 0 || experimentLoading}
                      className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Save Experiment
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">

                    <StatCard
                      label="CONFIGURATIONS"
                      value={String(experimentResults.length)}
                      description="Compared settings"
                    />
                    <StatCard
                      label="TOTAL TIME"
                      value={`${experimentTime} s`}
                      description="Experiment duration"
                    />
                    <StatCard
                      label="BEST MATCH RATE"
                      value={`${Math.max(...experimentResults.map((r: any) => r.match_rate))}%`}
                      description="Classifier consistency"
                    />
                    <StatCard
                      label="BEST CONFIDENCE"
                      value={Math.max(...experimentResults.map((r: any) => r.avg_confidence)).toFixed(3)}
                      description="Average classifier confidence"
                    />
                  </div>

                  {(() => {
                    const best = [...experimentResults].sort(
                      (a: any, b: any) =>
                        b.match_rate - a.match_rate ||
                        b.avg_confidence - a.avg_confidence ||
                        a.seconds - b.seconds
                    )[0];

                    return (
                      <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/[0.06] p-5">
                        <p className="text-xs font-semibold tracking-widest text-emerald-300">
                          EXPERIMENT RECOMMENDATION
                        </p>

                        <h4 className="mt-2 text-xl font-semibold">
                          Recommended: {best.parameter} = {best.value}
                        </h4>

                        <p className="mt-2 text-sm text-slate-400">
                          Selected using match rate first, classifier confidence second,
                          and inference time as a tie-breaker.
                        </p>

                        <div className="mt-4 grid grid-cols-3 gap-3">
                          <StatCard
                            label="MATCH RATE"
                            value={`${best.match_rate}%`}
                            description="Classifier consistency"
                          />
                          <StatCard
                            label="CONFIDENCE"
                            value={best.avg_confidence.toFixed(3)}
                            description="Average confidence"
                          />
                          <StatCard
                            label="TIME"
                            value={`${best.seconds} s`}
                            description="Inference time"
                          />
                        </div>
                      </div>
                    );
                  })()}

                  <div className="grid gap-5 lg:grid-cols-2">
                    {/* Match Rate Chart */}
                    <div className="rounded-2xl border border-white/10 bg-[#111522]/80 p-5">
                      <h4 className="mb-5 font-semibold">Match Rate Comparison</h4>

                      <div className="space-y-5">
                        {experimentResults.map((result: any, index: number) => (
                          <div key={index}>
                            <div className="mb-2 flex justify-between text-sm">
                              <span className="text-slate-300">
                                {result.parameter} = {result.value}
                              </span>
                              <span className="font-semibold text-emerald-400">
                                {result.match_rate}%
                              </span>
                            </div>
                            <div className="h-3 overflow-hidden rounded-full bg-white/10">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all"
                                style={{ width: `${result.match_rate}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Inference Time Chart */}
                    <div className="rounded-2xl border border-white/10 bg-[#111522]/80 p-5">
                      <h4 className="mb-5 font-semibold">Inference Time Comparison</h4>

                      <div className="space-y-5">
                        {experimentResults.map((result: any, index: number) => {
                          const maxTime = Math.max(
                            ...experimentResults.map((item: any) => item.seconds),
                            0.01
                          );

                          return (
                            <div key={index}>
                              <div className="mb-2 flex justify-between text-sm">
                                <span className="text-slate-300">
                                  {result.parameter} = {result.value}
                                </span>
                                <span className="font-semibold text-violet-300">
                                  {result.seconds} s
                                </span>
                              </div>
                              <div className="h-3 overflow-hidden rounded-full bg-white/10">
                                <div
                                  className="h-full rounded-full bg-gradient-to-r from-violet-500 to-pink-500 transition-all"
                                  style={{ width: `${(result.seconds / maxTime) * 100}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>

                  {experimentResults.map((result: any, index: number) => (
                    <div
                      key={index}
                      className="rounded-2xl border border-white/10 bg-[#111522]/80 p-5"
                    >
                      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                        <h4 className="text-lg font-semibold">
                          {result.parameter} = {result.value}
                        </h4>
                        <span className="text-sm text-emerald-400">
                          {result.matches}/{result.items.length} matches
                        </span>
                      </div>

                      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {(() => {
                          const bestMatch = Math.max(
                            ...experimentResults.map((r: any) => r.match_rate)
                          );
                          const bestConfidence = Math.max(
                            ...experimentResults.map((r: any) => r.avg_confidence)
                          );
                          const fastestTime = Math.min(
                            ...experimentResults.map((r: any) => r.seconds)
                          );

                          const metrics = [
                            {
                              label: "MATCH RATE",
                              value: `${result.match_rate}%`,
                              description: result.match_rate === bestMatch
                                ? "★ Best match rate"
                                : "Requested digit",
                              isBest: result.match_rate === bestMatch,
                            },
                            {
                              label: "AVG CONFIDENCE",
                              value: result.avg_confidence.toFixed(3),
                              description: result.avg_confidence === bestConfidence
                                ? "★ Highest confidence"
                                : "Classifier confidence",
                              isBest: result.avg_confidence === bestConfidence,
                            },
                            {
                              label: "INFERENCE TIME",
                              value: `${result.seconds} s`,
                              description: result.seconds === fastestTime
                                ? "★ Fastest configuration"
                                : "Generation time",
                              isBest: result.seconds === fastestTime,
                            },
                          ];

                          return metrics.map((metric) => (
                            <div
                              key={metric.label}
                              className={`rounded-2xl border p-4 transition-all ${metric.isBest
                                ? "border-emerald-400/50 bg-emerald-500/10 shadow-[0_0_15px_rgba(52,211,153,0.08)]"
                                : "border-white/10 bg-white/[0.03]"
                                }`}
                            >
                              <p className="text-[10px] font-semibold tracking-[0.18em] text-slate-500">
                                {metric.label}
                              </p>
                              <p className="mt-3 text-2xl font-bold text-white">
                                {metric.value}
                              </p>
                              <p className={`mt-1 text-xs ${metric.isBest ? "font-semibold text-emerald-400" : "text-slate-500"
                                }`}>
                                {metric.description}
                              </p>
                            </div>
                          ));
                        })()}
                      </div>
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        {result.items.map((item: GeneratedItem, imageIndex: number) => (
                          <ImageCard
                            key={imageIndex}
                            item={item}
                            index={imageIndex}
                            selectedDigit={digit}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div className="mt-10 border-t border-slate-700/60 pt-8">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-semibold">Experiment History</h3>
                    <p className="mt-1 text-sm text-slate-400">
                      {savedExperiments.length} saved experiments
                    </p>
                  </div>
                </div>

                {historyLoading ? (
                  <p className="text-sm text-slate-400">Loading experiment history...</p>
                ) : historyError ? (
                  <p className="text-sm text-red-400">{historyError}</p>
                ) : savedExperiments.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-slate-700 p-8 text-center">
                    <p className="font-medium">No saved experiments yet</p>
                    <p className="mt-2 text-sm text-slate-400">
                      Run an experiment and click Save Experiment to see it here.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {savedExperiments.map((experiment) => (
                      <div
                        key={experiment.id}
                        className="rounded-2xl border border-slate-700 bg-slate-900/50 p-5"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div>
                            <h4 className="font-semibold capitalize">
                              {experiment.experimentType} Experiment
                            </h4>
                            <p className="mt-1 text-xs text-slate-400">
                              {new Date(experiment.createdAt).toLocaleString()}
                            </p>
                          </div>

                          <button
                            onClick={async () => {
                              const db = await openDatabase();
                              const transaction = db.transaction("experiments", "readwrite");
                              transaction.objectStore("experiments").delete(experiment.id);

                              transaction.oncomplete = () => {
                                db.close();
                                setSavedExperiments((current) =>
                                  current.filter((item) => item.id !== experiment.id)
                                );
                              };

                              transaction.onerror = () => db.close();
                            }}
                            className="rounded-lg border border-red-500/30 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10"
                          >
                            Delete
                          </button>
                        </div>

                        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                          <StatCard
                            label="TARGET DIGIT"
                            value={String(experiment.digit)}
                            description="Generated class"
                          />
                          <StatCard
                            label="CONFIGURATIONS"
                            value={String(experiment.results.length)}
                            description="Compared settings"
                          />
                          <StatCard
                            label="TOTAL TIME"
                            value={`${experiment.totalSeconds.toFixed(2)}s`}
                            description="Experiment duration"
                          />
                          <StatCard
                            label="SAMPLES"
                            value={String(experiment.count)}
                            description="Per configuration"
                          />
                        </div>

                        <div className="mt-4 space-y-3">
                          {experiment.results.map((result: any, index: number) => (
                            <div
                              key={index}
                              className="rounded-xl border border-slate-700/70 p-4"
                            >
                              <p className="font-medium">
                                {result.parameter} = {result.value}
                              </p>
                              <p className="mt-1 text-sm text-slate-400">
                                Match rate: {result.match_rate.toFixed(1)}% ·
                                Confidence: {result.avg_confidence.toFixed(3)} ·
                                Time: {result.seconds.toFixed(2)}s
                              </p>

                              <div className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-8">
                                {result.items.map((item: GeneratedItem, imageIndex: number) => (
                                  <img
                                    key={imageIndex}
                                    src={`data:image/png;base64,${item.image}`}
                                    alt={`Generated digit ${experiment.digit}`}
                                    className="w-full rounded-lg border border-slate-700"
                                  />
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          )}

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
                    className="mt-5 rounded-xl bg-violet-600 px-5 py-3 text-sm font-medium"
                  >
                    Open Studio
                  </button>
                </div>
              ) : (
                <div className="space-y-6">
                  {history.map((record) => (
                    <div
                      key={record.id}
                      className="rounded-3xl border border-white/10 bg-[#111522]/70 p-5 sm:p-7"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <h4 className="text-lg font-semibold">
                            Digit {record.digit}
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
                            Reuse Settings
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
                            Download All
                          </button>
                          <button
                            onClick={() => removeRecord(record.id)}
                            className="rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-xs text-red-300"
                          >
                            Delete
                          </button>
                        </div>
                      </div>

                      <div className="my-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
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
                        <strong className="text-cyan-300">Diversity:</strong>{" "}
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

          <footer className="mt-10 border-t border-white/[0.07] py-6 text-xs text-slate-600">
            DiffusionLab · PyTorch + FastAPI + Next.js
          </footer>
        </section>
      </div>
    </main>
  );
}