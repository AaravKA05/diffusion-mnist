"use client";

import { useEffect, useState } from "react";

import {
  API,
  apiFetch,
  onEngineModeChange,
  setEngineMode,
  type EngineMode,
} from "@/lib/engine/apiClient";
import { preloadRuntime } from "@/lib/engine/runtime";

import RecognitionTimeline from "./RecognitionTimeline";
import SketchToDigit from "./SketchToDigit";

console.log("ACTIVE BACKEND:", API);
const DB_NAME = "DiffusionLabDB";

const TAB_TITLES: Record<string, string> = {
  Studio: "Creative Studio",
  "My Generations": "My Generations",
  "Model Insights": "Model Insights",
  Evaluation: "Model Evaluation",
  "Denoising Visualizer": "Denoising Visualizer",
  "Noise Interpolation": "Noise Interpolation",
  "Experiment Lab": "Experiment Lab",
  "Representation Explorer": "Representation Explorer",
  "Recognition Timeline": "Recognition Timeline",
  "Sketch to Digit": "Sketch to Digit",
};

const TAB_SUBTITLES: Record<string, string> = {
  Studio: "Explore handwritten digit generation with diffusion models.",
  "My Generations": "Your previously generated samples, saved in this browser.",
  "Model Insights": "A summary of your generation activity and model configuration.",
  Evaluation:
    "Evaluate class consistency, confidence, diversity and inference speed.",
  "Denoising Visualizer":
    "Watch a noisy image progressively become a generated digit.",
  "Noise Interpolation":
    "Explore smooth transitions between two random noise inputs.",
  "Experiment Lab":
    "Run controlled experiments to understand how diffusion parameters affect generated images.",
  "Representation Explorer":
    "See which internal features the U-Net uses while it denoises a digit.",
  "Recognition Timeline":
    "Find the sampling step where the digit becomes recognisable, and how guidance changes it.",
  "Sketch to Digit":
    "Draw a rough digit and let the diffusion model clean it up.",
};
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
      className={`dl-fade-in overflow-hidden rounded-xl border-2 bg-black/40 p-2 transition duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-violet-950/40 ${correct ? "border-emerald-500/70" : "border-red-500/70"
        }`}
    >
      <img
        src={`data:image/png;base64,${item.image}`}
        alt={`Generated digit ${selectedDigit}, sample ${index + 1}`}
        className="aspect-square w-full rounded-lg object-contain"
      />

      <p className="mt-2 text-center text-xs text-slate-400">
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

function SkeletonGrid({
  count,
  className = "grid grid-cols-2 gap-3 sm:grid-cols-3",
}: {
  count: number;
  className?: string;
}) {
  return (
    <div className={className} aria-hidden="true">
      {Array.from({ length: count }, (_, i) => (
        <div
          key={i}
          className="dl-skeleton rounded-xl border border-white/10 bg-black/30 p-2"
          style={{ animationDelay: `${i * 80}ms` }}
        >
          <div className="aspect-square w-full rounded-lg bg-white/[0.06]" />
          <div className="mx-auto mt-3 h-2.5 w-2/3 rounded-full bg-white/[0.06]" />
        </div>
      ))}
    </div>
  );
}

function LoadingNote({ text }: { text: string }) {
  return (
    <p
      role="status"
      aria-live="polite"
      className="mt-5 flex items-center justify-center gap-2 text-sm text-violet-300"
    >
      <span className="dl-spinner" aria-hidden="true" />
      {text}
    </p>
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
    <div className="rounded-xl border border-white/[0.08] bg-white/[0.035] p-5">
      <p className="text-xs font-semibold tracking-[0.2em] text-slate-400">
        {label}
      </p>
      <h3 className="mt-3 text-2xl font-semibold">{value}</h3>
      <p className="mt-1 text-xs text-slate-400">{description}</p>
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("Studio");

  const [representationDigit, setRepresentationDigit] = useState(7);
  const [representationTimestep, setRepresentationTimestep] = useState(200);
  const [representationSteps, setRepresentationSteps] = useState(50);
  const [representationGuidance, setRepresentationGuidance] = useState(1.2);
  const [representationSeed, setRepresentationSeed] = useState(42);
  const [representation, setRepresentation] = useState<any>(null);
  const [coloredHeatmaps, setColoredHeatmaps] = useState<string[]>([]);
  const [representationLoading, setRepresentationLoading] = useState(false);
  const [representationError, setRepresentationError] = useState("");
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [evolutionTimesteps, setEvolutionTimesteps] = useState([350, 250, 150, 50]);
  const [evolutionData, setEvolutionData] = useState<any>(null);
  const [evolutionLoading, setEvolutionLoading] = useState(false);
  const [evolutionError, setEvolutionError] = useState("");
  const [selectedEvolutionIndex, setSelectedEvolutionIndex] = useState(0);
  const [coloredEvolutionImages, setColoredEvolutionImages] = useState<string[]>([]);

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
  const [engineMode, setEngineModeState] = useState<EngineMode | null>(null);
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

  const [isVisualPlaying, setIsVisualPlaying] = useState(false);
  const [visualPlaybackSpeed, setVisualPlaybackSpeed] = useState(1);
  const [visualLoop, setVisualLoop] = useState(false);

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
    if (!isVisualPlaying || !visualization) return;

    const totalFrames = visualization.snapshots.length;

    if (totalFrames <= 1) {
      setIsVisualPlaying(false);
      return;
    }

    const interval = window.setInterval(() => {
      setActiveFrame((current) => {
        if (current >= totalFrames - 1) {
          if (visualLoop) return 0;

          setIsVisualPlaying(false);
          return current;
        }

        return current + 1;
      });
    }, 1000 / visualPlaybackSpeed);

    return () => window.clearInterval(interval);
  }, [isVisualPlaying, visualPlaybackSpeed, visualLoop, visualization]);

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

  // Track whether requests go to the laptop server or run in this browser.
  // The first time we land in browser mode, use lighter Studio defaults.
  useEffect(() => {
    let lightened = false;
    return onEngineModeChange((next) => {
      setEngineModeState(next);
      if (next === "browser") {
        setBackendDevice("browser");
        if (!lightened) {
          lightened = true;
          setSteps((value) => (value === 50 ? 20 : value));
          setCount((value) => (value === 8 ? 4 : value));
        }
      }
    });
  }, []);

  useEffect(() => {
    let mounted = true;

    async function checkBackend() {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 4000);
        const response = await fetch(`${API}/api/info`, {
          signal: controller.signal,
        }).finally(() => clearTimeout(timer));

        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (mounted) {
          setEngineMode("server");
          setBackendOnline(true);
          setBackendDevice(data.device || "Unknown");
          setImageSize(data.img_size ?? null);
        }
      } catch {
        // Laptop server not reachable: run the model in this browser instead
        if (mounted) {
          setEngineMode("browser");
          setBackendOnline(true);
          setBackendDevice("browser");
          setImageSize(32);
          preloadRuntime();
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
      const response = await apiFetch("/api/generate", {
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
      const response = await apiFetch("/api/compare", {
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

      const response = await apiFetch("/api/experiment", {
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
      const response = await apiFetch("/api/visualize", {
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

  function colorizeHeatmap(base64: string) {
    return new Promise<string>((resolve) => {
      const img = new Image();

      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext("2d");
        if (!ctx) return resolve(`data:image/png;base64,${base64}`);

        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(
          0, 0, canvas.width, canvas.height
        );

        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
          const v = data[i] / 255;

          data[i] = Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * v - 3))));
          data[i + 1] = Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * v - 2))));
          data[i + 2] = Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * v - 1))));
        }

        ctx.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL("image/png"));
      };

      img.src = `data:image/png;base64,${base64}`;
    });
  }

  async function runRepresentation() {
    setRepresentationLoading(true);
    setRepresentationError("");
    setRepresentation(null);
    setSelectedChannel(0);

    try {
      const response = await apiFetch("/api/representation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit: representationDigit,
          timestep: representationTimestep,
          steps: representationSteps,
          w: representationGuidance,
          seed: representationSeed,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Representation request failed (${response.status}): ${message}`
        );
      }

      const data = await response.json();

      if (!data.heatmaps || data.heatmaps.length === 0) {
        throw new Error("No activation heatmaps returned by backend.");
      }

      setRepresentation(data);
      const converted = await Promise.all(
        data.heatmaps.map((item: any) => colorizeHeatmap(item.image))
      );
      setColoredHeatmaps(converted);
    } catch (err) {
      setRepresentationError(
        err instanceof Error
          ? err.message
          : "Unable to load representations."
      );
    } finally {
      setRepresentationLoading(false);
    }
  }

  async function runRepresentationEvolution() {
    setEvolutionLoading(true);
    setEvolutionError("");
    setEvolutionData(null);
    setSelectedEvolutionIndex(0);

    try {
      const response = await apiFetch("/api/representation/evolution", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit: representationDigit,
          timesteps: evolutionTimesteps,
          steps: representationSteps,
          w: representationGuidance,
          seed: representationSeed,
          channel: selectedChannel,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `Evolution request failed (${response.status}): ${message}`
        );
      }

      const data = await response.json();

      if (!data.evolution || data.evolution.length === 0) {
        throw new Error("No timestep evolution data returned.");
      }

      setEvolutionData(data);
      const convertedEvolution = await Promise.all(
        data.evolution.map((item: any) => colorizeHeatmap(item.image))
      );

      setColoredEvolutionImages(convertedEvolution);
    } catch (err) {
      setEvolutionError(
        err instanceof Error
          ? err.message
          : "Unable to load timestep evolution."
      );
    } finally {
      setEvolutionLoading(false);
    }
  }

  async function runInterpolation() {
    setInterpolationLoading(true);
    setInterpolationError("");
    setInterpolation(null);
    setSelectedInterpFrame(0);
    setIsPlaying(false);

    try {
      const response = await apiFetch("/api/interpolate", {
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
      const response = await apiFetch("/api/evaluate", {
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
    ["◉", "Representation Explorer"],
    ["⟿", "Recognition Timeline"],
    ["✎", "Sketch to Digit"],
  ];

  return (
    <main className="min-h-screen bg-canvas text-white selection:bg-violet-500/30">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-40 top-20 h-96 w-96 rounded-full bg-indigo-600/10 blur-[130px]" />
        <div className="absolute right-0 top-40 h-96 w-96 rounded-full bg-violet-600/10 blur-[130px]" />
        <div className="absolute bottom-0 left-1/2 h-80 w-80 rounded-full bg-violet-500/5 blur-[130px]" />
      </div>

      <div className="relative mx-auto flex min-h-screen max-w-[1600px]">
        <aside className="hidden w-64 shrink-0 border-r border-white/[0.07] bg-white/[0.015] p-6 lg:flex lg:flex-col">
          <div className="mb-12 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-xl font-bold">
              D
            </div>
            <div>
              <h1 className="text-lg font-semibold">DiffusionLab</h1>
              <p className="whitespace-nowrap text-xs uppercase tracking-[0.12em] text-slate-400">
                Generative Studio
              </p>
            </div>
          </div>

          <p className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Workspace
          </p>

          <nav className="space-y-1.5" role="tablist" aria-label="Workspace sections">
            {tabs.map(([icon, name]) => (
              <button
                key={name}
                role="tab"
                aria-selected={activeTab === name}
                onClick={() => setActiveTab(name)}
                className={`flex w-full items-center gap-3 rounded-xl border px-4 py-3 text-left text-sm transition ${activeTab === name
                  ? "border-violet-400/25 bg-violet-500/10 text-white shadow-[inset_3px_0_0_0_rgb(139_92_246)]"
                  : "border-transparent text-slate-400 hover:bg-white/5 hover:text-white"
                  }`}
              >
                <span className="w-5 shrink-0 text-center text-lg" aria-hidden="true">
                  {icon}
                </span>
                {name}
              </button>
            ))}
          </nav>

          <div className="mt-auto rounded-xl border border-white/10 bg-white/[0.04] p-4">
            <div className="mb-3 flex items-center gap-2">
              <span
                aria-hidden="true"
                className={`h-2 w-2 rounded-full ${engineMode === "browser"
                  ? "bg-violet-400"
                  : backendOnline === true
                  ? "bg-emerald-400"
                  : backendOnline === false
                    ? "bg-red-400"
                    : "bg-amber-400"
                  }`}
              />
              <span className="text-xs text-slate-300">
                {engineMode === "browser"
                  ? "Browser Mode"
                  : backendOnline === true
                  ? "API Connected"
                  : backendOnline === false
                    ? "API Offline"
                    : "Checking API..."}
              </span>
            </div>
            <p className="text-sm font-medium">MNIST Model</p>
            <p className="mt-1 text-xs text-slate-400">
              Conditional DDPM · EMA
            </p>
            <p className="mt-3 text-xs text-slate-400">
              {engineMode === "browser"
                ? "ONNX Runtime Web · DDIM"
                : "FastAPI · PyTorch · DDIM"}
            </p>
          </div>
        </aside>

        <section className="min-w-0 flex-1 px-5 py-7 sm:px-8 lg:px-10">
          <div
            className="-mx-5 mb-6 flex gap-2 overflow-x-auto px-5 pb-1 sm:-mx-8 sm:px-8 lg:hidden"
            role="tablist"
            aria-label="Workspace sections"
          >
            {tabs.map(([icon, name]) => (
              <button
                key={name}
                role="tab"
                aria-selected={activeTab === name}
                onClick={() => setActiveTab(name)}
                className={`shrink-0 rounded-xl border px-3 py-2 text-xs ${activeTab === name
                  ? "border-violet-400/30 bg-violet-500/15 text-white"
                  : "border-white/10 bg-white/5 text-slate-400"
                  }`}
              >
                <span aria-hidden="true">{icon}</span> {name}
              </button>
            ))}
          </div>

          <header className="mb-10 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="mb-2 text-xs tracking-widest text-violet-300">
                YOUR GENERATIVE WORKSPACE
              </p>
              <h2 className="text-3xl font-semibold sm:text-4xl">
                {TAB_TITLES[activeTab] ?? activeTab}
                <span className="text-violet-400">.</span>
              </h2>
              <p className="mt-2 text-sm text-slate-400">
                {TAB_SUBTITLES[activeTab] ?? ""}
              </p>
            </div>

            <div className="flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.04] px-4 py-2.5">
              <span
                aria-hidden="true"
                className={`h-2 w-2 rounded-full ${engineMode === "browser"
                  ? "bg-violet-400"
                  : backendOnline === true
                  ? "bg-emerald-400"
                  : backendOnline === false
                    ? "bg-red-400"
                    : "bg-amber-400"
                  }`}
              />
              <span className="text-xs text-slate-300" role="status" aria-live="polite">
                {engineMode === "browser"
                  ? "Running in your browser"
                  : backendOnline === true
                  ? `Backend Online · ${backendDevice.toUpperCase()}`
                  : backendOnline === false
                    ? "Backend Offline"
                    : "Checking Backend..."}
              </span>
            </div>
          </header>

          {engineMode === "browser" && (
            <div
              role="note"
              className="mb-6 flex items-start gap-3 rounded-xl border border-violet-400/20 bg-violet-500/[0.06] px-4 py-3 text-sm text-slate-300"
            >
              <span className="mt-0.5 text-violet-300" aria-hidden="true">
                ◈
              </span>
              <p>
                The laptop server is offline, so the model is running on your
                device. The first run downloads it (about 25 MB). Fewer images
                and fewer sampling steps finish faster.
              </p>
            </div>
          )}

          {activeTab === "Recognition Timeline" && (
            <RecognitionTimeline backendOnline={backendOnline} />
          )}

          {activeTab === "Sketch to Digit" && (
            <SketchToDigit backendOnline={backendOnline} />
          )}

          {activeTab === "Denoising Visualizer" && (
            <section className="space-y-6">
              <div className="rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-6 sm:p-8">
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
                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
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
                          aria-pressed={visualDigit === i}
                          aria-label={`Digit ${i}`}
                          className={`aspect-square rounded-xl border transition text-lg font-semibold ${visualDigit === i
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
                      <p className="mt-1 text-xs text-slate-400">
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
                          className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none transition focus:border-violet-400/60 focus:ring-2 focus:ring-violet-500/30"
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
                    className="mt-8 w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
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

                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs tracking-widest text-violet-300">
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
                    <div className="mt-6 flex min-h-[380px] flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                      <div className="flex h-28 w-28 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-6xl text-violet-300">
                        {visualDigit}
                      </div>
                      <h4 className="mt-6 text-lg font-medium">
                        Ready to visualize
                      </h4>
                      <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
                        Configure the sampling parameters and run the
                        visualizer. The first frame represents noise and the
                        final frame is the generated sample.
                      </p>
                    </div>
                  )}

                  {visualizationLoading && (
                    <div className="dl-skeleton mt-6 flex min-h-[380px] flex-col items-center justify-center rounded-xl border border-white/10 bg-black/20 p-8 text-center">
                      <LoadingNote text="Running the DDIM sampler and capturing intermediate states..." />
                    </div>
                  )}

                  {visualization && visualization.snapshots.length > 0 && (
                    <div className="mt-6">
                      <div className="rounded-xl border border-white/10 bg-black/30 p-5">
                        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <p className="text-xs text-slate-400">
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

                        <div className="mx-auto max-w-[420px] rounded-xl border border-white/10 bg-canvas p-4">
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


                        <div className="mt-5 space-y-4">
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <button
                              onClick={() => {
                                if (!visualization) return;

                                if (activeFrame >= visualization.snapshots.length - 1) {
                                  setActiveFrame(0);
                                }

                                setIsVisualPlaying((playing) => !playing);
                              }}
                              className="rounded-xl bg-violet-600 px-5 py-2 text-sm font-semibold hover:bg-violet-500"
                            >
                              {isVisualPlaying ? "Pause" : "Play"}
                            </button>

                            <div className="flex items-center gap-2">
                              <label className="text-xs text-slate-400">Speed</label>
                              <select
                                value={visualPlaybackSpeed}
                                onChange={(e) => setVisualPlaybackSpeed(Number(e.target.value))}
                                className="rounded-lg border border-white/10 bg-surface px-3 py-2 text-sm"
                              >
                                <option value={0.5}>0.5x</option>
                                <option value={1}>1x</option>
                                <option value={2}>2x</option>
                              </select>
                            </div>

                            <label className="flex items-center gap-2 text-xs text-slate-400">
                              <input
                                type="checkbox"
                                checked={visualLoop}
                                onChange={(e) => setVisualLoop(e.target.checked)}
                                className="accent-violet-500"
                              />
                              Loop
                            </label>
                          </div>

                          <input
                            type="range"
                            min="0"
                            max={visualization.snapshots.length - 1}
                            value={activeFrame}
                            onChange={(e) => {
                              setActiveFrame(Number(e.target.value));
                              setIsVisualPlaying(false);
                            }}
                            className="w-full accent-violet-500"
                          />
                        </div>

                        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500"
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
                        <p className="mb-3 text-xs font-semibold tracking-widest text-slate-400">
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
                                className="aspect-square w-full rounded-lg bg-canvas object-contain"
                              />
                              <p className="mt-2 text-xs text-slate-400">
                                {index === 0
                                  ? "Initial noise"
                                  : index ===
                                    visualization.snapshots.length - 1
                                    ? "Final image"
                                    : `Frame ${index + 1}`}
                              </p>
                              <p className="text-xs text-slate-500">
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

          {activeTab === "Representation Explorer" && (
            <section className="space-y-6">
              <div className="rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-6 sm:p-8">
                <p className="text-xs tracking-widest text-violet-300">
                  MODEL INTERNALS / FEATURE ANALYSIS
                </p>
                <h2 className="mt-3 text-2xl font-semibold">
                  Latent & Representation Explorer
                </h2>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                  Explore internal UNet bottleneck activations during
                  class-conditional DDIM sampling. Inspect feature channels,
                  activation patterns and timestep-dependent representations.
                </p>
              </div>

              {/* Representation Controls */}
              <div className="mt-6 rounded-2xl border border-white/10 bg-surface/80 p-6 space-y-6">
                <div>
                  <h3 className="text-lg font-semibold">Analysis Configuration</h3>
                  <p className="mt-1 text-sm text-slate-400">
                    Configure the input digit and diffusion sampling parameters.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div>
                    <label className="text-sm text-slate-400">Select Digit</label>
                    <select
                      value={representationDigit}
                      onChange={(e) => setRepresentationDigit(Number(e.target.value))}
                      className="mt-2 w-full rounded-xl bg-surface-2 border border-white/10 p-3"
                    >
                      {Array.from({ length: 10 }, (_, i) => (
                        <option key={i} value={i}>{i}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="text-sm text-slate-400">Timestep: {representationTimestep}</label>
                    <input
                      type="range"
                      min="0"
                      max="399"
                      value={representationTimestep}
                      onChange={(e) => setRepresentationTimestep(Number(e.target.value))}
                      className="mt-4 w-full accent-violet-500"
                    />
                  </div>

                  <div>
                    <label className="text-sm text-slate-400">DDIM Steps: {representationSteps}</label>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      value={representationSteps}
                      onChange={(e) => setRepresentationSteps(Number(e.target.value))}
                      className="mt-4 w-full accent-violet-500"
                    />
                  </div>

                  <div>
                    <label className="text-sm text-slate-400">
                      Guidance Scale: {representationGuidance.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="3"
                      step="0.1"
                      value={representationGuidance}
                      onChange={(e) => setRepresentationGuidance(Number(e.target.value))}
                      className="mt-4 w-full accent-violet-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-sm text-slate-400">Random Seed</label>
                  <input
                    type="number"
                    value={representationSeed}
                    onChange={(e) => setRepresentationSeed(Number(e.target.value))}
                    className="mt-2 w-full rounded-xl bg-surface-2 border border-white/10 p-3"
                  />
                </div>

                <button
                  onClick={runRepresentation}
                  disabled={representationLoading}
                  className="w-full rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-5 py-3 font-semibold transition"
                >
                  {representationLoading ? "Analyzing Features..." : "Explore Representations"}
                </button>

                {representationError && (
                  <p className="text-sm text-red-400">{representationError}</p>
                )}
              </div>


              {representation && (
                <>
                  <div className="mt-6 space-y-6">

                    {/* Feature Statistics */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      {[
                        ["Mean", representation.statistics.mean],
                        ["Std Deviation", representation.statistics.std],
                        ["Minimum", representation.statistics.min],
                        ["Maximum", representation.statistics.max],
                      ].map(([label, value]) => (
                        <div
                          key={String(label)}
                          className="rounded-xl border border-white/10 bg-surface p-5"
                        >
                          <p className="text-xs uppercase tracking-widest text-slate-400">
                            {label}
                          </p>
                          <p className="mt-2 text-xl font-semibold tabular-nums">
                            {Number(value).toFixed(4)}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* Feature Information */}
                    <div className="rounded-xl border border-white/10 bg-surface p-5">
                      <h3 className="text-lg font-semibold">
                        Bottleneck Feature Analysis
                      </h3>
                      <p className="mt-2 text-sm text-slate-400">
                        Digit {representation.digit} · Requested timestep{" "}
                        {representation.requested_timestep} · Actual timestep{" "}
                        {representation.actual_timestep}
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        Feature shape: {representation.feature_shape.join(" × ")}
                      </p>
                    </div>

                    {/* Channel Heatmaps */}
                    <div className="rounded-2xl border border-white/10 bg-surface/80 p-6">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <h3 className="text-lg font-semibold">
                            Activation Heatmaps
                          </h3>
                          <p className="text-sm text-slate-400 mt-1">
                            Select a channel to inspect its spatial activation pattern.
                          </p>
                        </div>
                        <span className="text-xs text-slate-400">
                          {representation.heatmaps.length} channels
                        </span>
                      </div>

                      {/* Activation Intensity Legend */}
                      <div className="mt-5 rounded-xl border border-white/10 bg-surface-2 p-4">
                        <div className="flex items-center justify-between mb-3">
                          <p className="text-sm font-medium text-slate-300">
                            Activation Intensity
                          </p>
                          <p className="text-xs text-slate-400">
                            Normalized · 0 to 1
                          </p>
                        </div>

                        <div
                          className="h-4 w-full rounded-full"
                          style={{
                            background:
                              "linear-gradient(to right, #0000ff, #00ffff, #ffff00, #ff0000)",
                          }}
                        />

                        <div className="flex justify-between mt-2 text-xs text-slate-400">
                          <span>Low</span>
                          <span>Moderate</span>
                          <span>High</span>
                          <span>Very High</span>
                        </div>
                      </div>

                      {/* Channel Grid */}
                      <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                        {representation.heatmaps.map((item: any) => (
                          <button
                            key={item.channel}
                            onClick={() => setSelectedChannel(item.channel)}
                            className={`rounded-xl border p-2 transition ${selectedChannel === item.channel
                              ? "border-violet-500 bg-violet-500/10"
                              : "border-white/10 bg-surface-2 hover:border-violet-400/50"
                              }`}
                          >
                            <img
                              src={coloredHeatmaps[item.channel]}
                              alt={`Activation channel ${item.channel}`}
                              className="w-full aspect-square rounded-lg object-contain bg-black"
                            />
                            <p className="mt-2 text-xs text-slate-400">
                              Channel {item.channel}
                            </p>
                          </button>
                        ))}
                      </div>

                      {/* Selected Channel */}
                      {representation.heatmaps.find(
                        (item: any) => item.channel === selectedChannel
                      ) && (
                          <div className="mt-6 border-t border-white/10 pt-6">
                            <h4 className="text-sm font-semibold mb-3">
                              Selected Channel: {selectedChannel}
                            </h4>
                            <div className="flex justify-center rounded-xl bg-black/30 p-5">
                              <img
                                src={coloredHeatmaps[selectedChannel]}
                                alt={`Selected activation channel ${selectedChannel}`}
                                className="w-full max-w-sm rounded-xl"
                              />
                            </div>
                          </div>
                        )}
                    </div>
                  </div>

                  {/* Timestep Feature Evolution */}
                  <div className="rounded-2xl border border-violet-400/20 bg-surface/80 p-6 space-y-6">
                    <div>
                      <p className="text-xs tracking-widest text-violet-300">
                        TEMPORAL FEATURE ANALYSIS
                      </p>
                      <h3 className="mt-2 text-xl font-semibold">
                        Timestep-wise Feature Evolution
                      </h3>
                      <p className="mt-2 text-sm text-slate-400">
                        Compare how the selected UNet bottleneck channel changes
                        across the DDIM denoising trajectory.
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <p className="text-sm text-slate-300">
                        Selected channel: <strong>{selectedChannel}</strong>
                      </p>

                      <button
                        onClick={runRepresentationEvolution}
                        disabled={evolutionLoading}
                        className="rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-5 py-3 text-sm font-semibold transition"
                      >
                        {evolutionLoading
                          ? "Analyzing Evolution..."
                          : "Analyze Feature Evolution"}
                      </button>
                    </div>

                    {evolutionError && (
                      <p className="text-sm text-red-400">
                        {evolutionError}
                      </p>
                    )}

                    {evolutionData && (
                      <div className="space-y-5">
                        <div className="flex flex-wrap gap-2">
                          {evolutionData.evolution.map((item: any, index: number) => (
                            <button
                              key={item.timestep}
                              onClick={() => setSelectedEvolutionIndex(index)}
                              className={`rounded-lg border px-4 py-2 text-sm ${selectedEvolutionIndex === index
                                ? "border-violet-400 bg-violet-500/15 text-violet-300"
                                : "border-white/10 bg-surface-2 text-slate-400"
                                }`}
                            >
                              t = {item.timestep}
                            </button>
                          ))}
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                          {evolutionData.evolution.map((item: any, index: number) => (
                            <button
                              key={item.timestep}
                              onClick={() => setSelectedEvolutionIndex(index)}
                              className={`rounded-xl border p-4 text-left ${selectedEvolutionIndex === index
                                ? "border-violet-400 bg-violet-500/10"
                                : "border-white/10 bg-surface-2"
                                }`}
                            >
                              <p className="mb-3 text-sm font-semibold text-slate-300">
                                Timestep {item.timestep}
                              </p>
                              <img
                                src={coloredEvolutionImages[index]}
                                alt={`Channel ${selectedChannel} at timestep ${item.timestep}`}
                                className="w-full max-w-xs mx-auto rounded-xl"
                              />
                              <div className="mt-3 flex justify-between text-xs text-slate-400">
                                <span>Mean: {item.mean.toFixed(4)}</span>
                                <span>Std: {item.std.toFixed(4)}</span>
                              </div>
                            </button>
                          ))}
                        </div>

                        <div className="rounded-xl border border-white/10 bg-surface-2 p-4">
                          <p className="text-sm text-slate-300">
                            Digit {evolutionData.digit} · Channel {evolutionData.channel}
                          </p>
                          <p className="mt-2 text-xs text-slate-400">
                            Seed: {evolutionData.seed} · Guidance: {evolutionData.guidance}
                            · DDIM steps: {evolutionData.steps}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </section>
          )}

          {activeTab === "Noise Interpolation" && (
            <section className="space-y-6">
              <div className="rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-6 sm:p-8">
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
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
                  <h3 className="text-lg font-semibold">Interpolation Settings</h3>

                  <div className="mt-6 space-y-5">
                    <div>
                      <label className="text-sm text-slate-400">Target Digit</label>
                      <select
                        value={interpDigit}
                        onChange={(e) => setInterpDigit(Number(e.target.value))}
                        className="mt-2 w-full rounded-xl border border-white/10 bg-surface-2 p-3 text-white"
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
                          className="mt-2 w-full rounded-xl border border-white/10 bg-surface-2 p-3 text-sm text-white"
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
                          className="mt-2 w-full rounded-xl border border-white/10 bg-surface-2 p-3 text-sm text-white"
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

                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
                  {!interpolation ? (
                    <div className="flex min-h-[400px] flex-col items-center justify-center text-center">
                      <div className="text-5xl text-violet-400">◈</div>
                      <h3 className="mt-5 text-xl font-semibold">
                        Ready to interpolate
                      </h3>
                      <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
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
                          <p className="mt-1 text-xs text-slate-400">
                            Target digit: {interpolation.digit}
                          </p>
                        </div>
                        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-300">
                          {interpolation.count} frames generated
                        </span>
                      </div>

                      <div className="mt-7 grid grid-cols-1 gap-5 sm:grid-cols-2">
                        <div className="rounded-xl border border-white/10 bg-canvas p-4">
                          <p className="mb-3 text-xs text-slate-400">Starting Noise · Seed {seedA}</p>
                          <img
                            src={`data:image/png;base64,${interpolation.frames[0].image}`}
                            alt="Starting interpolation frame"
                            className="aspect-square w-full rounded-xl object-contain"
                          />
                        </div>

                        <div className="rounded-xl border border-white/10 bg-canvas p-4">
                          <p className="mb-3 text-xs text-slate-400">Ending Noise · Seed {seedB}</p>
                          <img
                            src={`data:image/png;base64,${interpolation.frames[interpolation.frames.length - 1].image}`}
                            alt="Ending interpolation frame"
                            className="aspect-square w-full rounded-xl object-contain"
                          />
                        </div>
                      </div>

                      <div className="mt-8 rounded-xl border border-violet-400/20 bg-violet-500/[0.05] p-5">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <h4 className="font-semibold">Explore Transition</h4>
                          <span className="text-sm text-violet-300">
                            α = {interpolation.frames[selectedInterpFrame]?.alpha.toFixed(3)}
                          </span>
                        </div>

                        <div className="mx-auto mt-5 max-w-sm rounded-xl border border-white/10 bg-canvas p-4">
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

                      <p className="mt-5 text-xs text-slate-400">
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
                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
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
                          aria-pressed={digit === i}
                          aria-label={`Digit ${i}`}
                          className={`aspect-square rounded-xl border text-lg font-semibold transition ${digit === i
                            ? "border-violet-400 bg-gradient-to-br from-violet-500 to-indigo-600 text-white"
                            : "border-white/10 bg-white/[0.035] text-slate-400 hover:border-violet-400/40 hover:bg-white/10 hover:text-white"
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
                          className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none transition focus:border-violet-400/60 focus:ring-2 focus:ring-violet-500/30"
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
                    className="mt-9 w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
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

                <div className="flex min-h-[480px] flex-col rounded-2xl border border-white/10 bg-surface/70 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-violet-300">
                    02 / OUTPUT
                  </p>
                  <h3 className="mb-6 mt-2 text-xl font-semibold">
                    Generation Preview
                  </h3>

                  {loading ? (
                    <>
                      <SkeletonGrid count={count} />
                      <LoadingNote text={`Denoising ${count} sample${count === 1 ? "" : "s"} with ${steps} DDIM steps...`} />
                    </>
                  ) : images.length === 0 ? (
                    <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
                      <div className="mb-6 flex h-24 w-24 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-5xl text-violet-300">
                        {digit}
                      </div>
                      <h4 className="text-lg font-medium">
                        Your canvas is ready
                      </h4>
                      <p className="mt-2 max-w-xs text-sm text-slate-400">
                        Pick a digit, adjust the sliders, then press
                        Generate Images to create new handwritten samples.
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

              <section className="mt-8 rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
                <p className="text-xs tracking-widest text-violet-300">
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
                  className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {compareLoading
                    ? "◌ Comparing..."
                    : "✦ Compare Guidance — 0.0 vs 1.2 vs 2.0"}
                </button>

                {compareLoading && (
                  <div className="mt-8 space-y-6">
                    <LoadingNote text="Running three sampling experiments..." />
                    {[0, 1, 2].map((row) => (
                      <div
                        key={row}
                        className="rounded-xl border border-white/10 bg-black/20 p-5"
                      >
                        <div className="dl-skeleton mb-4 h-5 w-40 rounded-full bg-white/[0.06]" />
                        <SkeletonGrid
                          count={Math.min(count, 8)}
                          className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6"
                        />
                      </div>
                    ))}
                  </div>
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
                          className="rounded-xl border border-white/10 bg-black/20 p-5"
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
                              className="h-full bg-gradient-to-r from-indigo-500 to-violet-500"
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
              <div className="rounded-2xl border border-violet-400/20 bg-surface/80 p-6 sm:p-8">
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white"
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white"
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white"
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white"
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white disabled:opacity-40"
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white disabled:opacity-40"
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
                      className="w-full rounded-xl border border-white/10 bg-canvas p-3 text-white disabled:opacity-40"
                    />
                  </div>
                </div>

                <button
                  onClick={runExperiment}
                  disabled={experimentLoading || backendOnline !== true}
                  className="mt-7 w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
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
                  <LoadingNote text="Generating comparison samples..." />
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
                      <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/[0.06] p-5">
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
                    <div className="rounded-xl border border-white/10 bg-surface/80 p-5">
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
                                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all"
                                style={{ width: `${result.match_rate}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Inference Time Chart */}
                    <div className="rounded-xl border border-white/10 bg-surface/80 p-5">
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
                      className="rounded-xl border border-white/10 bg-surface/80 p-5"
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
                              className={`rounded-xl border p-4 transition-all ${metric.isBest
                                ? "border-emerald-400/50 bg-emerald-500/10 shadow-[0_0_15px_rgba(52,211,153,0.08)]"
                                : "border-white/10 bg-white/[0.03]"
                                }`}
                            >
                              <p className="text-xs font-semibold tracking-[0.18em] text-slate-400">
                                {metric.label}
                              </p>
                              <p className="mt-3 text-2xl font-bold text-white">
                                {metric.value}
                              </p>
                              <p className={`mt-1 text-xs ${metric.isBest ? "font-semibold text-emerald-400" : "text-slate-400"
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
                        className="rounded-xl border border-slate-700 bg-slate-900/50 p-5"
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
                  <p className="mt-1 text-sm text-slate-400">
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
                <div className="rounded-xl border border-white/10 p-10 text-center text-slate-400">
                  Loading saved generations...
                </div>
              ) : history.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-white/10 p-12 text-center">
                  <h3 className="text-lg font-semibold">
                    No generations saved yet
                  </h3>
                  <p className="mt-2 text-sm text-slate-400">
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
                      className="rounded-2xl border border-white/10 bg-surface/70 p-5 sm:p-7"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <h4 className="text-lg font-semibold">
                            Digit {record.digit}
                          </h4>
                          <p className="mt-1 text-xs text-slate-400">
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

                      <p className="mt-4 text-xs text-slate-400">
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
                <p className="mt-1 text-sm text-slate-400">
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
                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
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
                        <span className="text-sm text-slate-400">{label}</span>
                        <span className="text-right text-sm font-medium">
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
                  <p className="text-xs tracking-widest text-violet-300">
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
                      <h4 className="font-medium text-violet-300">
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

              <div className="mt-8 rounded-2xl border border-violet-400/20 bg-violet-500/[0.06] p-6">
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

              <div className="rounded-2xl border border-white/10 bg-surface/80 p-6 sm:p-8">
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
                      className="w-full rounded-xl border border-white/10 bg-canvas px-4 py-3 text-sm"
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
                  className="mt-8 w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-4 font-semibold shadow-lg shadow-violet-900/30 transition enabled:hover:brightness-110 enabled:active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
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
                    <p className="text-xs tracking-widest text-violet-300">
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

                  <div className="rounded-xl border border-white/10 bg-surface/80 p-5">
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
                          <p className="text-xs text-slate-400">{label}</p>
                          <p className="mt-1 font-semibold">{value}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-surface/80 p-6">
                    <h4 className="text-lg font-semibold">
                      Generated Evaluation Samples
                    </h4>
                    <p className="mt-1 text-sm text-slate-400">
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

                  <div className="rounded-xl border border-violet-400/20 bg-violet-500/[0.06] p-5">
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
                        <strong className="text-violet-300">Diversity:</strong>{" "}
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

          <footer className="mt-10 border-t border-white/[0.07] py-6 text-xs text-slate-500">
            DiffusionLab · PyTorch + FastAPI + Next.js
          </footer>
        </section>
      </div>
    </main>
  );
}