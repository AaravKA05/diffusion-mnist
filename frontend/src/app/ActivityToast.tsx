"use client";

import { useEffect, useState } from "react";

import { onActivityChange, type Activity } from "@/lib/engine/apiClient";
import { onInlineProgressChange } from "@/lib/inlineProgress";

/**
 * Small floating progress bar at the bottom of the screen.
 * Browser mode: real "step 7 of 20" progress.
 * Server mode: the laptop cannot report steps, so the bar just moves.
 */
export default function ActivityToast() {
  const [activity, setActivity] = useState<Activity | null>(null);

  // Hidden while the page shows the same progress next to the work
  const [inline, setInline] = useState(false);

  useEffect(() => onActivityChange(setActivity), []);
  useEffect(() => onInlineProgressChange(setInline), []);

  if (!activity || inline) return null;

  const known = activity.total !== null && activity.total > 0;
  const percent = known ? Math.min(100, Math.round((activity.done / (activity.total as number)) * 100)) : 0;
  const where = activity.mode === "browser" ? "in your browser" : "on the laptop server";

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-40 flex justify-center px-4">
      <div
        role="status"
        aria-live="polite"
        className="dl-fade-in pointer-events-auto w-full max-w-md rounded-2xl border border-white/10 bg-surface/95 px-4 py-3 shadow-2xl shadow-black/50 backdrop-blur"
      >
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="flex min-w-0 items-center gap-2 text-slate-200">
            <span className="dl-spinner shrink-0" aria-hidden="true" />
            <span className="truncate">
              {activity.label} {where}
            </span>
          </span>
          {known && (
            <span className="shrink-0 tabular-nums text-xs text-slate-400">
              step {activity.done} of {activity.total}
            </span>
          )}
        </div>

        <div
          className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-white/10"
          role="progressbar"
          aria-label={activity.label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={known ? percent : undefined}
        >
          {known ? (
            <div
              className="h-full rounded-full bg-violet-500 transition-[width] duration-200 ease-out"
              style={{ width: `${percent}%` }}
            />
          ) : (
            <div className="dl-indeterminate h-full w-1/3 rounded-full bg-violet-500" />
          )}
        </div>
      </div>
    </div>
  );
}
