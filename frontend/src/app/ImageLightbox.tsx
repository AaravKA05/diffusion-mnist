"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Opened = {
  src: string;
  alt: string;
  caption: string;
};

/** Text that sits with the image in its card, e.g. "Read as 7 · 99.8%". */
function findCaption(img: HTMLImageElement) {
  const figcaption = img.closest("figure")?.querySelector("figcaption")?.textContent;
  if (figcaption) return figcaption.trim();

  const card = img.parentElement;
  if (!card) return "";
  const parts = Array.from(card.children)
    .filter((el) => el !== img && el.tagName !== "BUTTON")
    .map((el) => el.textContent?.trim() ?? "")
    .filter(Boolean);
  return parts.join(" · ").slice(0, 120);
}

/**
 * Click any generated image on the page to see it large.
 * Works for every tab without changing each image: it listens for clicks on
 * images whose source is an in-page PNG (the model outputs).
 */
export default function ImageLightbox() {
  const [opened, setOpened] = useState<Opened | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const lastFocus = useRef<HTMLElement | null>(null);

  const close = useCallback(() => {
    setOpened(null);
    lastFocus.current?.focus?.();
  }, []);

  useEffect(() => {
    function onClick(event: MouseEvent) {
      const target = event.target as HTMLElement | null;
      if (!(target instanceof HTMLImageElement)) return;
      if (!target.src.startsWith("data:image")) return;
      if (target.closest("[data-lightbox]")) return;
      if (!target.closest("main")) return;

      lastFocus.current = document.activeElement as HTMLElement | null;
      setOpened({ src: target.src, alt: target.alt, caption: findCaption(target) });
    }

    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  useEffect(() => {
    if (!opened) return;

    closeRef.current?.focus();

    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") close();
    }

    document.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [opened, close]);

  if (!opened) return null;

  return (
    <div
      data-lightbox
      role="dialog"
      aria-modal="true"
      aria-label={opened.alt || "Enlarged image"}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
      onClick={close}
    >
      <div
        className="dl-fade-in w-full max-w-lg rounded-2xl border border-white/10 bg-surface p-4 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <img
          src={opened.src}
          alt={opened.alt}
          className="aspect-square w-full rounded-xl bg-black object-contain"
        />

        {opened.caption && (
          <p className="mt-3 text-center text-sm text-slate-300">{opened.caption}</p>
        )}

        <div className="mt-4 flex gap-3">
          <a
            href={opened.src}
            download="diffusionlab-image.png"
            className="flex-1 rounded-xl border border-white/10 bg-white/5 py-2.5 text-center text-sm text-slate-200 hover:bg-white/10"
          >
            ↓ Download
          </a>
          <button
            ref={closeRef}
            onClick={close}
            className="flex-1 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-2.5 text-sm font-semibold"
          >
            Close
          </button>
        </div>
        <p className="mt-3 text-center text-xs text-slate-500">Press Esc or click outside to close</p>
      </div>
    </div>
  );
}
