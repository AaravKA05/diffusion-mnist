"use client";

import { useEffect, useRef, useState } from "react";

const GITHUB_URL = "https://github.com/AaravKA05/diffusion-mnist";

function ArrowUpRight({ className = "h-4 w-4" }: { className?: string }) {
    return (
        <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={className}
            aria-hidden="true"
        >
            <path d="M7 17 17 7" />
            <path d="M7 7h10v10" />
        </svg>
    );
}

/**
 * Full-screen start page shown when the site opens.
 * The background is a video of the model's own denoising (made from
 * results/denoising.gif). It is a local file, so it works with the
 * cross-origin isolation headers that browser mode needs. People who ask
 * their system to reduce motion see the still poster instead.
 * The Studio is already loading behind this screen, so it opens at once.
 */
export default function LaunchScreen({ onEnter }: { onEnter: () => void }) {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const [leaving, setLeaving] = useState(false);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
        video.play().catch(() => {
            // Autoplay can be blocked; the poster stays visible
        });
        return () => video.pause();
    }, []);

    function enter() {
        if (leaving) return;
        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (reduceMotion) {
            onEnter();
            return;
        }
        setLeaving(true);
        window.setTimeout(onEnter, 450);
    }

    return (
        <section
            aria-labelledby="launch-heading"
            className={`fixed inset-0 z-50 flex flex-col overflow-y-auto overscroll-contain bg-canvas text-white transition-[opacity,transform] duration-500 ease-out ${leaving ? "pointer-events-none scale-[1.02] opacity-0" : "opacity-100"
                }`}
        >
            {/* Background: the model turning noise into digits */}
            <video
                ref={videoRef}
                muted
                loop
                playsInline
                preload="auto"
                poster="/media/denoising-poster.jpg"
                aria-hidden="true"
                className="pointer-events-none fixed inset-0 z-0 h-full w-full object-cover opacity-20 grayscale"
            >
                <source src="/media/denoising-bg.webm" type="video/webm" />
                <source src="/media/denoising-bg.mp4" type="video/mp4" />
            </video>

            {/* Navy tint on the noise, fades, and a coral glow so the text stays readable */}
            <div className="pointer-events-none fixed inset-0 z-[1] bg-[#1e2f5c] opacity-45 mix-blend-color" />
            <div className="pointer-events-none fixed inset-x-0 top-0 z-[1] h-48 bg-gradient-to-b from-canvas to-transparent" />
            <div className="pointer-events-none fixed inset-x-0 bottom-0 z-[1] h-64 bg-gradient-to-t from-canvas via-canvas/80 to-transparent" />
            <div className="pointer-events-none fixed inset-0 z-[1] bg-[radial-gradient(ellipse_at_center,rgba(255,90,79,0.18),transparent_65%)]" />

            {/* Top bar */}
            <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between px-5 pt-6 sm:px-10">
                <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-lg font-bold">
                        D
                    </div>
                    <span className="font-semibold">DiffusionLab</span>
                </div>
                <a
                    href={GITHUB_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-sm text-slate-300 transition hover:text-white"
                >
                    GitHub
                    <ArrowUpRight className="h-3.5 w-3.5" />
                </a>
            </header>

            {/* Centre */}
            <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-5 py-16 text-center sm:px-10">
                <p className="text-xs font-medium uppercase tracking-[0.25em] text-violet-400">
                    Machine Learning (UML501)
                </p>

                <h1
                    id="launch-heading"
                    className="mx-auto mt-5 max-w-4xl text-4xl leading-[1.05] font-semibold tracking-tight text-white sm:text-6xl lg:text-7xl"
                >
                    From pure noise to a{" "}
                    <span className="bg-gradient-to-r from-violet-300 to-violet-500 bg-clip-text text-transparent">
                        handwritten digit.
                    </span>
                </h1>

                <p className="mx-auto mt-6 max-w-xl text-sm leading-6 text-slate-300 sm:text-base sm:leading-7">
                    A class-conditional diffusion model built from scratch in PyTorch. It runs on
                    our laptop server, or right here in your browser when the server is off.
                </p>

                <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
                    <button
                        onClick={enter}
                        className="liquid-glass flex items-center gap-2 rounded-full px-7 py-3.5 text-sm font-medium text-white hover:bg-white/10 sm:text-base"
                    >
                        Open the Studio
                        <ArrowUpRight className="h-5 w-5" />
                    </button>
                    <a
                        href={GITHUB_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 rounded-full bg-white px-7 py-3.5 text-sm font-medium text-black transition hover:bg-white/90 sm:text-base"
                    >
                        View on GitHub
                        <ArrowUpRight />
                    </a>
                </div>

                <p className="mt-7 text-xs text-slate-400">
                    PyTorch · FastAPI · Next.js · ONNX Runtime Web
                </p>
            </div>

        </section>
    );
}