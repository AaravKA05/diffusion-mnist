"use client";

import { useEffect, useRef } from "react";

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
 * Closing section with a background video of the model's own denoising
 * (made from results/denoising.gif). The video is a local file, so it works
 * with the cross-origin isolation headers that browser mode needs.
 * It only plays while it is on screen, and never for people who ask their
 * system to reduce motion (they see the still poster instead).
 */
export default function AboutFooter({ onOpenStudio }: { onOpenStudio: () => void }) {
    const videoRef = useRef<HTMLVideoElement | null>(null);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
        if (reduceMotion.matches) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    video.play().catch(() => {
                        // Autoplay can be blocked; the poster stays visible
                    });
                } else {
                    video.pause();
                }
            },
            { threshold: 0.15 }
        );

        observer.observe(video);
        return () => observer.disconnect();
    }, []);

    return (
        <section
            aria-labelledby="about-heading"
            className="relative mt-14 overflow-hidden rounded-2xl border border-white/10 bg-canvas px-6 pt-20 pb-8 text-center sm:px-12 sm:pt-28"
        >
            {/* Background: the model turning noise into digits */}
            <video
                ref={videoRef}
                muted
                loop
                playsInline
                preload="none"
                poster="/media/denoising-poster.jpg"
                aria-hidden="true"
                className="absolute inset-0 z-0 h-full w-full object-cover opacity-20 grayscale"
            >
                <source src="/media/denoising-bg.webm" type="video/webm" />
                <source src="/media/denoising-bg.mp4" type="video/mp4" />
            </video>

            {/* Navy tint on the noise, fades, and a coral glow so the text stays readable */}
            <div className="pointer-events-none absolute inset-0 z-[1] bg-[#1e2f5c] mix-blend-color opacity-45" />
            <div className="pointer-events-none absolute inset-x-0 top-0 z-[1] h-40 bg-gradient-to-b from-canvas to-transparent" />
            <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[1] h-56 bg-gradient-to-t from-canvas via-canvas/80 to-transparent" />
            <div className="pointer-events-none absolute inset-0 z-[1] bg-[radial-gradient(ellipse_at_center,rgba(255,90,79,0.18),transparent_65%)]" />

            <div className="relative z-10">
                <p className="text-xs font-medium uppercase tracking-[0.25em] text-violet-400">
                    DiffusionLab · Machine Learning (UML501)
                </p>

                <h2
                    id="about-heading"
                    className="mx-auto mt-5 max-w-3xl text-4xl leading-[1.05] font-semibold tracking-tight text-white sm:text-5xl lg:text-6xl"
                >
                    From pure noise to a{" "}
                    <span className="bg-gradient-to-r from-violet-300 to-violet-500 bg-clip-text text-transparent">
                        handwritten digit.
                    </span>
                </h2>

                <p className="mx-auto mt-5 max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
                    A class-conditional diffusion model built from scratch in PyTorch. It runs on
                    our laptop server, or right here in your browser when the server is off.
                </p>

                <div className="mt-9 flex flex-wrap items-center justify-center gap-4">
                    <button
                        onClick={onOpenStudio}
                        className="liquid-glass flex items-center gap-2 rounded-full px-6 py-3 text-sm font-medium text-white enabled:hover:bg-white/10"
                    >
                        Open the Studio
                        <ArrowUpRight className="h-5 w-5" />
                    </button>
                    <a
                        href={GITHUB_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 rounded-full bg-white px-6 py-3 text-sm font-medium text-black transition hover:bg-white/90"
                    >
                        View on GitHub
                        <ArrowUpRight />
                    </a>
                </div>

                <p className="mt-6 text-xs text-slate-400">
                    PyTorch · FastAPI · Next.js · ONNX Runtime Web
                </p>

                {/* Footer bar */}
                <div className="mt-20 flex flex-col items-center justify-between gap-4 border-t border-white/10 pt-6 sm:mt-24 md:flex-row">
                    <p className="text-xs text-slate-400">
                        © 2026 Aarav K Arora &amp; Nikunj Deewan · Thapar Institute of Engineering and
                        Technology
                    </p>
                    <nav aria-label="Footer" className="flex items-center gap-6">
                        <a
                            href={GITHUB_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-slate-400 transition hover:text-white"
                        >
                            GitHub
                        </a>
                        <a
                            href={`${GITHUB_URL}#readme`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-slate-400 transition hover:text-white"
                        >
                            README
                        </a>
                        <button
                            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
                            className="text-xs text-slate-400 hover:text-white"
                        >
                            Back to top
                        </button>
                    </nav>
                </div>
            </div>
        </section>
    );
}