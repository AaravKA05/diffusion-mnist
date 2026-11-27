"use client";

import { useEffect, useRef, useState } from "react";
import { FlaskConical, Gauge, PenLine, ScanEye, Waves, WandSparkles, type LucideIcon } from "lucide-react";

const GITHUB_URL = "https://github.com/AaravKA05/diffusion-mnist";

// What the lab contains, in the visitor's words (from the README).
const TOOLS: { icon: LucideIcon; name: string; text: string }[] = [
    {
        icon: WandSparkles,
        name: "Studio",
        text: "Pick a digit, guidance scale, sampling steps and seed, then generate samples the classifier checks.",
    },
    {
        icon: Waves,
        name: "Denoising visualizer",
        text: "Step through the reverse process and see each intermediate image.",
    },
    {
        icon: ScanEye,
        name: "Recognition timeline",
        text: "Find the sampling step where the classifier starts reading the digit you asked for.",
    },
    {
        icon: PenLine,
        name: "Sketch to digit",
        text: "Draw a digit with a mouse, finger or pen and let the model clean it up.",
    },
    {
        icon: FlaskConical,
        name: "Experiment lab",
        text: "Compare guidance values and sampling steps side by side.",
    },
    {
        icon: Gauge,
        name: "Evaluation",
        text: "Score generated digits with a classifier and check for copied training images.",
    },
];

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
            className={`fixed inset-0 z-50 flex flex-col overflow-y-auto overscroll-contain bg-canvas text-white transition-[opacity,transform] duration-500 ease-out ${
                leaving ? "pointer-events-none scale-[1.02] opacity-0" : "opacity-100"
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

            {/* Navy tint on the noise, plus fades so the text stays readable */}
            <div className="pointer-events-none fixed inset-0 z-[1] bg-[#1e2f5c] opacity-45 mix-blend-color" />
            <div className="pointer-events-none fixed inset-x-0 top-0 z-[1] h-48 bg-gradient-to-b from-canvas to-transparent" />

            <div className="relative z-10 mx-auto flex w-full max-w-7xl flex-col px-5 sm:px-10">
                {/* Top bar */}
                <header className="flex items-center justify-between gap-4 py-6">
                    <div className="flex items-center gap-2.5">
                        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-violet-500 text-base font-bold text-black">
                            D
                        </div>
                        <span className="text-[17px] font-semibold tracking-[-0.01em]">DiffusionLab</span>
                    </div>
                    <nav aria-label="Site" className="flex items-center gap-1 text-sm text-slate-300">
                        <a href="#lab" className="hidden min-h-11 items-center px-3 hover:text-white sm:flex">
                            What&apos;s inside
                        </a>
                        <a
                            href={GITHUB_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex min-h-11 items-center px-3 hover:text-white"
                        >
                            GitHub
                        </a>
                    </nav>
                </header>

                {/* Hero */}
                <div className="flex flex-col gap-7 py-20 sm:py-28">
                    <h1
                        id="launch-heading"
                        className="max-w-[14ch] text-5xl leading-[0.98] font-semibold tracking-[-0.035em] text-balance sm:text-7xl lg:text-8xl"
                    >
                        From pure noise to a handwritten digit.
                    </h1>
                    <div className="flex flex-wrap items-end justify-between gap-x-12 gap-y-6">
                        <p className="max-w-[54ch] text-base leading-7 text-slate-200 sm:text-lg">
                            A class-conditional diffusion model built from scratch in PyTorch. It runs on our
                            laptop server, or right here in your browser when the server is off.
                        </p>
                        <div className="flex flex-wrap gap-3">
                            <button
                                onClick={enter}
                                className="flex min-h-[52px] items-center rounded-lg bg-violet-500 px-6 font-semibold text-black hover:bg-violet-400"
                            >
                                Open the Studio
                            </button>
                            <a
                                href={GITHUB_URL}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex min-h-[52px] items-center rounded-lg border border-white/25 px-6 font-medium text-white hover:bg-white/5"
                            >
                                Read the code
                            </a>
                        </div>
                    </div>
                </div>
            </div>

            {/* Solid ground below the hero, so the list reads cleanly over the video */}
            <div className="relative z-10 mt-auto">
                <div aria-hidden="true" className="h-24 bg-gradient-to-b from-transparent to-canvas" />
                <div className="bg-canvas">
                    <div className="mx-auto w-full max-w-7xl px-5 sm:px-10">
                        {/* What's inside */}
                        <section
                            id="lab"
                            aria-labelledby="lab-heading"
                            className="flex flex-wrap gap-x-16 gap-y-8 border-t border-slate-700 py-12"
                        >
                            <div className="max-w-sm flex-[1_1_260px]">
                                <h2 id="lab-heading" className="text-3xl font-semibold tracking-[-0.02em]">
                                    What&apos;s inside the lab
                                </h2>
                                <p className="mt-3 text-slate-300">
                                    Tools for generating digits and seeing how the model decides what it is
                                    drawing.
                                </p>
                            </div>
                            <dl className="grid flex-[999_1_520px] grid-cols-1 gap-x-10 md:grid-cols-2">
                                {TOOLS.map(({ icon: Icon, name, text }) => (
                                    <div key={name} className="flex gap-3.5 border-t border-slate-700 py-5">
                                        <Icon
                                            className="mt-0.5 h-5 w-5 shrink-0 text-violet-400"
                                            strokeWidth={1.75}
                                            aria-hidden="true"
                                        />
                                        <div>
                                            <dt className="font-semibold">{name}</dt>
                                            <dd className="mt-1 text-[15px] text-slate-300">{text}</dd>
                                        </div>
                                    </div>
                                ))}
                            </dl>
                        </section>

                        <footer className="flex flex-wrap justify-between gap-x-8 gap-y-3 border-t border-slate-700 py-7 text-sm text-slate-400">
                            <p className="max-w-[70ch]">
                                A UML501 machine learning course project. Built with PyTorch, FastAPI, Next.js
                                and ONNX Runtime Web.
                            </p>
                            <a
                                href={GITHUB_URL}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-300 hover:text-white"
                            >
                                AaravKA05/diffusion-mnist
                            </a>
                        </footer>
                    </div>
                </div>
            </div>
        </section>
    );
}
