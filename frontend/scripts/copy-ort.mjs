// Copies the ONNX Runtime Web files the browser engine needs into public/ort.
// Runs automatically after `npm install` (see "postinstall" in package.json),
// on the laptop and on Vercel, so these 12 MB are never committed to git.

import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dest = join(root, "public", "ort");

const files = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

if (!existsSync(src)) {
  console.warn("[copy-ort] onnxruntime-web is not installed yet; skipping.");
  process.exit(0);
}

mkdirSync(dest, { recursive: true });
for (const file of files) {
  copyFileSync(join(src, file), join(dest, file));
}
console.log(`[copy-ort] copied ${files.length} files to public/ort`);
