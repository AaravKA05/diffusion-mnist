# DiffusionLab frontend

Next.js app for [DiffusionLab](../README.md). It talks to the FastAPI backend (`server2.py`) when that is running, and otherwise runs the model in the browser with ONNX Runtime Web.

```bash
npm install      # also copies ONNX Runtime Web files to public/ort (scripts/copy-ort.mjs)
npm run dev      # http://localhost:3000
npm run build    # production build
npm run lint
```

| Path | What it holds |
|---|---|
| `src/app/page.tsx` | Main page: sidebar, Studio and most tabs |
| `src/app/RecognitionTimeline.tsx`, `SketchToDigit.tsx`, `Diffusion2D.tsx` | Recognition timeline, Sketch to digit and 2D playground tabs |
| `src/app/LaunchScreen.tsx`, `LiveDenoise.tsx` | Launch page and its live denoising demo |
| `src/app/globals.css` | Theme colours and the glass tile styles |
| `src/lib/engine/` | Browser engine: ONNX Runtime Web, DDIM sampler, local API |
| `public/models/` | `unet.onnx`, `classifier.onnx`, `schedule.json` (made by `export_onnx.py`) and `toy2d/` (made by `toy2d.py`) |
| `public/media/` | Background video on the launch page |

The backend URL and setup are described in the main README.
