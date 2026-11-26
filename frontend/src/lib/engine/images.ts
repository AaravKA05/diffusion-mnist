/*
 * Image helpers: turn model tensors into PNG base64 strings (the same format
 * the FastAPI server returns) and read the sketch canvas back into pixels.
 */

function makeCanvas(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is not available in this browser.");
  return { canvas, ctx };
}

function toBase64(canvas: HTMLCanvasElement) {
  return canvas.toDataURL("image/png").split(",", 2)[1];
}

/** Grey image (values in [-1, 1], width x height) -> PNG base64, smoothly resized. */
export function greyToPng(data: ArrayLike<number>, width: number, height: number, size?: number) {
  const src = makeCanvas(width, height);
  const img = src.ctx.createImageData(width, height);
  for (let j = 0; j < width * height; j++) {
    const v = Math.round(((data[j] + 1) / 2) * 255);
    const g = v < 0 ? 0 : v > 255 ? 255 : v;
    img.data[4 * j] = g;
    img.data[4 * j + 1] = g;
    img.data[4 * j + 2] = g;
    img.data[4 * j + 3] = 255;
  }
  src.ctx.putImageData(img, 0, 0);

  if (!size || size === width) return toBase64(src.canvas);

  const out = makeCanvas(size, size);
  out.ctx.imageSmoothingEnabled = true;
  out.ctx.imageSmoothingQuality = "high";
  out.ctx.drawImage(src.canvas, 0, 0, size, size);
  return toBase64(out.canvas);
}

/** Bilinear resize, same as F.interpolate(mode="bilinear", align_corners=False). */
export function resizeBilinear(src: ArrayLike<number>, h: number, w: number, outH: number, outW: number) {
  const out = new Float32Array(outH * outW);
  const sy = h / outH;
  const sx = w / outW;
  for (let oy = 0; oy < outH; oy++) {
    let y = (oy + 0.5) * sy - 0.5;
    if (y < 0) y = 0;
    const y0 = Math.min(Math.floor(y), h - 1);
    const y1 = Math.min(y0 + 1, h - 1);
    const fy = y - y0;
    for (let ox = 0; ox < outW; ox++) {
      let x = (ox + 0.5) * sx - 0.5;
      if (x < 0) x = 0;
      const x0 = Math.min(Math.floor(x), w - 1);
      const x1 = Math.min(x0 + 1, w - 1);
      const fx = x - x0;
      const top = src[y0 * w + x0] * (1 - fx) + src[y0 * w + x1] * fx;
      const bottom = src[y1 * w + x0] * (1 - fx) + src[y1 * w + x1] * fx;
      out[oy * outW + ox] = top * (1 - fy) + bottom * fy;
    }
  }
  return out;
}

function loadImage(dataUrl: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read the sketch image."));
    img.src = dataUrl;
  });
}

/**
 * Port of lab_features.decode_sketch: crop the strokes, fit them into a 20x20
 * box, centre by centre of mass in 28x28 (like MNIST), resize to 32x32.
 * Returns 32*32 values in [-1, 1]. Throws if the canvas is empty.
 */
export async function decodeSketch(image: string) {
  const dataUrl = image.startsWith("data:") ? image : `data:image/png;base64,${image}`;
  const img = await loadImage(dataUrl);

  const full = makeCanvas(img.naturalWidth, img.naturalHeight);
  full.ctx.fillStyle = "#000";
  full.ctx.fillRect(0, 0, img.naturalWidth, img.naturalHeight);
  full.ctx.drawImage(img, 0, 0);
  const { data, width, height } = full.ctx.getImageData(0, 0, img.naturalWidth, img.naturalHeight);

  const grey = new Float32Array(width * height);
  let total = 0;
  for (let j = 0; j < width * height; j++) {
    const v = (0.299 * data[4 * j] + 0.587 * data[4 * j + 1] + 0.114 * data[4 * j + 2]) / 255;
    grey[j] = v;
    total += v;
  }
  if (total / (width * height) > 0.5) {
    for (let j = 0; j < grey.length; j++) grey[j] = 1 - grey[j];
  }

  let top = height, bottom = -1, left = width, right = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (grey[y * width + x] > 0.1) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }
  if (bottom < 0) throw new Error("The sketch is empty. Draw a digit first.");

  const cropW = right - left + 1;
  const cropH = bottom - top + 1;
  const scale = 20 / Math.max(cropW, cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));

  // Grey crop onto a canvas, then let the browser downscale it smoothly
  const crop = makeCanvas(cropW, cropH);
  const cropData = crop.ctx.createImageData(cropW, cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const g = Math.round(grey[(top + y) * width + (left + x)] * 255);
      const k = 4 * (y * cropW + x);
      cropData.data[k] = g;
      cropData.data[k + 1] = g;
      cropData.data[k + 2] = g;
      cropData.data[k + 3] = 255;
    }
  }
  crop.ctx.putImageData(cropData, 0, 0);
  const small = makeCanvas(newW, newH);
  small.ctx.imageSmoothingEnabled = true;
  small.ctx.imageSmoothingQuality = "high";
  small.ctx.drawImage(crop.canvas, 0, 0, newW, newH);
  const smallData = small.ctx.getImageData(0, 0, newW, newH).data;

  const box = new Float32Array(28 * 28);
  const offY = Math.floor((28 - newH) / 2);
  const offX = Math.floor((28 - newW) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      box[(offY + y) * 28 + (offX + x)] = smallData[4 * (y * newW + x)] / 255;
    }
  }

  // Centre of mass to the middle (np.roll wraps around, so we do the same)
  let mass = 0, cy = 0, cx = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = box[y * 28 + x];
      mass += v;
      cy += y * v;
      cx += x * v;
    }
  }
  let centred = box;
  if (mass > 0) {
    const dy = Math.round(13.5 - cy / mass);
    const dx = Math.round(13.5 - cx / mass);
    centred = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const ny = (((y + dy) % 28) + 28) % 28;
        const nx = (((x + dx) % 28) + 28) % 28;
        centred[ny * 28 + nx] = box[y * 28 + x];
      }
    }
  }

  const resized = resizeBilinear(centred, 28, 28, 32, 32);
  const out = new Float32Array(32 * 32);
  for (let j = 0; j < out.length; j++) {
    const v = resized[j] < 0 ? 0 : resized[j] > 1 ? 1 : resized[j];
    out[j] = v * 2 - 1;
  }
  return out;
}
