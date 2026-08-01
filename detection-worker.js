"use strict";
// ── detection-worker.js ───────────────────────────────────────────────────
// Face-detection Web Worker. Loaded as an EXTENSION-ORIGIN worker (from the
// hidden sandbox.html iframe) — see faceManager.js for why.
//
// Recognition pipeline (primary — "arcface"):
//   1. BlazeFace detector + FaceMesh landmarks   (human.js, GPU)
//   2. 5-point similarity alignment to the 112×112 ArcFace template
//   3. InsightFace w600k MobileFaceNet (ONNX)     (onnxruntime-web, WASM-SIMD)
// This is the same pipeline commercial face apps use. Proper landmark
// alignment is what makes ArcFace embeddings robust to pose/sunglasses —
// human.js's built-in models skip it, which is why they fragmented badly.
//
// Fallback — "faceres": human.js's combined model, used only if the ONNX
// stack fails to load. Same message protocol either way.

// ── DOM polyfills — human.js's IIFE touches document/window on evaluation ──
// NOTE: only polyfill what a worker does NOT already have. Built-ins like
// `navigator` and `globalThis` exist natively on WorkerGlobalScope as
// getter-only accessors — assigning to them throws in strict mode.
self.window = self;
self.document = {
  createElement(tag) {
    if (tag === "canvas") return new OffscreenCanvas(1, 1);
    return { style: {}, setAttribute() {}, appendChild() {}, getContext() { return null; } };
  },
  createElementNS(_ns, tag) { return this.createElement(tag); },
  body:            { appendChild() {}, style: {} },
  head:            { appendChild() {} },
  documentElement: { style: {}, setAttribute() {} },
  getElementById()   { return null; },
  querySelector()    { return null; },
  querySelectorAll() { return []; },
};
self.HTMLCanvasElement = OffscreenCanvas;
self.HTMLImageElement  = class HTMLImageElement {};
self.HTMLVideoElement  = class HTMLVideoElement {};
self.Image             = class Image {};
self.screen            = { width: 1920, height: 1080 };
// navigator: already present natively in workers (WorkerNavigator) — no polyfill

// ── State ─────────────────────────────────────────────────────────────────
let _human           = null;
let _arcSession      = null;   // onnxruntime InferenceSession (w600k_mbf)
let _activeModel     = null;   // "arcface" | "faceres"
let _descriptorScale = 1.0;
let _ready           = false;
let _humanLoaded     = false;  // importScripts(human.js) done once
let _ortLoaded       = false;  // importScripts(ort.min.js) done once

// ── Resolve Human constructor from the namespace object ───────────────────
function getHumanCtor() {
  if (typeof Human === "undefined") return null;
  if (typeof Human === "function")  return Human;
  return Human.Human || Human.default || null;
}

// ── L2 normalize ──────────────────────────────────────────────────────────
function l2Normalize(arr) {
  let norm = 0;
  for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return arr;
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] / norm;
  return out;
}

// ── Model calibration ─────────────────────────────────────────────────────
// DESCRIPTOR_SCALE maps each model's natural distance distribution onto the
// app-wide threshold scale (cluster ~0.45-0.58, match ~0.52). w600k ArcFace:
// same-person video pairs ~0.6-1.0 apart raw, different people ≥ ~1.25 —
// scale 0.50 lands same ≤ ~0.50 and different ≥ ~0.63.
// NOTE: baked into every SAVED descriptor — never change without migration.
const MODEL_SCALES = { arcface: 0.50, faceres: 1.0 };

// ── ArcFace 112×112 alignment template (InsightFace standard) ─────────────
// Order: viewer-left eye, viewer-right eye, nose tip, viewer-left mouth
// corner, viewer-right mouth corner.
const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// ── 5 landmarks from the FaceMesh point cloud ─────────────────────────────
// Eye centers are averaged from 4 contour points each; eyes and mouth
// corners are ordered by X at runtime so MediaPipe's left/right naming
// convention can never flip the mapping.
function fiveLandmarksFromMesh(mesh) {
  if (!Array.isArray(mesh) || mesh.length < 400) return null;

  const avg = (idxs) => {
    let x = 0, y = 0;
    for (const i of idxs) { x += mesh[i][0]; y += mesh[i][1]; }
    return [x / idxs.length, y / idxs.length];
  };

  const eyeA   = avg([33, 133, 159, 145]);   // one eye (outer, inner, top, bottom)
  const eyeB   = avg([263, 362, 386, 374]);  // other eye
  const nose   = [mesh[1][0], mesh[1][1]];   // nose tip
  const mouthA = [mesh[61][0], mesh[61][1]]; // one mouth corner
  const mouthB = [mesh[291][0], mesh[291][1]];

  const [eyeL, eyeR]     = eyeA[0] <= eyeB[0]   ? [eyeA, eyeB]     : [eyeB, eyeA];
  const [mouthL, mouthR] = mouthA[0] <= mouthB[0] ? [mouthA, mouthB] : [mouthB, mouthA];

  return [eyeL, eyeR, nose, mouthL, mouthR];
}

// ── Least-squares similarity transform (src → dst, no reflection) ─────────
// Solves qx = a·x − b·y + tx ; qy = b·x + a·y + ty  over the 5 point pairs.
function similarityTransform(src, dst) {
  const n = src.length;
  let mx = 0, my = 0, mqx = 0, mqy = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0]; my += src[i][1];
    mqx += dst[i][0]; mqy += dst[i][1];
  }
  mx /= n; my /= n; mqx /= n; mqy /= n;

  let num_a = 0, num_b = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - mx,  y = src[i][1] - my;
    const qx = dst[i][0] - mqx, qy = dst[i][1] - mqy;
    num_a += x * qx + y * qy;
    num_b += x * qy - y * qx;
    den   += x * x + y * y;
  }
  if (den === 0) return null;

  const a = num_a / den;
  const b = num_b / den;
  const tx = mqx - a * mx + b * my;
  const ty = mqy - b * mx - a * my;
  return { a, b, tx, ty };
}

// ── Warp a face to the aligned 112×112 crop ───────────────────────────────
const _alignCanvas = new OffscreenCanvas(112, 112);
function alignFace(srcCanvas, landmarks5) {
  const M = similarityTransform(landmarks5, ARCFACE_DST);
  if (!M) return null;

  const ctx = _alignCanvas.getContext("2d", { willReadFrequently: true });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, 112, 112);
  // Canvas matrix maps source→dest: x' = a·x + c·y + e, y' = b·x + d·y + f
  ctx.setTransform(M.a, M.b, -M.b, M.a, M.tx, M.ty);
  ctx.drawImage(srcCanvas, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return ctx.getImageData(0, 0, 112, 112);
}

// ── ArcFace preprocessing + inference ─────────────────────────────────────
// InsightFace standard: RGB, (v − 127.5) / 127.5, NCHW float32 [1,3,112,112].
async function arcfaceEmbed(alignedImageData) {
  const px = alignedImageData.data;
  const HW = 112 * 112;
  const input = new Float32Array(3 * HW);
  for (let i = 0; i < HW; i++) {
    input[i]          = (px[i * 4]     - 127.5) / 127.5; // R
    input[HW + i]     = (px[i * 4 + 1] - 127.5) / 127.5; // G
    input[2 * HW + i] = (px[i * 4 + 2] - 127.5) / 127.5; // B
  }

  const feeds = {};
  feeds[_arcSession.inputNames[0]] =
    new ort.Tensor("float32", input, [1, 3, 112, 112]);
  const results = await _arcSession.run(feeds);
  const out = results[_arcSession.outputNames[0]].data;
  return Array.from(out);
}

// ── Human configs per variant ─────────────────────────────────────────────
function buildHuman(HumanClass, base, backend, variant) {
  return new HumanClass({
    backend,
    modelBasePath: base,
    debug: false,
    async: true,
    // For faceres the embedded-sample warmup compiles the whole pipeline.
    // For arcface we warm up manually after the ONNX session is ready.
    warmup: variant === "faceres" ? "face" : "none",
    cacheSensitivity: 0,
    face: {
      enabled: true,
      detector: {
        enabled: true,
        modelPath: "blazeface.json",
        // arcface: alignment handles rotation properly — skip the extra cost.
        // faceres: no alignment stage, so let human rotate crops itself.
        rotation: variant === "faceres",
        maxDetected: 20,
        skipFrames: 0,
        minConfidence: 0.20,
      },
      // FaceMesh provides the 5 landmarks the ArcFace alignment needs.
      mesh:    { enabled: variant === "arcface", modelPath: "facemesh.json" },
      iris:    { enabled: false },
      emotion: { enabled: false },
      description: {
        enabled: variant === "faceres",
        modelPath: "faceres.json",
        minConfidence: 0.10,
        skipFrames: 0,
      },
      antispoof: { enabled: false },
      liveness:  { enabled: false },
    },
    body:         { enabled: false },
    hand:         { enabled: false },
    object:       { enabled: false },
    gesture:      { enabled: false },
    segmentation: { enabled: false },
  });
}

// ── Backend / model init ──────────────────────────────────────────────────
function offscreenWebGLAvailable() {
  try {
    return !!new OffscreenCanvas(8, 8).getContext("webgl2");
  } catch (_) { return false; }
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(label + " timeout after " + ms + " ms")), ms)
    ),
  ]);
}

const GPU_INIT_TIMEOUT_MS = 20000;
const ORT_INIT_TIMEOUT_MS = 45000; // 13 MB model + WASM compile on first run

async function initOrtSession(msg) {
  if (!_ortLoaded) {
    importScripts(msg.ortUrl);           // defines self.ort (UMD)
    _ortLoaded = true;
  }
  ort.env.wasm.wasmPaths  = msg.ortBaseUrl;
  ort.env.wasm.numThreads = 1;           // extension pages aren't crossOriginIsolated
  ort.env.logLevel        = "error";

  _arcSession = await withTimeout(
    ort.InferenceSession.create(msg.modelBaseUrl + "w600k_mbf.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }),
    ORT_INIT_TIMEOUT_MS,
    "onnx session"
  );
  console.log("[WORKER] w600k ArcFace session ready — inputs:",
    _arcSession.inputNames.join(","), "outputs:", _arcSession.outputNames.join(","));

  // Warm up WASM kernels once so the first real face isn't slow
  const warm = new ImageData(112, 112);
  await arcfaceEmbed(warm);
}

async function initHumanWithBestBackend(HumanClass, msg) {
  const base = msg.modelBaseUrl.endsWith("/") ? msg.modelBaseUrl : msg.modelBaseUrl + "/";

  const backends = [];
  if (self.navigator?.gpu)      backends.push("webgpu");
  if (offscreenWebGLAvailable()) backends.push("humangl");
  backends.push("cpu"); // guaranteed to work — last resort

  // arcface primary; faceres only if the ONNX stack can't come up.
  const variants = ["arcface", "faceres"];

  let lastErr = null;

  for (const variant of variants) {
    // The ONNX session is backend-independent (WASM) — bring it up once.
    if (variant === "arcface") {
      try {
        if (!_arcSession) await initOrtSession(msg);
      } catch (err) {
        lastErr = err;
        console.warn("[WORKER] ONNX stack failed — falling back to faceres:", err.message);
        continue;
      }
    }

    for (const backend of backends) {
      let h = null;
      try {
        console.log("[WORKER] trying:", variant, "on", backend);
        h = buildHuman(HumanClass, base, backend, variant);

        const doLoad = (async () => {
          await h.load();
          await h.warmup(); // no-op for warmup:"none"
          if (variant === "arcface") {
            // Compile detector+mesh GPU shaders with a realistic frame
            await h.detect(new OffscreenCanvas(640, 360));
          }
        })();

        if (backend === "cpu") {
          await doLoad;
        } else {
          await withTimeout(doLoad, GPU_INIT_TIMEOUT_MS, variant + "/" + backend);
        }

        // Human can silently fall back to cpu — verify we really got the GPU
        const active = h.tf?.getBackend?.() || backend;
        if (backend !== "cpu" && active === "cpu") {
          throw new Error("silently fell back to cpu");
        }

        _descriptorScale = MODEL_SCALES[variant];
        console.log("[WORKER] ✓ ready:", variant, "on", active,
          "(descriptor scale " + _descriptorScale + ")");
        return { human: h, backend: active, model: variant };

      } catch (err) {
        lastErr = err;
        console.warn("[WORKER]", variant, "on", backend, "failed:", err.message);
        try { h?.tf?.disposeVariables?.(); } catch (_) {}
        try { h?.tf?.engine?.()?.reset?.();  } catch (_) {}
      }
    }
  }

  throw lastErr || new Error("all model/backend combinations failed");
}

// ── Per-face descriptor extraction ────────────────────────────────────────
function scaleDescriptor(unit) {
  if (_descriptorScale === 1.0) return unit;
  for (let i = 0; i < unit.length; i++) unit[i] *= _descriptorScale;
  return unit;
}

async function facesToDescriptors(raw, srcCanvas) {
  const out = [];

  for (const f of (raw.face || [])) {
    if (!f.box) continue;

    let embedding = null;

    if (_activeModel === "arcface") {
      const lm5 = fiveLandmarksFromMesh(f.mesh);
      if (!lm5) continue; // no landmarks → alignment impossible → skip face
      const aligned = alignFace(srcCanvas, lm5);
      if (!aligned) continue;
      embedding = await arcfaceEmbed(aligned);
    } else {
      embedding = f.embedding?.length ? Array.from(f.embedding) : null;
    }

    if (!embedding || !embedding.length) continue;

    out.push({
      descriptor: scaleDescriptor(l2Normalize(embedding)),
      score:      f.score ?? null,
      box: {
        x:      f.box[0],
        y:      f.box[1],
        width:  f.box[2],
        height: f.box[3],
      },
    });
  }

  return out;
}

// ── Message handler ───────────────────────────────────────────────────────
self.onmessage = async function (e) {
  const msg = e.data;

  // ── INIT ────────────────────────────────────────────────────────────────
  if (msg.type === "INIT") {
    try {
      // Load human.js from the extension origin (once). importScripts is
      // synchronous, so Human is guaranteed available on the next line.
      if (!_humanLoaded) {
        importScripts(msg.humanUrl);
        _humanLoaded = true;
      }

      const HumanClass = getHumanCtor();
      if (!HumanClass) {
        throw new Error(
          "Human constructor not found. typeof Human=" + (typeof Human) +
          (typeof Human === "object" ? " keys=" + Object.keys(Human).join(",") : "")
        );
      }

      const { human, backend, model } = await initHumanWithBestBackend(HumanClass, msg);
      _human       = human;
      _activeModel = model;

      _ready = true;
      self.postMessage({ type: "INIT_OK", backend, model });
    } catch (err) {
      console.error("[WORKER] INIT failed:", err.message, err.stack);
      self.postMessage({ type: "INIT_ERROR", error: err.message });
    }
    return;
  }

  // ── DETECT ──────────────────────────────────────────────────────────────
  if (msg.type === "DETECT") {
    const { id, pixels, width, height } = msg;

    if (!_ready || !_human) {
      self.postMessage({ type: "DETECT_RESULT", id, error: "Worker not ready" });
      return;
    }

    try {
      const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
      const canvas    = new OffscreenCanvas(width, height);
      const ctx       = canvas.getContext("2d");
      ctx.putImageData(imageData, 0, 0);

      const t0    = performance.now();
      const raw   = await _human.detect(canvas);
      const faces = await facesToDescriptors(raw, canvas);
      const ms    = Math.round(performance.now() - t0);

      self.postMessage({
        type: "DETECT_RESULT",
        id,
        result: {
          frameWidth:      width,
          frameHeight:     height,
          faceCount:       faces.length,
          detectionTimeMs: ms,
          faces,
        },
      });
    } catch (err) {
      console.error("[WORKER] DETECT failed:", err.message);
      self.postMessage({ type: "DETECT_RESULT", id, error: err.message });
    }
    return;
  }
};
