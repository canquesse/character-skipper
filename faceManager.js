// ── Face recognition engine ──────────────────────────────────────────────
// Detection runs in a dedicated Web Worker (detection-worker.js) inside a
// hidden extension-origin iframe, so ALL neural-net computation happens OFF
// the main thread and outside YouTube's CSP.
//
// Embedding model: MobileFaceNet (ArcFace) preferred, faceres fallback.
// The worker scales descriptors so distances are model-agnostic app-wide.

let contentSandboxReady = false;   // checked by content.js
let engineModel = null;            // "arcface" | "faceres" — set on INIT_OK, read by content.js
let sandboxInitAttempts = 0;
const MAX_SANDBOX_INIT_ATTEMPTS = 3;

// Reuse a single canvas for frame capture (main thread)
const _captureCanvas = document.createElement("canvas");
// Separate canvas for decoding JPEG data URLs → ImageData
const _decodeCanvas  = document.createElement("canvas");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── L2 normalize (main thread copy — used for descriptor from image upload) ──
function l2Normalize(arr) {
  let norm = 0;
  for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return arr;
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] / norm;
  return out;
}

// ── Engine state ─────────────────────────────────────────────────────────
// The engine lives in a hidden EXTENSION-ORIGIN iframe (sandbox.html).
// Why an iframe? A content script runs in the PAGE origin, so it can NOT
// construct a Worker from a chrome-extension:// URL ("cannot be accessed
// from origin 'https://www.youtube.com'"), and blob: workers are blocked by
// YouTube's strict CSP. The iframe IS extension-origin, so inside it the
// detection worker (human.js + models) loads freely under the extension's
// own CSP. Pixels travel content script → iframe → worker as Transferables
// (zero-copy the whole way).
const _ENGINE_ORIGIN = "chrome-extension://" + chrome.runtime.id;

let _frame               = null;
let _frameListener       = null;
let _workerReady         = false;
let _workerInitPromise   = null;
let _workerInitResolve   = null;
let _workerInitReject    = null;
let _nextDetectionId     = 0;
const _pendingDetections = new Map(); // id → { resolve, reject }

// ── Start the detection engine (hidden iframe + worker inside it) ────────
function startWorker() {
  if (_workerInitPromise) return _workerInitPromise;

  _workerInitPromise = new Promise((resolve, reject) => {
    _workerInitResolve = resolve;
    _workerInitReject  = reject;
  });

  // 90-second timeout (model files ~7 MB, first load can be slow)
  const _initTimeout = setTimeout(() => {
    _workerInitReject?.(new Error("Engine INIT timeout after 90 s"));
    _workerInitResolve = null;
    _workerInitReject  = null;
  }, 90000);

  try {
    // Listen for replies from the iframe (INIT_OK / INIT_ERROR / DETECT_RESULT)
    _frameListener = function (event) {
      if (event.origin !== _ENGINE_ORIGIN) return;
      if (!_frame || event.source !== _frame.contentWindow) return;

      const msg = event.data;
      if (!msg || msg.__ycs !== true) return;

      if (msg.type === "INIT_OK") {
        clearTimeout(_initTimeout);
        _workerReady = true;
        contentSandboxReady = true;
        engineModel = msg.model || null;
        console.log("[FACE MANAGER] ✓ detection engine ready — model:",
          msg.model || "unknown", "· backend:", msg.backend || "unknown");
        _workerInitResolve?.();
        _workerInitResolve = null;
        _workerInitReject  = null;

      } else if (msg.type === "INIT_ERROR") {
        clearTimeout(_initTimeout);
        console.error("[FACE MANAGER] engine init error:", msg.error);
        _workerInitReject?.(new Error(msg.error));
        _workerInitResolve = null;
        _workerInitReject  = null;

      } else if (msg.type === "DETECT_RESULT") {
        const pending = _pendingDetections.get(msg.id);
        if (!pending) return;
        _pendingDetections.delete(msg.id);
        if (msg.error) {
          pending.reject(new Error(msg.error));
        } else {
          pending.resolve(msg.result);
        }
      }
    };
    window.addEventListener("message", _frameListener);

    // Create the hidden extension-origin iframe
    _frame = document.createElement("iframe");
    _frame.src = chrome.runtime.getURL("sandbox.html");
    _frame.setAttribute("aria-hidden", "true");
    _frame.style.cssText =
      "display:none!important;width:0;height:0;border:0;position:absolute;left:-9999px;";

    _frame.addEventListener("load", () => {
      if (!_frame || !_frame.contentWindow) return;
      console.log("[FACE MANAGER] engine frame loaded, sending INIT");
      _frame.contentWindow.postMessage({ __ycs: true, type: "INIT" }, _ENGINE_ORIGIN);
    }, { once: true });

    (document.body || document.documentElement).appendChild(_frame);

  } catch (err) {
    clearTimeout(_initTimeout);
    console.error("[FACE MANAGER] failed to start engine:", err.message);
    _workerInitReject?.(err);
    _workerInitResolve = null;
    _workerInitReject  = null;
  }

  return _workerInitPromise;
}

// ── Send one detection request to the engine ─────────────────────────────
function detectViaWorker(imageData) {
  return new Promise((resolve, reject) => {
    if (!_workerReady || !_frame || !_frame.contentWindow) {
      reject(new Error("Detection engine not ready"));
      return;
    }

    const id = _nextDetectionId++;
    _pendingDetections.set(id, { resolve, reject });

    // Transfer the underlying ArrayBuffer — zero-copy, very fast
    const buffer = imageData.data.buffer.slice(0); // slice to avoid transferring shared buffers
    _frame.contentWindow.postMessage(
      { __ycs: true, type: "DETECT", id, pixels: buffer, width: imageData.width, height: imageData.height },
      _ENGINE_ORIGIN,
      [buffer]
    );
  });
}

// ── Decode a JPEG/PNG data URL → ImageData (main thread) ─────────────────
async function dataUrlToImageData(dataUrl, maxWidth) {
  const img = await createImageFromDataUrl(dataUrl);
  const scale = maxWidth ? Math.min(1, maxWidth / img.naturalWidth) : 1;
  const w = Math.max(1, Math.round(img.naturalWidth  * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));

  _decodeCanvas.width  = w;
  _decodeCanvas.height = h;
  const ctx = _decodeCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// ── Public init / reset / retry ─────────────────────────────────────────
async function initContentSandbox() {
  if (contentSandboxReady) return true;

  console.log("[FACE MANAGER] initializing detection worker…");

  await startWorker();

  console.log("[FACE MANAGER] ✓ face engine ready (worker)");
  return true;
}

async function initContentSandboxWithRetry() {
  for (let attempt = 1; attempt <= MAX_SANDBOX_INIT_ATTEMPTS; attempt++) {
    try {
      console.log("[FACE MANAGER] engine init attempt", attempt);
      await initContentSandbox();
      sandboxInitAttempts = 0;
      return true;
    } catch (err) {
      console.error("[FACE MANAGER] engine init attempt", attempt, "failed:", err.message);
      sandboxInitAttempts = attempt;

      if (attempt < MAX_SANDBOX_INIT_ATTEMPTS) {
        resetSandbox();
        await sleep(1000 * attempt);
      } else {
        throw err;
      }
    }
  }
}

function resetSandbox() {
  contentSandboxReady = false;
  _workerReady        = false;

  if (_frameListener) {
    window.removeEventListener("message", _frameListener);
    _frameListener = null;
  }
  if (_frame) {
    try { _frame.remove(); } catch (_) {}   // tears down iframe + its worker
    _frame = null;
  }
  // Reject all pending detections
  _pendingDetections.forEach(p => p.reject(new Error("Sandbox reset")));
  _pendingDetections.clear();

  _workerInitPromise = null;
  _workerInitResolve = null;
  _workerInitReject  = null;
}

// No second sandbox — kept for backward compat with content.js call sites
function resetSandbox2() { /* no-op */ }

// ── Video frame capture (main thread — used for thumbnails too) ──────────
function captureVideoFrameDataUrl(video, maxWidth = 640, quality = 0.85) {
  const width  = video.videoWidth  || video.clientWidth;
  const height = video.videoHeight || video.clientHeight;

  if (!width || !height) throw new Error("Video frame size unavailable");

  const scale       = Math.min(1, maxWidth / width);
  const targetWidth  = Math.max(1, Math.round(width  * scale));
  const targetHeight = Math.max(1, Math.round(height * scale));

  _captureCanvas.width  = targetWidth;
  _captureCanvas.height = targetHeight;

  const ctx = _captureCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas context unavailable");

  try {
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
    const dataUrl = _captureCanvas.toDataURL("image/jpeg", quality);
    return { dataUrl, frameWidth: targetWidth, frameHeight: targetHeight };
  } catch (err) {
    throw new Error("Failed to capture video frame: " + err.message);
  }
}

// ── Create an Image element from a data URL ──────────────────────────────
function createImageFromDataUrl(dataUrl) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:image/")) {
    return Promise.reject(new Error("Invalid image data URL"));
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    const tid = setTimeout(() => reject(new Error("Image load timeout")), 10000);
    img.onload  = () => { clearTimeout(tid); resolve(img); };
    img.onerror = () => { clearTimeout(tid); reject(new Error("Image load failed")); };
    img.src = dataUrl;
  });
}

// ── Detect all faces from a live video element ───────────────────────────
async function detectFacesFromVideo(video) {
  await initContentSandbox();

  const width  = video.videoWidth  || video.clientWidth;
  const height = video.videoHeight || video.clientHeight;
  if (!width || !height) throw new Error("Video frame size unavailable");

  const scale = Math.min(1, 640 / width);
  const tw    = Math.max(1, Math.round(width  * scale));
  const th    = Math.max(1, Math.round(height * scale));

  _captureCanvas.width  = tw;
  _captureCanvas.height = th;
  const ctx = _captureCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas context unavailable");
  ctx.drawImage(video, 0, 0, tw, th);

  // Get raw pixels — transfer to worker (zero-copy via Transferable)
  const imageData = ctx.getImageData(0, 0, tw, th);
  return detectViaWorker(imageData);
}

// ── Capture a video frame as raw ImageData (NO JPEG round-trip) ──────────
// toDataURL(jpeg) + Image-decode costs ~60-150 ms per frame; getImageData is
// ~10-20 ms. Used by the full-video scan for maximum throughput.
function captureVideoFrameImageData(video, maxWidth = 640) {
  const width  = video.videoWidth  || video.clientWidth;
  const height = video.videoHeight || video.clientHeight;
  if (!width || !height) throw new Error("Video frame size unavailable");

  const scale = Math.min(1, maxWidth / width);
  const tw    = Math.max(1, Math.round(width  * scale));
  const th    = Math.max(1, Math.round(height * scale));

  _captureCanvas.width  = tw;
  _captureCanvas.height = th;
  const ctx = _captureCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas context unavailable");
  ctx.drawImage(video, 0, 0, tw, th);

  return { imageData: ctx.getImageData(0, 0, tw, th), frameWidth: tw, frameHeight: th };
}

// ── Detect faces from pre-captured ImageData ─────────────────────────────
async function detectFacesFromImageData(imageData) {
  await initContentSandbox();
  return detectViaWorker(imageData);
}

// ── Crop a square face thumbnail straight from ImageData (synchronous) ───
// No JPEG decode needed — pixels are already in RAM.
const _thumbSrcCanvas = document.createElement("canvas");
const _thumbDstCanvas = document.createElement("canvas");
function cropFaceFromImageData(imageData, box) {
  try {
    const fw = imageData.width, fh = imageData.height;
    _thumbSrcCanvas.width  = fw;
    _thumbSrcCanvas.height = fh;
    const sctx = _thumbSrcCanvas.getContext("2d");
    sctx.putImageData(imageData, 0, 0);

    const padding = Math.max(box.width, box.height) * 0.25;
    const sx = Math.max(0, box.x - padding);
    const sy = Math.max(0, box.y - padding);
    const sw = Math.min(fw - sx, box.width  + padding * 2);
    const sh = Math.min(fh - sy, box.height + padding * 2);
    // Cap at 160 px — plenty for gallery thumbnails, keeps memory small
    const size = Math.min(160, Math.ceil(Math.max(sw, sh, 48)));

    _thumbDstCanvas.width  = size;
    _thumbDstCanvas.height = size;
    const dctx = _thumbDstCanvas.getContext("2d");
    dctx.drawImage(_thumbSrcCanvas, sx, sy, sw, sh, 0, 0, size, size);
    return _thumbDstCanvas.toDataURL("image/jpeg", 0.8);
  } catch (_) {
    return null;
  }
}

// ── Detect faces from an already-captured data URL ───────────────────────
async function detectFacesFromDataUrl(dataUrl, frameWidth, frameHeight) {
  await initContentSandbox();

  const imageData = await dataUrlToImageData(dataUrl, 640);
  const result    = await detectViaWorker(imageData);

  if (!result.frameWidth)  result.frameWidth  = frameWidth  || imageData.width;
  if (!result.frameHeight) result.frameHeight = frameHeight || imageData.height;

  return result;
}

// ── Extract a single descriptor from an uploaded image ───────────────────
async function extractSingleDescriptorFromImageDataUrl(dataUrl) {
  await initContentSandbox();

  const imageData = await dataUrlToImageData(dataUrl, 640);
  const result    = await detectViaWorker(imageData);

  if (!result.faces?.length) return null;

  // Return descriptor of the highest-confidence face
  const best = result.faces.reduce((a, b) => (b.score > a.score ? b : a));
  return best.descriptor || null;
}

// ── Matching utilities ────────────────────────────────────────────────────
function euclideanDistance(a, b) {
  if (!a || !b || a.length !== b.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function getBestMatchForDescriptor(descriptor, characters, globalThreshold = 0.52) {
  let best = null;
  const K = 3;

  for (const character of characters) {
    if (!character.enabled || !Array.isArray(character.descriptors)) continue;

    const effectiveThreshold =
      typeof character.threshold === "number" ? character.threshold : globalThreshold;

    if (!character.descriptors.length) continue;

    // Descriptors saved with a DIFFERENT embedding model have a different
    // dimension and can never match — warn (once) instead of failing silently.
    if (
      character.descriptors[0]?.length &&
      descriptor?.length &&
      character.descriptors[0].length !== descriptor.length
    ) {
      if (!getBestMatchForDescriptor._dimWarned) {
        getBestMatchForDescriptor._dimWarned = true;
        console.warn(
          "[FACE MANAGER] character \"" + character.name + "\" was saved with an older " +
          "embedding model (" + character.descriptors[0].length + "-dim vs current " +
          descriptor.length + "-dim). Re-scan the video and re-save the character."
        );
      }
      continue;
    }

    const dists = character.descriptors.map((saved) => euclideanDistance(descriptor, saved));
    dists.sort((a, b) => a - b);

    const kActual = Math.min(K, dists.length);
    const kAvg = dists.slice(0, kActual).reduce((s, d) => s + d, 0) / kActual;

    // Match on EITHER criterion:
    //  • single very close descriptor (strong direct evidence), OR
    //  • k-nearest average within a barely relaxed threshold (consensus).
    // NOTE: the old ×1.10 relax stacked on top of the adaptive per-character
    // threshold pushed the effective cut past the different-person floor in
    // ArcFace's compressed distance space — that made the skipper fire on
    // EVERY face. The adaptive threshold already includes its margin; keep
    // any extra relaxation minimal.
    const isMatch =
      dists[0] <= effectiveThreshold ||
      kAvg     <= effectiveThreshold * 1.05;

    if (!best || kAvg < best.distance) {
      best = {
        characterId:   character.id,
        characterName: character.name,
        distance:      kAvg,
        threshold:     effectiveThreshold,
        matched:       isMatch,
      };
    }

    if (dists[0] < 0.18) break;
  }

  return best;
}

function analyzeFaceMatches(detectionResult, characters, threshold = 0.52) {
  const safeResult = detectionResult || {};
  const faces = Array.isArray(safeResult.faces) ? safeResult.faces : [];

  let bestOverallMatch = null;
  let matchedFace      = null;

  const analyzedFaces = faces.map((face) => {
    const bestMatch = getBestMatchForDescriptor(face.descriptor, characters, threshold);

    const analyzedFace = {
      descriptor: face.descriptor,
      box:        face.box  || null,
      score:      typeof face.score === "number" ? face.score : null,
      bestMatch: bestMatch
        ? {
            characterId:   bestMatch.characterId,
            characterName: bestMatch.characterName,
            distance:      bestMatch.distance,
            threshold:     bestMatch.threshold,
            matched:       bestMatch.matched,
          }
        : null,
    };

    if (bestMatch && bestMatch.matched) {
      if (!bestOverallMatch || bestMatch.distance < bestOverallMatch.distance) {
        bestOverallMatch = bestMatch;
        matchedFace      = analyzedFace;
      }
    }

    return analyzedFace;
  });

  return {
    frameWidth:  safeResult.frameWidth  || 0,
    frameHeight: safeResult.frameHeight || 0,
    faceCount:
      typeof safeResult.faceCount === "number" ? safeResult.faceCount : analyzedFaces.length,
    detectionTimeMs:
      typeof safeResult.detectionTimeMs === "number" ? safeResult.detectionTimeMs : 0,
    faces:       analyzedFaces,
    visible:     !!bestOverallMatch,
    match: bestOverallMatch
      ? {
          characterId:   bestOverallMatch.characterId,
          characterName: bestOverallMatch.characterName,
          distance:      bestOverallMatch.distance,
        }
      : null,
    matchedFace,
  };
}
