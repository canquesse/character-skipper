let skipLoopStarted = false;
let busySkipping = false;
let lastRecordedDetectionKey = null;
let lastAutoLearnKey = null;

// ── Passive face discovery ────────────────────────────────────────────────
// Clusters unknown faces seen during normal playback.
// Stored in memory only (thumbnails are too large for chrome.storage).
const discoveredClusters = [];
const DISC_CLUSTER_THRESHOLD = 0.58; // centroid distance — model-agnostic (worker scales descriptors)
const DISC_MAX_CLUSTERS = 20;
const DISC_MAX_DESCRIPTORS = 30;  // per cluster

let lastDiscoveryVideoTime = -999;
let sandboxInitialized = false;   // true once sandbox is confirmed ready
let discoveryAttemptCount = 0;    // total discovery scans attempted
let discoveryFaceFoundCount = 0;  // total faces found across all discovery scans
let lastSandboxRetryAt = 0;       // timestamp of last sandbox retry attempt

// ── Full video scan state ─────────────────────────────────────────────────
// activeScan is set when a seek-based full-video scan is running/done.
// { id, status: "running"|"done"|"error", progress, total, clusters, error }
let activeScan = null;

// rawFaces: [{descriptor, thumbnail?, box?, ...}]
// Each face may carry a pre-cropped .thumbnail (data URL).
function updateDiscoveredClusters(rawFaces, savedCharacters) {
  if (!Array.isArray(rawFaces) || !rawFaces.length) return;

  // Build flat list of saved descriptors to filter out already-known faces
  const knownDescriptors = [];
  for (const ch of (savedCharacters || [])) {
    if (Array.isArray(ch.descriptors)) knownDescriptors.push(...ch.descriptors);
  }

  for (const face of rawFaces) {
    const desc = face.descriptor;
    if (!Array.isArray(desc) || !desc.length) continue;

    // Skip if this face matches a saved character
    let isSaved = false;
    for (const k of knownDescriptors) {
      if (euclideanDistance(desc, k) < DISC_CLUSTER_THRESHOLD) { isSaved = true; break; }
    }
    if (isSaved) continue;

    // Find best existing discovered cluster — centroid distance, not
    // any-member distance (any-member chains outliers and splits identities)
    let bestCluster = null;
    let bestDist = Infinity;
    for (const cluster of discoveredClusters) {
      if (!cluster.centroid) {
        cluster.centroid = normalizedDescriptorCentroid(cluster.descriptors);
      }
      const d = euclideanDistance(desc, cluster.centroid);
      if (d < DISC_CLUSTER_THRESHOLD && d < bestDist) {
        bestDist = d;
        bestCluster = cluster;
      }
    }

    if (bestCluster) {
      if (bestCluster.descriptors.length < DISC_MAX_DESCRIPTORS) {
        bestCluster.descriptors.push(desc);
        bestCluster.centroid = normalizedDescriptorCentroid(bestCluster.descriptors);
      }
      bestCluster.count += 1;
      bestCluster.lastSeenAt = Date.now();
      // Keep first good thumbnail
      if (!bestCluster.thumbnail && face.thumbnail) {
        bestCluster.thumbnail = face.thumbnail;
      }
    } else if (discoveredClusters.length < DISC_MAX_CLUSTERS) {
      discoveredClusters.push({
        id: "disc_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6),
        descriptors: [desc],
        centroid: desc.slice(),
        thumbnail: face.thumbnail || null,
        count: 1,
        firstSeenAt: Date.now(),
        lastSeenAt: Date.now()
      });
    }
  }

  // Sort by occurrence count, most frequent first
  discoveredClusters.sort((a, b) => b.count - a.count);
}

// Crops individual face thumbnails from a frame capture and attaches them
// to each face object. Returns a new array with .thumbnail added.
async function enrichFacesWithThumbnails(faces, rawDetection, video) {
  let frameDataUrl = null;
  try {
    const cap = captureVideoFrameDataUrl(video, 320, 0.75);
    frameDataUrl = cap.dataUrl;
  } catch (_) {}

  if (!frameDataUrl) return faces;

  return Promise.all(
    faces.map(async (face) => {
      if (!face.box) return face;
      try {
        const thumbnail = await cropFaceDataUrl(
          frameDataUrl, face.box,
          rawDetection.frameWidth, rawDetection.frameHeight
        );
        return { ...face, thumbnail };
      } catch (_) {
        return face;
      }
    })
  );
}

// Remove discovered clusters that were saved as characters
function pruneDiscoveredClusters(savedCharacters) {
  const knownDescriptors = [];
  for (const ch of (savedCharacters || [])) {
    if (Array.isArray(ch.descriptors)) knownDescriptors.push(...ch.descriptors);
  }
  for (let i = discoveredClusters.length - 1; i >= 0; i--) {
    const cluster = discoveredClusters[i];
    let matchesSaved = false;
    for (const desc of cluster.descriptors) {
      for (const k of knownDescriptors) {
        if (euclideanDistance(desc, k) < DISC_CLUSTER_THRESHOLD) { matchesSaved = true; break; }
      }
      if (matchesSaved) break;
    }
    if (matchesSaved) discoveredClusters.splice(i, 1);
  }
}

let overlayRoot = null;
let overlayTextEl = null;
let overlayHideTimer = null;

let boxLayer = null;

let currentSettings = null;
let runtimeAlive = true;
let storageWatcherInstalled = false;

// (Alt+Shift+S pause removed — use the library toggle or per-character enable/disable)

const scanState = {
  lastAnalyzedVideoTime: -999,
  lastAnalyzeWallTime: 0,
  consecutiveNoFaceScans: 0,
  consecutiveFaceScans: 0,
  consecutiveMatchScans: 0
};

const autoLearnState = {
  candidateCharacterId: null,
  candidateCount: 0,
  lastCandidateWallTime: 0,
  lastCandidateVideoTime: -999
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Profile-tracking assist (skip continuation only) ──────────────────────
// Recognition dips on hard profiles/sunglasses mid-scene, which used to end
// skips early — the character "reappeared" in profile right after a skip.
// Assist rule: while a skip is running, a face that (a) overlaps the last
// CONFIRMED match's box and (b) is a near-miss for the SAME character counts
// as still-visible. Strictly additive: it can only EXTEND an active skip,
// never start one — worst case is a slightly longer skip, bounded by the
// probe's own maxWindow. Set to false to disable entirely.
const PROFILE_TRACKING_ASSIST = true;
const TRACK_IOU_MIN     = 0.25; // box overlap with last confirmed match
const TRACK_NEAR_FACTOR = 1.25; // near-miss = distance ≤ threshold × this

function boxIoU(a, b) {
  if (!a || !b) return 0;
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width,  b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

function isContextInvalidatedError(error) {
  const message = String(error?.message || error || "");
  return (
    message.includes("Extension context invalidated") ||
    message.includes("context invalidated")
  );
}


function killRuntime(reason) {
  if (!runtimeAlive) return;

  runtimeAlive = false;
  busySkipping = false;

  console.warn("[CONTENT] runtime stopped:", reason);

  hideOverlay();
  clearFaceBoxes();
}

async function safeCall(fn, fallback = null) {
  if (!runtimeAlive) return fallback;

  try {
    return await fn();
  } catch (error) {
    if (isContextInvalidatedError(error)) {
      killRuntime(error);
      return fallback;
    }

    throw error;
  }
}

function getVideo() {
  return document.querySelector("video");
}

async function waitForVideo() {
  for (let i = 0; i < 60; i++) {
    if (!runtimeAlive) return null;

    const video = getVideo();
    if (video) return video;

    await sleep(500);
  }

  return null;
}

async function loadCurrentSettings() {
  const data = await safeCall(() => getAppData(), null);
  currentSettings = data;
  return currentSettings;
}

function ensureOverlay() {
  if (overlayRoot && document.documentElement.contains(overlayRoot)) {
    return;
  }

  overlayRoot = document.createElement("div");
  overlayRoot.id = "yt-character-skipper-overlay-root";
  overlayRoot.style.position = "fixed";
  overlayRoot.style.top = "12px";
  overlayRoot.style.right = "12px";
  overlayRoot.style.zIndex = "999999";
  overlayRoot.style.pointerEvents = "none";
  overlayRoot.style.display = "none";
  overlayRoot.style.fontFamily = "Inter, Arial, Helvetica, sans-serif";

  const box = document.createElement("div");
  box.style.background = "rgba(15, 15, 20, 0.88)";
  box.style.backdropFilter = "blur(8px)";
  box.style.color = "#ffffff";
  box.style.border = "1px solid rgba(255,255,255,0.12)";
  box.style.borderRadius = "12px";
  box.style.padding = "10px 12px";
  box.style.boxShadow = "0 10px 30px rgba(0,0,0,0.35)";
  box.style.fontSize = "13px";
  box.style.fontWeight = "700";
  box.style.maxWidth = "320px";
  box.style.whiteSpace = "pre-line";

  overlayTextEl = document.createElement("div");
  overlayTextEl.textContent = "";

  box.appendChild(overlayTextEl);
  overlayRoot.appendChild(box);
  document.documentElement.appendChild(overlayRoot);
}

function ensureBoxLayer() {
  if (boxLayer && document.documentElement.contains(boxLayer)) {
    return;
  }

  boxLayer = document.createElement("div");
  boxLayer.id = "yt-character-skipper-box-layer";
  boxLayer.style.position = "fixed";
  boxLayer.style.left = "0";
  boxLayer.style.top = "0";
  boxLayer.style.width = "100vw";
  boxLayer.style.height = "100vh";
  boxLayer.style.zIndex = "999998";
  boxLayer.style.pointerEvents = "none";
  boxLayer.style.display = "none";

  document.documentElement.appendChild(boxLayer);
}

function hideOverlay() {
  if (overlayHideTimer) {
    clearTimeout(overlayHideTimer);
    overlayHideTimer = null;
  }

  if (overlayRoot) {
    overlayRoot.style.display = "none";
  }
}

function showOverlayText(text) {
  if (!runtimeAlive) return;

  ensureOverlay();

  if (!overlayRoot || !overlayTextEl) return;

  if (overlayHideTimer) {
    clearTimeout(overlayHideTimer);
    overlayHideTimer = null;
  }

  overlayTextEl.textContent = text;
  overlayRoot.style.display = "block";
}

function showSkippingOverlay(characterName) {
  showOverlayText(`Skipping ${characterName}...`);
}

function showDebugOverlay(runtimeDebug) {
  if (!runtimeAlive || !runtimeDebug) return;

  const lines = [
    `Mode: ${runtimeDebug.mode || "idle"}`,
    `Detected: ${runtimeDebug.lastDetectedCharacter || "-"}`,
    `Match score: ${
      typeof runtimeDebug.lastMatchScore === "number"
        ? runtimeDebug.lastMatchScore.toFixed(3)
        : "-"
    }`,
    `Faces: ${
      typeof runtimeDebug.lastFaceCount === "number"
        ? runtimeDebug.lastFaceCount
        : 0
    }`,
    `Detect ms: ${
      typeof runtimeDebug.lastDetectionTimeMs === "number"
        ? runtimeDebug.lastDetectionTimeMs
        : 0
    }`,
    `Video time: ${
      typeof runtimeDebug.lastVideoTime === "number"
        ? runtimeDebug.lastVideoTime.toFixed(2)
        : "0.00"
    }s`,
    `No-face streak: ${scanState.consecutiveNoFaceScans}`,
    `Face streak: ${scanState.consecutiveFaceScans}`,
    `Learn candidate: ${autoLearnState.candidateCharacterId || "-"}`,
    `Learn streak: ${autoLearnState.candidateCount}`
  ];

  showOverlayText(lines.join("\n"));
}

function clearFaceBoxes() {
  ensureBoxLayer();

  if (!boxLayer) return;

  boxLayer.innerHTML = "";
  boxLayer.style.display = "none";
}

function drawFaceBoxes(video, analyzedResult) {
  ensureBoxLayer();
  clearFaceBoxes();

  if (!runtimeAlive) return;
  if (!currentSettings?.debugBoxesEnabled) return;
  if (!video) return;
  if (!analyzedResult || !Array.isArray(analyzedResult.faces) || !analyzedResult.faces.length) {
    return;
  }

  const videoRect = video.getBoundingClientRect();
  if (!videoRect.width || !videoRect.height) {
    return;
  }

  const frameWidth = analyzedResult.frameWidth || 1;
  const frameHeight = analyzedResult.frameHeight || 1;

  const scaleX = videoRect.width / frameWidth;
  const scaleY = videoRect.height / frameHeight;

  boxLayer.style.display = "block";

  for (const face of analyzedResult.faces) {
    if (!face.box) continue;

    const matched = !!face.bestMatch?.matched;
    const matchedName = matched ? face.bestMatch.characterName || "Character" : null;
    const matchedDistance =
      matched && typeof face.bestMatch?.distance === "number"
        ? face.bestMatch.distance
        : null;

    const x = videoRect.left + face.box.x * scaleX;
    const y = videoRect.top + face.box.y * scaleY;
    const width = face.box.width * scaleX;
    const height = face.box.height * scaleY;

    const boxEl = document.createElement("div");
    boxEl.style.position = "fixed";
    boxEl.style.left = `${x}px`;
    boxEl.style.top = `${y}px`;
    boxEl.style.width = `${width}px`;
    boxEl.style.height = `${height}px`;
    boxEl.style.borderRadius = "8px";
    boxEl.style.boxSizing = "border-box";
    boxEl.style.border = matched
      ? "2px solid rgba(0, 255, 120, 0.95)"
      : "2px solid rgba(255, 200, 0, 0.95)";
    boxEl.style.boxShadow = "0 0 0 1px rgba(0,0,0,0.35) inset";

    const labelEl = document.createElement("div");
    labelEl.style.position = "absolute";
    labelEl.style.left = "0";
    labelEl.style.top = "-24px";
    labelEl.style.background = matched
      ? "rgba(0, 120, 60, 0.88)"
      : "rgba(140, 110, 0, 0.88)";
    labelEl.style.color = "#fff";
    labelEl.style.fontSize = "11px";
    labelEl.style.padding = "3px 6px";
    labelEl.style.borderRadius = "6px";
    labelEl.style.whiteSpace = "nowrap";

    if (matched) {
      labelEl.textContent = `${matchedName} | ${matchedDistance.toFixed(3)}`;
    } else {
      // Show the closest match distance even for unmatched faces —
      // helps calibrate the threshold ("how far is this unknown?")
      const unknownDist =
        typeof face.bestMatch?.distance === "number"
          ? ` | ${face.bestMatch.distance.toFixed(3)}`
          : "";
      labelEl.textContent = `Unknown${unknownDist}`;
    }

    boxEl.appendChild(labelEl);
    boxLayer.appendChild(boxEl);
  }
}

async function getRuntimeDebugSafe() {
  return await safeCall(() => getRuntimeDebug(), null);
}

async function resetRuntimeDebugSafe() {
  return await safeCall(() => resetRuntimeDebug(), null);
}

async function updateRuntimeDebug(patch) {
  return await safeCall(() => setRuntimeDebug(patch), null);
}

async function setMode(mode, extra = {}) {
  return await updateRuntimeDebug({
    mode,
    ...extra
  });
}

function buildRuntimePatchFromResult(mode, result, videoTime) {
  const patch = {
    mode,
    lastDetectedCharacter: result?.match?.characterName || null,
    lastFaceCount: typeof result?.faceCount === "number" ? result.faceCount : 0,
    lastDetectionTimeMs:
      typeof result?.detectionTimeMs === "number" ? result.detectionTimeMs : 0,
    lastVideoTime: typeof videoTime === "number" ? videoTime : 0
  };

  if (typeof result?.match?.distance === "number") {
    patch.lastMatchScore = result.match.distance;
  }

  return patch;
}

async function recordDetectionIfNeeded(result, videoTime) {
  if (!result?.visible || !result?.match) return;
  if (typeof videoTime !== "number") return;

  const roundedTime = videoTime.toFixed(1);
  const key = `${result.match.characterId}_${roundedTime}`;

  if (lastRecordedDetectionKey === key) {
    return;
  }

  lastRecordedDetectionKey = key;

  await safeCall(
    () =>
      recordCharacterDetection(
        result.match.characterId,
        result.match.characterName,
        typeof result.match.distance === "number" ? result.match.distance : null
      ),
    null
  );
}

async function recordSkipStats(characterId, skippedSeconds) {
  if (!characterId) return;
  await safeCall(() => recordCharacterSkip(characterId, skippedSeconds), null);
}

function resetAutoLearnCandidate() {
  autoLearnState.candidateCharacterId = null;
  autoLearnState.candidateCount = 0;
  autoLearnState.lastCandidateWallTime = 0;
  autoLearnState.lastCandidateVideoTime = -999;
}

function shouldAutoLearn(face) {
  if (!currentSettings?.autoLearnEnabled) return false;
  if (!face?.bestMatch?.matched) return false;
  if (!Array.isArray(face?.descriptor) || !face.descriptor.length) return false;

  const distance = face.bestMatch.distance;
  if (typeof distance !== "number") return false;

  // Scale with the character's own (adaptive) threshold instead of a fixed
  // constant — fixed 0.50 was ABOVE the different-person floor in ArcFace's
  // compressed space, so auto-learn could absorb strangers' descriptors and
  // poison the character.
  const thr = typeof face.bestMatch.threshold === "number" ? face.bestMatch.threshold : 0.50;
  return distance <= thr * 0.95;
}

async function tryAutoLearnFromResult(result, videoTime) {
  if (!currentSettings?.autoLearnEnabled) {
    resetAutoLearnCandidate();
    return;
  }

  if (!result || !Array.isArray(result.faces) || !result.faces.length) {
    resetAutoLearnCandidate();
    return;
  }

  const eligibleFaces = result.faces.filter(shouldAutoLearn);
  if (!eligibleFaces.length) {
    resetAutoLearnCandidate();
    return;
  }

  let bestFace = eligibleFaces[0];
  for (const face of eligibleFaces) {
    const d = face.bestMatch.distance;
    const bestD = bestFace.bestMatch.distance;
    if (d < bestD) {
      bestFace = face;
    }
  }

  const characterId = bestFace.bestMatch.characterId;
  const distance = bestFace.bestMatch.distance;

  if (!characterId || typeof distance !== "number") {
    resetAutoLearnCandidate();
    return;
  }

  const now = Date.now();
  const timeGapOk =
    autoLearnState.lastCandidateWallTime === 0 ||
    now - autoLearnState.lastCandidateWallTime <= 2500;

  const videoGapOk =
    autoLearnState.lastCandidateVideoTime < 0 ||
    Math.abs(videoTime - autoLearnState.lastCandidateVideoTime) <= 3.0;

  if (
    autoLearnState.candidateCharacterId === characterId &&
    timeGapOk &&
    videoGapOk
  ) {
    autoLearnState.candidateCount += 1;
  } else {
    autoLearnState.candidateCharacterId = characterId;
    autoLearnState.candidateCount = 1;
  }

  autoLearnState.lastCandidateWallTime = now;
  autoLearnState.lastCandidateVideoTime = videoTime;

  // Fractions of the character's own threshold (see shouldAutoLearn)
  const _learnThr   = typeof bestFace.bestMatch.threshold === "number"
    ? bestFace.bestMatch.threshold : 0.50;
  const directLearn = distance <= _learnThr * 0.80;
  const mediumLearn = distance <= _learnThr * 0.95 && autoLearnState.candidateCount >= 3;

  if (!directLearn && !mediumLearn) {
    return;
  }

  const key = `${characterId}_${videoTime.toFixed(1)}_${distance.toFixed(3)}_${autoLearnState.candidateCount}`;
  if (lastAutoLearnKey === key) {
    return;
  }

  lastAutoLearnKey = key;

  await safeCall(
    () =>
      tryAutoLearnCharacterDescriptor(
        characterId,
        bestFace.descriptor,
        distance
      ),
    null
  );

  resetAutoLearnCandidate();
}

function applyImmediateVisualState() {
  if (!currentSettings?.extensionEnabled) {
    hideOverlay();
    clearFaceBoxes();
    return;
  }

  if (!currentSettings?.debugOverlayEnabled && !busySkipping) {
    hideOverlay();
  }

  if (!currentSettings?.debugBoxesEnabled) {
    clearFaceBoxes();
  }
}

function installStorageWatcher() {
  if (storageWatcherInstalled) return;
  storageWatcherInstalled = true;

  chrome.storage.onChanged.addListener(async (changes, areaName) => {
    if (!runtimeAlive) return;
    if (areaName !== "local") return;

    const watchedKeys = [
      "extensionEnabled",
      "debugOverlayEnabled",
      "debugBoxesEnabled",
      "detectionThreshold",
      "characters",
      "runtimeDebug",
      "stats",
      "autoLearnEnabled",
      "maxDescriptorsPerCharacter",
      "charactersCollapsed"
    ];

    const touched = watchedKeys.some((key) => key in changes);
    if (!touched) return;

    await loadCurrentSettings();
    if (!runtimeAlive || !currentSettings) return;

    applyImmediateVisualState();

    if (currentSettings.debugOverlayEnabled && !busySkipping) {
      const runtimeDebug = currentSettings.runtimeDebug || null;
      if (runtimeDebug) {
        showDebugOverlay(runtimeDebug);
      }
    }
  });
}

// timeoutMs > 0: if seeked event doesn't fire within that many ms, resolve anyway
// (prevents hangs when the browser is buffering a remote segment).
//
// IMPORTANT — no setTimeout settle delay here.
// "seeked" already guarantees the decoded frame is ready for canvas.drawImage().
// The old setTimeout(resolve, 80) was the main cause of slow scans: Chrome
// throttles background-tab timers from 80 ms → ~1000 ms, multiplying across
// every scan frame and making the whole scan ~10× slower in background tabs.
async function seekVideo(video, time, timeoutMs = 0) {
  return new Promise((resolve) => {
    const target = Math.max(0, Math.min(time, video.duration || time));
    let settled = false;
    let timerId  = null;

    const settle = () => {
      if (settled) return;
      settled = true;
      if (timerId !== null) { clearTimeout(timerId); timerId = null; }
      video.removeEventListener("seeked", settle);
      resolve(); // immediate — frame is ready when "seeked" fires
    };

    video.addEventListener("seeked", settle, { once: true });

    if (timeoutMs > 0) timerId = setTimeout(settle, timeoutMs);

    video.currentTime = target;
  });
}

// ── Compute centroid of a descriptor array ────────────────────────────────
function computeDescriptorCentroid(descriptors) {
  if (!descriptors || !descriptors.length) return null;
  const dim = descriptors[0].length;
  const sum = new Array(dim).fill(0);
  for (const d of descriptors) for (let k = 0; k < dim; k++) sum[k] += d[k];
  return sum.map((v) => v / descriptors.length);
}

// ── Norm-preserving centroid ───────────────────────────────────────────────
// Averaging shrinks vector length, which would make face→centroid distances
// smaller than face→face distances and skew clustering. Rescale the mean back
// to the average norm of its members so distances stay on one scale. Works
// for ANY embedding model regardless of its calibration scale (the worker
// scales arcface descriptors to 0.50, faceres stays at 1.0 — plain
// l2Normalize here would silently break the arcface scale).
function normalizedDescriptorCentroid(descriptors) {
  const mean = computeDescriptorCentroid(descriptors);
  if (!mean) return null;

  let avgNorm = 0;
  for (const d of descriptors) {
    let n = 0;
    for (let k = 0; k < d.length; k++) n += d[k] * d[k];
    avgNorm += Math.sqrt(n);
  }
  avgNorm /= descriptors.length;

  let mNorm = 0;
  for (let k = 0; k < mean.length; k++) mNorm += mean[k] * mean[k];
  mNorm = Math.sqrt(mNorm);
  if (mNorm === 0) return mean;

  const f = avgNorm / mNorm;
  return mean.map((v) => v * f);
}

async function detectTrackedCharacters(video, activeCharacters, threshold) {
  if (!runtimeAlive) return null;

  const rawDetection = await detectFacesFromVideo(video);
  if (!runtimeAlive) return null;

  return analyzeFaceMatches(rawDetection, activeCharacters, threshold);
}

function resetScanState() {
  scanState.lastAnalyzedVideoTime = -999;
  scanState.lastAnalyzeWallTime = 0;
  scanState.consecutiveNoFaceScans = 0;
  scanState.consecutiveFaceScans = 0;
  scanState.consecutiveMatchScans = 0;
  lastRecordedDetectionKey = null;
  lastAutoLearnKey = null;
  resetAutoLearnCandidate();
}

function registerScanResult(result) {
  const hasFaces = !!(result && typeof result.faceCount === "number" && result.faceCount > 0);
  const hasMatch = !!(result && result.visible && result.match);

  if (hasFaces) {
    scanState.consecutiveFaceScans += 1;
    scanState.consecutiveNoFaceScans = 0;
  } else {
    scanState.consecutiveNoFaceScans += 1;
    scanState.consecutiveFaceScans = 0;
  }

  if (hasMatch) {
    scanState.consecutiveMatchScans += 1;
  } else {
    scanState.consecutiveMatchScans = 0;
  }
}

function getMinVideoProgressForNextScan() {
  if (scanState.consecutiveNoFaceScans >= 5) return 2.4;
  if (scanState.consecutiveNoFaceScans >= 3) return 1.6;
  if (scanState.consecutiveFaceScans >= 1) return 0.45;
  return 0.8;
}

function getIdleSleepAfterScan(result) {
  const hasFaces = !!(result && typeof result.faceCount === "number" && result.faceCount > 0);
  const hasMatch = !!(result && result.visible && result.match);

  if (hasMatch) return 120;
  if (hasFaces) return 350;
  if (scanState.consecutiveNoFaceScans >= 5) return 1800;
  if (scanState.consecutiveNoFaceScans >= 3) return 1300;
  return 800;
}

function shouldAnalyzeNow(video) {
  const now = Date.now();
  const currentVideoTime =
    typeof video?.currentTime === "number" ? video.currentTime : 0;

  const minProgress = getMinVideoProgressForNextScan();
  const progressDelta = currentVideoTime - scanState.lastAnalyzedVideoTime;
  const wallDelta = now - scanState.lastAnalyzeWallTime;

  if (progressDelta >= minProgress) return true;
  if (scanState.consecutiveFaceScans >= 1 && wallDelta >= 600) return true;
  if (scanState.consecutiveNoFaceScans >= 3 && wallDelta >= 1800) return true;
  if (wallDelta >= 2500) return true;

  return false;
}

function markAnalyzeMoment(video) {
  scanState.lastAnalyzedVideoTime =
    typeof video?.currentTime === "number" ? video.currentTime : 0;
  scanState.lastAnalyzeWallTime = Date.now();
}

async function renderOverlayFromState(runtimeDebug, skippingName = null) {
  if (!runtimeAlive) {
    hideOverlay();
    clearFaceBoxes();
    return;
  }

  if (!currentSettings?.extensionEnabled) {
    hideOverlay();
    clearFaceBoxes();
    return;
  }

  if (busySkipping) {
    showSkippingOverlay(skippingName || runtimeDebug?.lastDetectedCharacter || "Character");
    return;
  }

  if (currentSettings?.debugOverlayEnabled) {
    showDebugOverlay(runtimeDebug);
    return;
  }

  hideOverlay();
}

async function skipDetectedScene(video, activeCharacters, threshold, detectedName, detectedCharacterId) {
  if (!runtimeAlive) return;
  if (busySkipping) return;

  busySkipping = true;

  // ── Pause/resume: if the user presses play while we are probing
  //    (video is paused by us), treat it as "I want to cancel this skip".
  //    Conversely, extension-triggered play() calls must NOT abort the skip.
  let _userPressedPlay = false;
  let _pendingExtPlays  = 0;       // counter of extension-triggered play() calls
  const _onVideoPlay = () => {
    if (_pendingExtPlays > 0) { _pendingExtPlays--; return; }
    _userPressedPlay = true;       // user pressed play → abort skip
  };
  video.addEventListener("play", _onVideoPlay);

  // Wrapper: play the video in a way the listener won't miscount
  const _extPlay = async () => {
    if (_userPressedPlay) return;  // user already resumed — don't double-play
    _pendingExtPlays++;
    await video.play().catch(() => {});
  };

  try {
    const startTime = video.currentTime;
    const characterName = detectedName || "Character";
    const characterId = detectedCharacterId || null;

    await setMode("skipping", {
      lastDetectedCharacter: characterName,
      lastSkipStartedAt: Date.now(),
      lastVideoTime: startTime
    });

    if (!runtimeAlive) return;

    showSkippingOverlay(characterName);
    video.pause();

    let probeTime = startTime;
    let step = 1.0;
    const maxWindow = 180;
    let cleanFrames = 0;
    let trailingVerificationCount = 0;

    // Live check: the storage watcher keeps currentSettings fresh, so the
    // user disabling THIS character (or deleting it) mid-skip stops the probe
    // within one iteration — previously the stale character list kept the
    // skip running for up to 180 s and the toggle "did nothing".
    const _charStillEnabled = () => {
      if (!characterId) return true; // no id to check — keep legacy behaviour
      const chars = currentSettings?.characters;
      if (!Array.isArray(chars)) return true; // settings unavailable — don't abort
      const c = chars.find((ch) => ch.id === characterId);
      return !!c && c.enabled !== false; // disabled OR deleted → stop
    };

    // Profile-tracking assist state: box of the last CONFIRMED match.
    let _lastMatchBox = null;
    const _stillTracked = (res) => {
      if (!PROFILE_TRACKING_ASSIST || !_lastMatchBox || !res?.faces?.length) return false;
      for (const face of res.faces) {
        if (!face.box || !face.bestMatch) continue;
        if (characterId && face.bestMatch.characterId !== characterId) continue;
        const thr = typeof face.bestMatch.threshold === "number" ? face.bestMatch.threshold : 0.52;
        if (typeof face.bestMatch.distance !== "number") continue;
        if (face.bestMatch.distance > thr * TRACK_NEAR_FACTOR) continue;
        if (boxIoU(face.box, _lastMatchBox) >= TRACK_IOU_MIN) return true;
      }
      return false;
    };

    while (
      runtimeAlive &&
      !_userPressedPlay &&
      currentSettings?.extensionEnabled !== false &&  // user toggled OFF → stop skipping NOW
      _charStillEnabled() &&                          // character toggled off / deleted → stop
      probeTime < video.duration &&
      probeTime - startTime < maxWindow
    ) {
      probeTime += step;
      await seekVideo(video, probeTime);

      if (!runtimeAlive) return;
      if (_userPressedPlay) break;  // user played while we were seeking

      const result = await detectTrackedCharacters(video, activeCharacters, threshold);
      if (!runtimeAlive || !result) return;

      await updateRuntimeDebug(buildRuntimePatchFromResult("skipping", result, probeTime));

      if (currentSettings?.debugBoxesEnabled) {
        drawFaceBoxes(video, result);
      } else {
        clearFaceBoxes();
      }

      showSkippingOverlay(characterName);

      // Confirmed match refreshes the tracking anchor; a near-miss face over
      // the same spot keeps the skip alive through profile/sunglass frames.
      const stillVisible = result.visible || _stillTracked(result);
      if (result.visible && result.matchedFace?.box) {
        _lastMatchBox = result.matchedFace.box;
      }

      if (!stillVisible) {
        cleanFrames += 1;
        trailingVerificationCount += 1;
      } else {
        cleanFrames = 0;
        trailingVerificationCount = 0;
      }

      if (cleanFrames >= 2) {
        const verifyAhead = probeTime + 0.8;
        if (verifyAhead < video.duration) {
          await seekVideo(video, verifyAhead);

          if (_userPressedPlay) break;  // user played during verify seek
          const verifyResult = await detectTrackedCharacters(video, activeCharacters, threshold);
          if (!runtimeAlive || !verifyResult) return;

          const verifyVisible = verifyResult.visible || _stillTracked(verifyResult);
          if (verifyResult.visible && verifyResult.matchedFace?.box) {
            _lastMatchBox = verifyResult.matchedFace.box;
          }

          if (!verifyVisible) {
            // Character is confirmed gone — seek BACK to probeTime (where character
            // actually disappeared), not to verifyAhead, to avoid over-skipping.
            await seekVideo(video, probeTime);

            const skippedSeconds = probeTime - startTime;

            await setMode("idle", {
              lastSkipEndedAt: Date.now(),
              lastSkipDuration: skippedSeconds,
              lastVideoTime: probeTime
            });

            await recordSkipStats(characterId, skippedSeconds);

            resetScanState();

            await _extPlay();
            hideOverlay();
            clearFaceBoxes();
            return;
          } else {
            cleanFrames = 0;
            trailingVerificationCount = 0;
            probeTime = verifyAhead;
            step = 0.9;
            continue;
          }
        } else {
          // Near end of video and character is gone — stop here immediately.
          await seekVideo(video, probeTime);

          const skippedSeconds = probeTime - startTime;

          await setMode("idle", {
            lastSkipEndedAt: Date.now(),
            lastSkipDuration: skippedSeconds,
            lastVideoTime: probeTime
          });

          await recordSkipStats(characterId, skippedSeconds);

          resetScanState();

          await _extPlay();
          hideOverlay();
          clearFaceBoxes();
          return;
        }
      }

      if (stillVisible) {
        step = 1.0;
      } else if (trailingVerificationCount === 1) {
        step = 0.5;
      } else if (trailingVerificationCount >= 2) {
        step = 0.8;
      } else {
        step = 0.7;
      }
    }

    const skippedSeconds = probeTime - startTime;

    await setMode("idle", {
      lastSkipEndedAt: Date.now(),
      lastSkipDuration: skippedSeconds,
      lastVideoTime: probeTime
    });

    await recordSkipStats(characterId, skippedSeconds);

    resetScanState();

    await _extPlay();
    hideOverlay();
    clearFaceBoxes();
  } catch (err) {
    if (isContextInvalidatedError(err)) {
      killRuntime(err);
      return;
    }

    console.error("[CONTENT] skipDetectedScene error:", err);

    await setMode("error", {
      lastVideoTime: typeof video?.currentTime === "number" ? video.currentTime : 0
    });

    hideOverlay();
    clearFaceBoxes();
    await _extPlay();
  } finally {
    video.removeEventListener("play", _onVideoPlay);
    busySkipping = false;
  }
}

async function mainLoop() {
  if (skipLoopStarted) return;
  skipLoopStarted = true;

  ensureOverlay();
  ensureBoxLayer();

  await loadCurrentSettings();
  if (!runtimeAlive) return;

  await resetRuntimeDebugSafe();
  if (!runtimeAlive) return;

  resetScanState();
  installStorageWatcher();

  // Pre-warm: start sandbox init and video discovery in parallel.
  // Model loading takes ~2-4 s; the YouTube player takes ~1-2 s to appear.
  // Running both concurrently hides most of the model-load latency.
  const [_sandboxSettle, _videoSettle] = await Promise.allSettled([
    initContentSandboxWithRetry(),
    waitForVideo()
  ]);

  if (!runtimeAlive) return;

  if (_sandboxSettle.status === "fulfilled") {
    sandboxInitialized = true;
    console.log("[CONTENT] sandbox initialized successfully");
  } else {
    const _sandboxErr = _sandboxSettle.reason;
    if (isContextInvalidatedError(_sandboxErr)) {
      killRuntime(_sandboxErr);
      return;
    }
    // Don't exit — keep loop alive so we can retry the sandbox later
    console.error("[CONTENT] sandbox init failed, will retry:", _sandboxErr.message);
    showOverlayText("Face recognition failed to start — retrying automatically…");
    await setMode("error");
  }

  if (!runtimeAlive) return;

  const video = _videoSettle.status === "fulfilled" ? _videoSettle.value : null;
  if (!runtimeAlive) return;

  if (!video) {
    console.error("[CONTENT] video not found");
    await setMode("error");
    return;
  }

  console.log("[CONTENT] main loop started");

  while (runtimeAlive) {
    try {
      currentSettings = await safeCall(() => getAppData(), null);
      if (!runtimeAlive) return;
      if (!currentSettings) return;

      if (!currentSettings.extensionEnabled) {
        await setMode("disabled", {
          lastVideoTime: typeof video.currentTime === "number" ? video.currentTime : 0
        });

        hideOverlay();
        clearFaceBoxes();
        resetScanState();
        await sleep(1000);
        continue;
      }

      const activeCharacters = currentSettings.characters.filter((character) => character.enabled);
      const allSavedCharacters = currentSettings.characters || [];

      // If sandbox not ready, retry every 30 s then skip this iteration
      if (!sandboxInitialized) {
        const now = Date.now();
        if (now - lastSandboxRetryAt >= 30000) {
          lastSandboxRetryAt = now;
          try {
            resetSandbox();
            resetSandbox2();
            await initContentSandboxWithRetry();
            sandboxInitialized = true;
            console.log("[CONTENT] sandbox recovered");
            hideOverlay();
          } catch (_) { /* will retry later */ }
        }
        if (!sandboxInitialized) {
          await sleep(2000);
          continue;
        }
      }

      // Even with no active characters, run discovery scanning every ~3s of video time
      if (!activeCharacters.length) {
        const vt = typeof video.currentTime === "number" ? video.currentTime : 0;

        await setMode("idle", { lastVideoTime: vt });
        hideOverlay();
        clearFaceBoxes();

        // Discovery scan: detect faces to build the discovered clusters list
        const videoAdvanced = vt - lastDiscoveryVideoTime >= 3.0;
        if (
          videoAdvanced &&
          !video.paused && !video.ended && video.readyState >= 2
        ) {
          lastDiscoveryVideoTime = vt;
          discoveryAttemptCount++;
          try {
            const rawDetection = await detectFacesFromVideo(video);
            if (rawDetection && rawDetection.faces.length) {
              discoveryFaceFoundCount += rawDetection.faces.length;
              const enriched = await enrichFacesWithThumbnails(rawDetection.faces, rawDetection, video);
              updateDiscoveredClusters(enriched, allSavedCharacters);
              console.log(`[CONTENT] discovery: ${rawDetection.faces.length} face(s), clusters: ${discoveredClusters.length}`);
            } else {
              console.log("[CONTENT] discovery: no faces in this frame");
            }
          } catch (e) {
            console.warn("[CONTENT] discovery scan error:", e?.message);
          }
        }

        await sleep(1000);
        continue;
      }

      if (busySkipping) {
        const runtimeDebug = await getRuntimeDebugSafe();
        if (!runtimeAlive) return;

        await renderOverlayFromState(runtimeDebug);
        await sleep(250);
        continue;
      }

      if (video.paused || video.ended || video.readyState < 2) {
        await setMode("waiting", {
          lastVideoTime: typeof video.currentTime === "number" ? video.currentTime : 0
        });

        const runtimeDebug = await getRuntimeDebugSafe();
        if (!runtimeAlive) return;

        await renderOverlayFromState(runtimeDebug);
        clearFaceBoxes();
        await sleep(700);
        continue;
      }

      if (!shouldAnalyzeNow(video)) {
        if (currentSettings.debugBoxesEnabled) {
          clearFaceBoxes();
        }

        if (currentSettings.debugOverlayEnabled) {
          const runtimeDebug = await getRuntimeDebugSafe();
          if (!runtimeAlive) return;
          await renderOverlayFromState(runtimeDebug);
        } else {
          hideOverlay();
        }

        await sleep(220);
        continue;
      }

      markAnalyzeMoment(video);

      let result;
      try {
        result = await detectTrackedCharacters(
          video,
          activeCharacters,
          currentSettings.detectionThreshold
        );
      } catch (detectErr) {
        if (isContextInvalidatedError(detectErr)) {
          killRuntime(detectErr);
          return;
        }
        console.error("[CONTENT] detection failed:", detectErr.message);
        // Try to reinitialize sandbox on detection failure
        if (detectErr.message && (detectErr.message.includes("timeout") || detectErr.message.includes("Sandbox"))) {
          console.log("[CONTENT] attempting sandbox recovery...");
          try {
            resetSandbox();
            resetSandbox2();
            sandboxInitialized = false;
            await initContentSandboxWithRetry();
            sandboxInitialized = true;
            console.log("[CONTENT] sandbox recovered");
          } catch (recoveryErr) {
            console.error("[CONTENT] sandbox recovery failed:", recoveryErr);
            showOverlayText("Detection error - reload page to fix");
          }
        }
        await sleep(2000);
        continue;
      }

      if (!runtimeAlive) return;
      if (!result) {
        await sleep(500);
        continue;
      }

      registerScanResult(result);

      await updateRuntimeDebug(
        buildRuntimePatchFromResult("scanning", result, video.currentTime)
      );

      if (result.visible && result.match) {
        await recordDetectionIfNeeded(result, video.currentTime);
      }

      await tryAutoLearnFromResult(result, video.currentTime);

      // Passive discovery: cluster any faces that didn't match a saved character
      if (result.faces && result.faces.length) {
        const unknownFaces = result.faces.filter((f) => !f.bestMatch?.matched);
        if (unknownFaces.length) {
          const vt = video.currentTime;
          const shouldDiscoverNow = vt - lastDiscoveryVideoTime >= 2.0;
          if (shouldDiscoverNow) {
            lastDiscoveryVideoTime = vt;
            try {
              const enriched = await enrichFacesWithThumbnails(unknownFaces, result, video);
              updateDiscoveredClusters(enriched, allSavedCharacters);
            } catch (_) {}
          }
        }
      }

      if (currentSettings.debugBoxesEnabled) {
        drawFaceBoxes(video, result);
      } else {
        clearFaceBoxes();
      }

      if (result.visible && result.match) {
        await skipDetectedScene(
          video,
          activeCharacters,
          currentSettings.detectionThreshold,
          result.match.characterName,
          result.match.characterId
        );

        await sleep(180);
        continue;
      }

      if (currentSettings.debugOverlayEnabled) {
        const runtimeDebug = await getRuntimeDebugSafe();
        if (!runtimeAlive) return;

        await renderOverlayFromState(runtimeDebug);
      } else {
        hideOverlay();
      }

      await sleep(getIdleSleepAfterScan(result));
    } catch (err) {
      if (isContextInvalidatedError(err)) {
        killRuntime(err);
        return;
      }

      console.error("[CONTENT] main loop error:", err);

      await setMode("error", {
        lastVideoTime: typeof video?.currentTime === "number" ? video.currentTime : 0
      });

      hideOverlay();
      clearFaceBoxes();
      await sleep(1500);
    }
  }
}


// ── Full video scan (seek through entire video, cluster all faces) ────────
async function runVideoScan(scanId) {
  if (!sandboxInitialized) {
    throw new Error("Yüz tanıma motoru hazır değil — biraz bekle ve tekrar dene.");
  }

  const video = getVideo();
  if (!video || video.readyState < 2) {
    throw new Error("Oynatılabilir video bulunamadı.");
  }

  const duration = video.duration;
  if (!duration || !isFinite(duration) || duration < 5) {
    throw new Error("Video süresi okunamadı — video tam yüklenene kadar bekle.");
  }

  const wasPaused = video.paused;
  const savedTime = video.currentTime;
  const wasMuted  = video.muted;
  const wasVolume = video.volume;

  // ── Prevent Chrome background-tab throttling ────────────────────────────
  // Chrome throttles JS timers and WebGL tasks in background tabs that have
  // NO active audio.  A *muted* video is treated as "silent" and does NOT
  // prevent throttling.  Setting volume=0.001 (inaudible to humans) keeps the
  // audio pipeline "active" so Chrome skips the throttle entirely.
  //
  // Additionally, YouTube's own event handlers call video.pause() after each
  // programmatic seek — so we add a "pause" listener that immediately re-plays,
  // ensuring the video stays in playing state throughout the whole scan.
  video.muted  = false;
  video.volume = 0.001; // near-silent but Chrome counts it as active audio
  video.play().catch(() => {});

  const _keepScanPlaying = () => {
    if (!video.ended) video.play().catch(() => {});
  };
  video.addEventListener("pause", _keepScanPlaying);

  try {

  // Sample every SCAN_INTERVAL_S seconds across the whole video (no hard cap).
  // Detection is fully decoupled from seeking (in-flight queue below), so
  // wall time ≈ sampleCount × seek-time (~150-250 ms) — detection is FREE.
  const SCAN_INTERVAL_S = 5;
  const sampleCount     = Math.max(1, Math.floor(duration / SCAN_INTERVAL_S));
  const sampleTimes     = Array.from({ length: sampleCount }, (_, i) =>
    Math.min((i + 0.5) * SCAN_INTERVAL_S, duration - 1)
  );

  activeScan.total = sampleTimes.length;

  const clusters        = [];
  // ── Two-phase clustering ──────────────────────────────────────────────────
  // Phase 1 (during scan): group with a TIGHT threshold so every cluster is
  //   almost certainly a single person (fragments are fine — a fragmented
  //   person is recoverable, two people fused into one cluster is not, and a
  //   loose rolling centroid drifts across identities: that is exactly how a
  //   10-person video collapsed into one ×662 blob).
  // Phase 2 (post-scan): merge fragments with a threshold AUTO-CALIBRATED
  //   from the gap in this video's own inter-cluster distance distribution.
  const CLUSTER_THRESH  = 0.45;  // tight phase-1 centroid distance
  const MAX_CLUSTERS    = 150;   // fragments allowed — phase 2 merges them
  const MAX_DESCRIPTORS = 40;
  const MAX_THUMBNAILS  = 6; // up to 6 thumbnails stored per cluster for gallery view

  // Quality gate: tiny or low-confidence faces produce NOISY descriptors that
  // fragment clusters ("×1 ×1 ×1…") and later poison recognition. Skip them —
  // the same person will appear larger in other sampled frames.
  // Score 0.5 also rejects non-face false positives (hands, objects) that
  // BlazeFace occasionally reports at low confidence.
  const MIN_FACE_SIDE  = 36;   // px at 640-wide capture
  const MIN_FACE_SCORE = 0.5;
  const faceQualityOk = (face) =>
    (typeof face.score !== "number" || face.score >= MIN_FACE_SCORE) &&
    (!face.box || (face.box.width >= MIN_FACE_SIDE && face.box.height >= MIN_FACE_SIDE));

  // ── Centroid-based cluster assignment ─────────────────────────────────────
  // Compare against the cluster CENTROID, not "any member" — any-member
  // matching chains outliers together and splits real identities apart.
  const assignFaceToClusters = (desc, makeThumb) => {
    let best = null, bestDist = Infinity;
    for (const cl of clusters) {
      const d = euclideanDistance(desc, cl.centroid);
      if (d < CLUSTER_THRESH && d < bestDist) { bestDist = d; best = cl; }
    }

    if (best) {
      if (best.descriptors.length < MAX_DESCRIPTORS) {
        best.descriptors.push(desc);
        best.centroid = normalizedDescriptorCentroid(best.descriptors);
      }
      best.count++;
      best.lastSeenAt = Date.now();
      if (!best.thumbnail || best.thumbnails.length < MAX_THUMBNAILS) {
        const thumb = makeThumb();
        if (thumb) {
          if (!best.thumbnail) best.thumbnail = thumb;
          if (best.thumbnails.length < MAX_THUMBNAILS) best.thumbnails.push(thumb);
        }
      }
    } else if (clusters.length < MAX_CLUSTERS) {
      const thumb = makeThumb();
      clusters.push({
        id:          "scan_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6),
        descriptors: [desc],
        centroid:    desc.slice(),
        thumbnail:   thumb,
        thumbnails:  thumb ? [thumb] : [],
        count:       1,
        firstSeenAt: Date.now(),
        lastSeenAt:  Date.now()
      });
    }
  };

  // ── Decoupled scan pipeline ───────────────────────────────────────────────
  // Old flow AWAITED each detection before the next capture, so per-frame cost
  // was seek + detect. New flow: capture raw ImageData (no JPEG round-trip),
  // fire detection into the worker WITHOUT awaiting, and seek immediately.
  // Up to MAX_INFLIGHT detections ride along behind the seeks; per-frame wall
  // time collapses to just the seek. Thumbnails are cropped from the retained
  // ImageData only AFTER a face is confirmed — empty frames cost nothing.
  const SEEK_TIMEOUT_MS = 4000;
  const MAX_INFLIGHT    = 3;   // bounded queue — ~1.3 MB pixels per slot
  const inFlight        = []; // [{ promise, imageData, t }]

  const processJob = async (job) => {
    try {
      const detection = await job.promise;
      if (!detection?.faces?.length) return;

      for (const face of detection.faces) {
        const desc = face.descriptor;
        if (!Array.isArray(desc) || !desc.length) continue;
        if (!faceQualityOk(face)) continue;

        assignFaceToClusters(desc, () =>
          face.box ? cropFaceFromImageData(job.imageData, face.box) : null
        );
      }
    } catch (e) {
      console.warn("[CONTENT] scan frame error at", job.t.toFixed(1) + "s:", e?.message);
    } finally {
      job.imageData = null; // release pixels ASAP
    }
  };

  let currentSeekPromise = seekVideo(video, sampleTimes[0], SEEK_TIMEOUT_MS);

  for (let i = 0; i < sampleTimes.length; i++) {
    if (activeScan?.id !== scanId) break;
    activeScan.progress = i;

    try {
      // Wait for the seek that was pre-started at the end of the previous iteration
      await currentSeekPromise;
      if (!runtimeAlive || activeScan?.id !== scanId) break;

      // Raw-pixel snapshot at 640 px — SAME resolution as live detection, so
      // scan descriptors are directly comparable to live ones (this is what
      // makes saved characters actually match during playback).
      let captured = null;
      try { captured = captureVideoFrameImageData(video, 640); } catch (_) {}

      // ── PIPELINE: start seeking to the next frame IMMEDIATELY ───────────
      if (i + 1 < sampleTimes.length) {
        currentSeekPromise = seekVideo(video, sampleTimes[i + 1], SEEK_TIMEOUT_MS);
      }

      if (!captured) continue;

      // Fire detection WITHOUT awaiting — it runs in the worker while we seek.
      inFlight.push({
        promise:   detectFacesFromImageData(captured.imageData),
        imageData: captured.imageData,
        t:         sampleTimes[i],
      });

      // Backpressure: only block once the queue is full (oldest job first,
      // which by then has almost certainly already finished).
      if (inFlight.length >= MAX_INFLIGHT) {
        await processJob(inFlight.shift());
      }
    } catch (e) {
      console.warn("[CONTENT] scan frame error at", sampleTimes[i].toFixed(1) + "s:", e?.message);
    }
  }

  // Drain remaining in-flight detections
  while (inFlight.length) {
    await processJob(inFlight.shift());
  }

  // Restore audio + video state before seeking back
  video.removeEventListener("pause", _keepScanPlaying);
  video.muted  = wasMuted;
  video.volume = wasVolume;

  // Restore video position and play-state
  try {
    await seekVideo(video, savedTime);
    if (wasPaused) video.pause(); else video.play().catch(() => {});
  } catch (_) {}

  if (activeScan?.id !== scanId) return;

  // ── Post-scan: AUTO-CALIBRATED merge threshold ────────────────────────────
  // Fixed thresholds keep failing in both directions (×1 fragments on one
  // video, a 10-person ×662 blob on another) because every video/model has
  // its own distance scale. So derive the cut from THIS video's data: sort
  // all inter-cluster centroid distances — same-person fragment pairs form
  // the low end, different-person pairs the high end, and the identity
  // boundary shows up as the largest GAP between consecutive distances.
  for (const cl of clusters) {
    if (!cl.centroid) cl.centroid = normalizedDescriptorCentroid(cl.descriptors);
  }

  const interDists = [];
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      interDists.push(euclideanDistance(clusters[i].centroid, clusters[j].centroid));
    }
  }
  interDists.sort((a, b) => a - b);

  // Diagnostics — paste these when reporting clustering issues.
  const pct = (p) => interDists.length
    ? interDists[Math.min(interDists.length - 1, Math.floor(interDists.length * p))]
    : NaN;
  console.log(
    "[CONTENT] scan calib: " + clusters.length + " raw clusters, inter-centroid dists " +
    "p10=" + pct(0.10).toFixed(3) + " p25=" + pct(0.25).toFixed(3) +
    " p50=" + pct(0.50).toFixed(3) + " p75=" + pct(0.75).toFixed(3) +
    " p90=" + pct(0.90).toFixed(3)
  );

  // Largest gap between consecutive sorted distances inside the plausible
  // identity-boundary band decides the cut. The band depends on the active
  // embedding model (engineModel set by faceManager on INIT_OK):
  //   arcface (scaled 0.50): same-person fragments ~0.30-0.50 apart,
  //     different people ≥ ~0.63 → gap lives around 0.50-0.63.
  //   faceres (scale 1.0):   same ~0.45-0.75, different ~0.85-1.1.
  //
  // No clear gap = unimodal distances = single-person video → fall back to
  // p75 of the distribution (merges most fragments of that one person),
  // clamped to stay under the model's different-person floor.
  const isArc = (typeof engineModel === "string" && engineModel === "arcface");
  const BAND_LO   = isArc ? 0.32 : 0.45;
  const BAND_HI   = isArc ? 0.80 : 0.95;
  const FB_MIN    = isArc ? 0.42 : 0.55;
  const FB_MAX    = isArc ? 0.56 : 0.68;
  const CEILING   = isArc ? 0.62 : 0.82; // hard cap — never cut at the different-person floor

  const fallbackThresh = Math.min(FB_MAX, Math.max(FB_MIN, pct(0.75) || FB_MIN));
  let MERGE_THRESH = fallbackThresh;
  let bestGap = 0;
  let gapTop  = null; // smallest observed different-person distance (upper gap edge)
  for (let k = 0; k + 1 < interDists.length; k++) {
    const a = interDists[k], b = interDists[k + 1];
    if (a < BAND_LO || b > BAND_HI) continue; // only search the plausible band
    const gap = b - a;
    if (gap > bestGap) { bestGap = gap; MERGE_THRESH = (a + b) / 2; gapTop = b; }
  }
  if (bestGap < 0.04) { MERGE_THRESH = fallbackThresh; gapTop = null; } // gap too weak
  MERGE_THRESH = Math.min(MERGE_THRESH, CEILING);
  console.log(
    "[CONTENT] scan calib: model=" + (engineModel || "?") +
    " merge threshold=" + MERGE_THRESH.toFixed(3) +
    " (gap=" + bestGap.toFixed(3) + ", gapTop=" + (gapTop ? gapTop.toFixed(3) : "-") +
    ", fallback=" + fallbackThresh.toFixed(3) + ")"
  );

  const mergeInto = (target, src) => {
    for (const desc of src.descriptors) {
      if (target.descriptors.length < MAX_DESCRIPTORS) target.descriptors.push(desc);
    }
    target.centroid = normalizedDescriptorCentroid(target.descriptors);
    target.count += src.count;
    if (!target.thumbnail && src.thumbnail) target.thumbnail = src.thumbnail;
    if (!Array.isArray(target.thumbnails)) target.thumbnails = [];
    for (const t of (src.thumbnails || [])) {
      if (target.thumbnails.length < MAX_THUMBNAILS) target.thumbnails.push(t);
    }
  };

  let mergedSomething = true;
  while (mergedSomething) {
    mergedSomething = false;
    // Find the globally closest pair below threshold, merge it, repeat.
    let bi = -1, bj = -1, bd = Infinity;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const d = euclideanDistance(clusters[i].centroid, clusters[j].centroid);
        if (d < MERGE_THRESH && d < bd) { bd = d; bi = i; bj = j; }
      }
    }
    if (bi >= 0) {
      mergeInto(clusters[bi], clusters[bj]);
      clusters.splice(bj, 1);
      mergedSomething = true;
    }
  }

  // ── Rescue pass: absorb hard-pose fragments into established clusters ─────
  // Sunglasses / full-profile / harsh-light shots sit FAR from a cluster's
  // frontal-dominated CENTROID, so the centroid merge above strands them as
  // ×1 fragments. But such a shot is usually CLOSE to some individual members
  // of the right cluster (other profile shots absorbed earlier). So compare
  // small fragments member-to-member against ESTABLISHED clusters. Evidence
  // requirements keep the risk of absorbing a different person low:
  //   • fragment is small (count ≤ 2), target is established (count ≥ 4)
  //   • distance = mean of the 3 NEAREST member pairs (single-outlier proof)
  //   • threshold stays strictly below the measured different-person floor
  //     (gapTop) when the video provided one
  const RESCUE_CAP    = isArc ? 0.60 : 0.80;
  const RESCUE_THRESH = gapTop !== null
    ? Math.min(gapTop - 0.02, MERGE_THRESH * 1.25, RESCUE_CAP)
    : Math.min(MERGE_THRESH * 1.15, RESCUE_CAP);

  const fragToTargetDist = (frag, target) => {
    const all = [];
    for (const fd of frag.descriptors) {
      for (const td of target.descriptors) {
        all.push(euclideanDistance(fd, td));
      }
    }
    all.sort((x, y) => x - y);
    const k = Math.min(3, all.length);
    let s = 0;
    for (let i = 0; i < k; i++) s += all[i];
    return k ? s / k : Infinity;
  };

  let rescued = 0;
  let rescueChanged = true;
  while (rescueChanged) {
    rescueChanged = false;
    for (let j = clusters.length - 1; j >= 0; j--) {
      const frag = clusters[j];
      if (frag.count > 2) continue;

      let bestTarget = null, bestD = Infinity;
      for (const target of clusters) {
        if (target === frag || target.count < 4) continue;
        const d = fragToTargetDist(frag, target);
        if (d < RESCUE_THRESH && d < bestD) { bestD = d; bestTarget = target; }
      }

      if (bestTarget) {
        mergeInto(bestTarget, frag);
        clusters.splice(j, 1);
        rescued++;
        rescueChanged = true;
      }
    }
  }
  if (rescued) {
    console.log("[CONTENT] scan calib: rescued " + rescued +
      " hard-pose fragment(s) via member-level matching (thresh=" + RESCUE_THRESH.toFixed(3) + ")");
  }

  clusters.sort((a, b) => b.count - a.count);
  activeScan.clusters  = clusters;
  activeScan.status    = "done";
  activeScan.progress  = sampleTimes.length;
  console.log(`[CONTENT] full scan done: ${clusters.length} unique persons across ${sampleTimes.length} frames (post-merge)`);

  } finally {
    // Safety: always restore audio state and remove listener on any exit path
    video.removeEventListener("pause", _keepScanPlaying);
    try { video.muted  = wasMuted;  } catch (_) {}
    try { video.volume = wasVolume; } catch (_) {}
  }
}

// ── YouTube SPA navigation: cancel stale scan when user switches videos ───
// YouTube is an SPA — the content script stays alive across video navigations.
// If a scan was running for the old video, cancel it so the popup shows idle.
document.addEventListener("yt-navigate-finish", () => {
  if (activeScan && activeScan.status === "running") {
    console.log("[CONTENT] yt-navigate-finish: cancelling stale scan");
    activeScan.status = "error";
    activeScan.error  = "Video değiştirildi — tarama iptal edildi.";
  }
  // Reset discovery state for the new video
  discoveredClusters.length = 0;
  lastDiscoveryVideoTime    = -999;
  discoveryAttemptCount     = 0;
  discoveryFaceFoundCount   = 0;
});

// ── GET_DISCOVERED_CLUSTERS ───────────────────────────────────────────────
// Returns the clusters accumulated passively during normal playback.
// Single synchronous sendResponse — no message-port issues.
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "GET_DISCOVERED_CLUSTERS") return false;

  sendResponse({
    ok: true,
    clusters: discoveredClusters.map((c) => ({
      id: c.id,
      descriptors: c.descriptors,
      thumbnail: c.thumbnail || null,
      count: c.count,
      firstSeenAt: c.firstSeenAt,
      lastSeenAt: c.lastSeenAt
    })),
    // Debug fields so popup can show meaningful status
    sandboxReady: sandboxInitialized,
    discoveryAttempts: discoveryAttemptCount,
    discoveryFacesFound: discoveryFaceFoundCount,
    loopAlive: runtimeAlive && skipLoopStarted
  });

  return false; // synchronous — no async sendResponse needed
});

// ── START_SCAN ────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "START_SCAN") return false;

  if (activeScan?.status === "running") {
    sendResponse({ ok: true, scanId: activeScan.id, alreadyRunning: true });
    return false;
  }

  const scanId = "scan_" + Date.now();
  activeScan = { id: scanId, status: "running", progress: 0, total: 0, clusters: [], error: null, mode: "normal" };

  runVideoScan(scanId).catch((e) => {
    if (activeScan?.id === scanId) {
      activeScan.status = "error";
      activeScan.error  = e?.message || "Tarama başarısız.";
      console.error("[CONTENT] runVideoScan error:", e?.message);
    }
  });

  sendResponse({ ok: true, scanId });
  return false;
});

// ── GET_SCAN_STATUS ───────────────────────────────────────────────────────
// Popup polls this every ~600 ms to track progress and get final clusters.
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "GET_SCAN_STATUS") return false;

  if (!activeScan) {
    sendResponse({ status: "idle", sandboxReady: sandboxInitialized });
    return false;
  }

  sendResponse({
    status:      activeScan.status,
    progress:    activeScan.progress,
    total:       activeScan.total,
    error:       activeScan.error,
    mode:        activeScan.mode || "normal",
    sandboxReady: sandboxInitialized,
    clusters: activeScan.status === "done"
      ? activeScan.clusters.map((c) => ({
          id:          c.id,
          descriptors: c.descriptors,
          thumbnail:   c.thumbnail || null,
          thumbnails:  Array.isArray(c.thumbnails) ? c.thumbnails
                         : (c.thumbnail ? [c.thumbnail] : []),
          count:       c.count,
          firstSeenAt: c.firstSeenAt,
          lastSeenAt:  c.lastSeenAt
        }))
      : []
  });
  return false;
});

// ── Crop face from a frame dataURL → square JPEG thumbnail ────────────────
async function cropFaceDataUrl(frameDataUrl, box, frameWidth, frameHeight) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const scaleX = img.width / (frameWidth || img.width);
        const scaleY = img.height / (frameHeight || img.height);

        const padding = Math.max(box.width, box.height) * 0.25;
        const sx = Math.max(0, (box.x - padding) * scaleX);
        const sy = Math.max(0, (box.y - padding) * scaleY);
        const sw = Math.min(img.width - sx, (box.width + padding * 2) * scaleX);
        const sh = Math.min(img.height - sy, (box.height + padding * 2) * scaleY);
        const size = Math.ceil(Math.max(sw, sh, 48));

        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);
        resolve(canvas.toDataURL("image/jpeg", 0.8));
      } catch (_) {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = frameDataUrl;
  });
}

// ── CAPTURE_FACES_REQUEST ──────────────────────────────────────────────────
// Popup requests a face-capture from the current video frame.
// Returns {ok, faces:[{index, descriptor, box, score}], frameDataUrl, frameWidth, frameHeight}
// or {error: string} on failure.
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "CAPTURE_FACES_REQUEST") return false;

  (async () => {
    try {
      const video = getVideo();
      if (!video || video.readyState < 2) {
        sendResponse({ error: "No playable video found on this page." });
        return;
      }

      // Make sure the sandbox is ready (it should already be initialised by mainLoop,
      // but guard against the popup opening before mainLoop has finished init).
      await initContentSandboxWithRetry();

      const rawDetection = await detectFacesFromVideo(video);

      if (!rawDetection || !rawDetection.faceCount || !rawDetection.faces.length) {
        sendResponse({ error: "No faces detected in the current frame. Try a different moment in the video." });
        return;
      }

      // Capture a high-quality frame for thumbnail cropping in the popup.
      const frameCapture = captureVideoFrameDataUrl(video, 640, 0.88);

      sendResponse({
        ok: true,
        faces: rawDetection.faces.map((face, index) => ({
          index,
          descriptor: face.descriptor,
          box: face.box,
          score: face.score
        })),
        frameDataUrl: frameCapture.dataUrl,
        frameWidth: frameCapture.frameWidth,
        frameHeight: frameCapture.frameHeight
      });
    } catch (err) {
      console.error("[CONTENT] CAPTURE_FACES_REQUEST error:", err);
      sendResponse({ error: err?.message || "Face capture failed." });
    }
  })();

  return true; // keep message channel open for async sendResponse
});

// ── Extract descriptor from a photo (used by popup) ───────────────────────
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "EXTRACT_DESCRIPTOR") return false;

  (async () => {
    try {
      await initContentSandboxWithRetry();
      const descriptor = await extractSingleDescriptorFromImageDataUrl(message.dataUrl);
      sendResponse({ descriptor });
    } catch (err) {
      console.error("[CONTENT] EXTRACT_DESCRIPTOR error:", err);
      sendResponse({ error: err?.message || "Descriptor extraction failed." });
    }
  })();

  return true;
});

// ── Re-init sandbox when tab becomes visible ──────────────────────────────
// Chrome throttles WebGL in background tabs, which can cause the humangl/
// webgl backends to fail during model loading. When the user switches back
// to this tab, force an immediate retry so the engine is ready quickly.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && !sandboxInitialized && runtimeAlive) {
    console.log("[CONTENT] tab became visible — forcing sandbox retry");
    lastSandboxRetryAt = 0; // bypass the 30-s cooldown
  }
});

mainLoop();