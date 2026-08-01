// ── Descriptor arithmetic helpers ─────────────────────────────────────────
// Used for smart centroid-based auto-learn. Intentionally kept private to
// this file so storage.js stays self-contained (no faceManager.js dependency).
function _descDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

function _descAvg(a, b, alpha = 0.5) {
  // Weighted average: alpha = weight of b, (1-alpha) = weight of a
  return a.map((v, i) => v * (1 - alpha) + b[i] * alpha);
}

const DEFAULT_SETTINGS = {
  schemaVersion: 1,
  extensionEnabled: true,
  detectionThreshold: 0.52,
  characters: [],
  debugOverlayEnabled: false,
  debugBoxesEnabled: false,
  autoLearnEnabled: true,
  maxDescriptorsPerCharacter: 30,
  charactersCollapsed: false,
  runtimeDebug: {
    mode: "idle",
    lastDetectedCharacter: null,
    lastMatchScore: null,
    lastFaceCount: 0,
    lastDetectionTimeMs: 0,
    lastVideoTime: 0,
    lastSkipStartedAt: null,
    lastSkipEndedAt: null,
    lastSkipDuration: 0,
    updatedAt: 0
  },
  stats: {
    totalDetections: 0,
    totalSkips: 0,
    totalSkippedSeconds: 0,
    lastDetectedCharacter: null,
    lastMatchScore: null,
    lastDetectionAt: null,
    updatedAt: 0
  }
};

function createDefaultRuntimeDebug() {
  return {
    mode: "idle",
    lastDetectedCharacter: null,
    lastMatchScore: null,
    lastFaceCount: 0,
    lastDetectionTimeMs: 0,
    lastVideoTime: 0,
    lastSkipStartedAt: null,
    lastSkipEndedAt: null,
    lastSkipDuration: 0,
    updatedAt: Date.now()
  };
}

function createDefaultCharacterStats() {
  return {
    detectCount: 0,
    skipCount: 0,
    skippedSeconds: 0,
    lastSeenAt: null,
    lastMatchScore: null,
    autoLearnCount: 0
  };
}

function createDefaultStats() {
  return {
    totalDetections: 0,
    totalSkips: 0,
    totalSkippedSeconds: 0,
    lastDetectedCharacter: null,
    lastMatchScore: null,
    lastDetectionAt: null,
    updatedAt: Date.now()
  };
}

function normalizeRuntimeDebug(runtimeDebug) {
  const safe = runtimeDebug || {};
  const defaults = createDefaultRuntimeDebug();

  return {
    mode: typeof safe.mode === "string" ? safe.mode : defaults.mode,
    lastDetectedCharacter:
      typeof safe.lastDetectedCharacter === "string" ? safe.lastDetectedCharacter : null,
    lastMatchScore:
      typeof safe.lastMatchScore === "number" ? safe.lastMatchScore : null,
    lastFaceCount:
      typeof safe.lastFaceCount === "number" ? safe.lastFaceCount : defaults.lastFaceCount,
    lastDetectionTimeMs:
      typeof safe.lastDetectionTimeMs === "number"
        ? safe.lastDetectionTimeMs
        : defaults.lastDetectionTimeMs,
    lastVideoTime:
      typeof safe.lastVideoTime === "number" ? safe.lastVideoTime : defaults.lastVideoTime,
    lastSkipStartedAt:
      typeof safe.lastSkipStartedAt === "number" ? safe.lastSkipStartedAt : null,
    lastSkipEndedAt:
      typeof safe.lastSkipEndedAt === "number" ? safe.lastSkipEndedAt : null,
    lastSkipDuration:
      typeof safe.lastSkipDuration === "number" ? safe.lastSkipDuration : defaults.lastSkipDuration,
    updatedAt: typeof safe.updatedAt === "number" ? safe.updatedAt : Date.now()
  };
}

function normalizeCharacter(character) {
  const safe = character || {};
  const defaultStats = createDefaultCharacterStats();
  const inputStats = safe.stats || {};

  return {
    id: typeof safe.id === "string" ? safe.id : generateId(),
    name: typeof safe.name === "string" ? safe.name : "Unnamed",
    enabled: typeof safe.enabled === "boolean" ? safe.enabled : true,
    // Per-character detection threshold; null means "use global threshold"
    threshold: typeof safe.threshold === "number" ? safe.threshold : null,
    descriptors: Array.isArray(safe.descriptors) ? safe.descriptors : [],
    // was silently DROPPED before — every save wiped the avatar
    thumbnail: typeof safe.thumbnail === "string" ? safe.thumbnail : null,
    stats: {
      detectCount:
        typeof inputStats.detectCount === "number"
          ? inputStats.detectCount
          : defaultStats.detectCount,
      skipCount:
        typeof inputStats.skipCount === "number"
          ? inputStats.skipCount
          : defaultStats.skipCount,
      skippedSeconds:
        typeof inputStats.skippedSeconds === "number"
          ? inputStats.skippedSeconds
          : defaultStats.skippedSeconds,
      lastSeenAt:
        typeof inputStats.lastSeenAt === "number" ? inputStats.lastSeenAt : null,
      lastMatchScore:
        typeof inputStats.lastMatchScore === "number"
          ? inputStats.lastMatchScore
          : null,
      autoLearnCount:
        typeof inputStats.autoLearnCount === "number"
          ? inputStats.autoLearnCount
          : defaultStats.autoLearnCount
    }
  };
}

function normalizeCharacters(characters) {
  if (!Array.isArray(characters)) return [];
  return characters.map(normalizeCharacter);
}

function normalizeStats(stats) {
  const safe = stats || {};
  const defaults = createDefaultStats();

  return {
    totalDetections:
      typeof safe.totalDetections === "number"
        ? safe.totalDetections
        : defaults.totalDetections,
    totalSkips:
      typeof safe.totalSkips === "number" ? safe.totalSkips : defaults.totalSkips,
    totalSkippedSeconds:
      typeof safe.totalSkippedSeconds === "number"
        ? safe.totalSkippedSeconds
        : defaults.totalSkippedSeconds,
    lastDetectedCharacter:
      typeof safe.lastDetectedCharacter === "string" ? safe.lastDetectedCharacter : null,
    lastMatchScore:
      typeof safe.lastMatchScore === "number" ? safe.lastMatchScore : null,
    lastDetectionAt:
      typeof safe.lastDetectionAt === "number" ? safe.lastDetectionAt : null,
    updatedAt: typeof safe.updatedAt === "number" ? safe.updatedAt : Date.now()
  };
}

async function getAppData() {
  const data = await chrome.storage.local.get(DEFAULT_SETTINGS);

  return {
    extensionEnabled: !!data.extensionEnabled,
    detectionThreshold:
      typeof data.detectionThreshold === "number" ? data.detectionThreshold : 0.52,
    characters: normalizeCharacters(data.characters),
    debugOverlayEnabled:
      typeof data.debugOverlayEnabled === "boolean" ? data.debugOverlayEnabled : false,
    debugBoxesEnabled:
      typeof data.debugBoxesEnabled === "boolean" ? data.debugBoxesEnabled : false,
    autoLearnEnabled:
      typeof data.autoLearnEnabled === "boolean" ? data.autoLearnEnabled : true,
    maxDescriptorsPerCharacter:
      typeof data.maxDescriptorsPerCharacter === "number"
        ? data.maxDescriptorsPerCharacter
        : 12,
    charactersCollapsed:
      typeof data.charactersCollapsed === "boolean" ? data.charactersCollapsed : false,
    runtimeDebug: normalizeRuntimeDebug(data.runtimeDebug),
    stats: normalizeStats(data.stats)
  };
}

async function saveAppData(data) {
  await chrome.storage.local.set({
    extensionEnabled: !!data.extensionEnabled,
    detectionThreshold:
      typeof data.detectionThreshold === "number" ? data.detectionThreshold : 0.52,
    characters: normalizeCharacters(data.characters),
    debugOverlayEnabled:
      typeof data.debugOverlayEnabled === "boolean" ? data.debugOverlayEnabled : false,
    debugBoxesEnabled:
      typeof data.debugBoxesEnabled === "boolean" ? data.debugBoxesEnabled : false,
    autoLearnEnabled:
      typeof data.autoLearnEnabled === "boolean" ? data.autoLearnEnabled : true,
    maxDescriptorsPerCharacter:
      typeof data.maxDescriptorsPerCharacter === "number"
        ? data.maxDescriptorsPerCharacter
        : 12,
    charactersCollapsed:
      typeof data.charactersCollapsed === "boolean" ? data.charactersCollapsed : false,
    runtimeDebug: normalizeRuntimeDebug(data.runtimeDebug),
    stats: normalizeStats(data.stats)
  });
}

async function getRuntimeDebug() {
  const data = await chrome.storage.local.get(["runtimeDebug"]);
  return normalizeRuntimeDebug(data.runtimeDebug);
}

async function setRuntimeDebug(patch) {
  const current = await getRuntimeDebug();
  const next = normalizeRuntimeDebug({
    ...current,
    ...(patch || {}),
    updatedAt: Date.now()
  });

  await chrome.storage.local.set({ runtimeDebug: next });
  return next;
}

async function resetRuntimeDebug() {
  const next = createDefaultRuntimeDebug();
  await chrome.storage.local.set({ runtimeDebug: next });
  return next;
}

async function recordCharacterDetection(characterId, characterName, matchScore) {
  const data = await getAppData();
  const now = Date.now();

  data.stats.totalDetections += 1;
  data.stats.lastDetectedCharacter = characterName || null;
  data.stats.lastMatchScore = typeof matchScore === "number" ? matchScore : null;
  data.stats.lastDetectionAt = now;
  data.stats.updatedAt = now;

  data.characters = data.characters.map((character) => {
    if (character.id !== characterId) return character;

    const nextStats = {
      ...createDefaultCharacterStats(),
      ...(character.stats || {})
    };

    nextStats.detectCount += 1;
    nextStats.lastSeenAt = now;
    nextStats.lastMatchScore =
      typeof matchScore === "number" ? matchScore : nextStats.lastMatchScore;

    return {
      ...character,
      stats: nextStats
    };
  });

  await saveAppData(data);
  return data;
}

async function recordCharacterSkip(characterId, skippedSeconds) {
  const data = await getAppData();
  const safeSkippedSeconds =
    typeof skippedSeconds === "number" && Number.isFinite(skippedSeconds)
      ? Math.max(0, skippedSeconds)
      : 0;

  data.stats.totalSkips += 1;
  data.stats.totalSkippedSeconds += safeSkippedSeconds;
  data.stats.updatedAt = Date.now();

  data.characters = data.characters.map((character) => {
    if (character.id !== characterId) return character;

    const nextStats = {
      ...createDefaultCharacterStats(),
      ...(character.stats || {})
    };

    nextStats.skipCount += 1;
    nextStats.skippedSeconds += safeSkippedSeconds;

    return {
      ...character,
      stats: nextStats
    };
  });

  await saveAppData(data);
  return data;
}

// Smart centroid auto-learn.
//
// Principle: descriptors should function as *stable cluster centroids* that
// each represent a distinct appearance of the character (pose, lighting, angle).
// More auto-learns → more stable centroids → better accuracy.
//
// Three cases on each auto-learn call:
//   1. New descriptor is very close to an existing one (< 0.22):
//      → Merge via exponential weighted average (α=0.25, conservative).
//        The centroid becomes more stable with each observation. ✓
//   2. New descriptor is distinct AND capacity not reached:
//      → Add as a new centroid point (new pose/lighting covered). ✓
//   3. Capacity reached AND new descriptor is distinct:
//      → Find the most redundant pair (closest two existing centroids),
//        merge that pair into one, then add the new distinct descriptor.
//        Net effect: coverage is maintained, redundancy is eliminated. ✓
//
// With all three cases, adding more data always improves the centroid set —
// it never degrades accuracy the way a raw FIFO queue did.
async function tryAutoLearnCharacterDescriptor(characterId, descriptor, distance) {
  const data = await getAppData();

  if (!data.autoLearnEnabled) {
    return data;
  }

  if (!Array.isArray(descriptor) || !descriptor.length) {
    return data;
  }

  const maxDescriptors =
    typeof data.maxDescriptorsPerCharacter === "number"
      ? Math.max(3, data.maxDescriptorsPerCharacter)
      : 12;

  data.characters = data.characters.map((character) => {
    if (character.id !== characterId) return character;

    const descriptors = Array.isArray(character.descriptors)
      ? [...character.descriptors]
      : [];

    const stats = {
      ...createDefaultCharacterStats(),
      ...(character.stats || {})
    };

    // ── Find closest existing descriptor ─────────────────────────────────
    let minDist = Infinity;
    let closestIdx = -1;
    for (let i = 0; i < descriptors.length; i++) {
      const d = _descDist(descriptor, descriptors[i]);
      if (d < minDist) { minDist = d; closestIdx = i; }
    }

    if (closestIdx >= 0 && minDist < 0.22) {
      // Case 1: Very close to existing — refine centroid (EWA, α=0.25)
      // The existing centroid has already "seen" multiple observations, so
      // giving it 75% weight keeps it stable while incorporating new info.
      descriptors[closestIdx] = _descAvg(descriptors[closestIdx], descriptor, 0.25);

    } else if (descriptors.length < maxDescriptors) {
      // Case 2: New appearance — add as a distinct centroid
      descriptors.push(descriptor);

    } else {
      // Case 3: At capacity, new appearance → merge most redundant pair, add new
      let minPairDist = Infinity;
      let pairA = 0, pairB = 1;
      for (let i = 0; i < descriptors.length; i++) {
        for (let j = i + 1; j < descriptors.length; j++) {
          const d = _descDist(descriptors[i], descriptors[j]);
          if (d < minPairDist) { minPairDist = d; pairA = i; pairB = j; }
        }
      }
      // Merge the redundant pair into one centroid (equal weight)
      descriptors[pairA] = _descAvg(descriptors[pairA], descriptors[pairB]);
      descriptors.splice(pairB, 1);
      // Add the new distinct appearance
      descriptors.push(descriptor);
    }

    stats.autoLearnCount += 1;
    stats.lastMatchScore =
      typeof distance === "number" ? distance : stats.lastMatchScore;
    stats.lastSeenAt = Date.now();

    return {
      ...character,
      descriptors,
      stats
    };
  });

  await saveAppData(data);
  return data;
}

async function resetAllStats() {
  const data = await getAppData();

  data.stats = createDefaultStats();
  data.characters = data.characters.map((character) => ({
    ...character,
    stats: createDefaultCharacterStats()
  }));

  await saveAppData(data);
  return data;
}

function generateId() {
  return "char_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
}