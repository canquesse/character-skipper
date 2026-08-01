// ── Adaptive per-character matching threshold ─────────────────────────────
// The threshold must sit BETWEEN two measured quantities:
//   • intra: how spread out this person's own descriptors are (p90 pairwise)
//   • inter: how close the NEAREST OTHER PERSON in the same scan gets
//     (the real imposter floor for this exact content)
// A threshold above inter matches strangers (skips everyone); below intra it
// misses the person. Both are measured from the scan itself — no guessing.
//
// Caps are embedding-model aware via descriptor dimension:
//   512-dim  = ArcFace (worker-scaled 0.50) — compressed distance space
//   1024-dim = faceres (scale 1.0)
// (_descDist comes from storage.js, loaded before this file.)
function computeAdaptiveThreshold(descriptors, otherClusters) {
  if (!Array.isArray(descriptors) || !descriptors.length) return null;

  const dim   = descriptors[0].length;
  const isArc = dim <= 640; // 512-dim ArcFace vs 1024-dim faceres
  const FLOOR = isArc ? 0.32 : 0.40;
  const CAP   = isArc ? 0.55 : 0.72;

  // intra spread (p90 of pairwise distances) — needs ≥3 descriptors
  let intraP90 = null;
  if (descriptors.length >= 3) {
    const dists = [];
    for (let i = 0; i < descriptors.length; i++) {
      for (let j = i + 1; j < descriptors.length; j++) {
        dists.push(_descDist(descriptors[i], descriptors[j]));
      }
    }
    dists.sort((a, b) => a - b);
    intraP90 = dists[Math.min(dists.length - 1, Math.floor(dists.length * 0.9))];
  }

  // imposter floor: nearest other person in the scan (member-level min)
  let interMin = null;
  for (const other of (otherClusters || [])) {
    if (!Array.isArray(other?.descriptors)) continue;
    for (const od of other.descriptors) {
      if (!od || od.length !== dim) continue;
      for (const d of descriptors) {
        const dist = _descDist(d, od);
        if (interMin === null || dist < interMin) interMin = dist;
      }
    }
  }

  let thr;
  if (interMin !== null && intraP90 !== null) {
    // Midpoint of the identity margin, hard-capped below the imposter floor
    thr = Math.min((intraP90 + interMin) / 2, interMin * 0.80, intraP90 * 1.15);
  } else if (interMin !== null) {
    thr = interMin * 0.75; // no intra data — stay well under the floor
  } else if (intraP90 !== null) {
    thr = intraP90 * 1.15; // no imposter data (single-person scan)
  } else {
    return null; // too little data — fall back to the global slider
  }

  return Math.min(CAP, Math.max(FLOOR, thr));
}

// ── Representative descriptor selection ───────────────────────────────────
// OLD BUG: the save flow split the descriptor budget EQUALLY across selected
// clusters — selecting the main ×87 cluster plus 15 hard-pose ×1 fragments
// saved ~1 frontal descriptor and ~15 weird-pose ones, so the character
// failed to match on plain frontal scenes.
// Fix: budget proportional to how often each cluster was SEEN (count), and
// within a cluster pick maximally DIVERSE descriptors (farthest-point
// sampling) instead of the first-k chronological ones.
function selectRepresentativeDescriptors(clusters, budget) {
  const pool = (clusters || []).filter(
    (c) => Array.isArray(c.descriptors) && c.descriptors.length
  );
  if (!pool.length) return [];

  const weightOf   = (c) => Math.max(1, c.count || c.descriptors.length);
  const totalWeight = pool.reduce((s, c) => s + weightOf(c), 0);

  // Proportional allocation (min 1 each), then fix rounding drift.
  const alloc = pool.map((c) =>
    Math.max(1, Math.min(c.descriptors.length,
      Math.floor(budget * weightOf(c) / totalWeight)))
  );
  const bySizeDesc = pool.map((_, i) => i).sort((a, b) => weightOf(pool[b]) - weightOf(pool[a]));

  let used = alloc.reduce((s, a) => s + a, 0);
  // Too many → trim from the smallest clusters (keep their minimum of 1)
  for (let k = bySizeDesc.length - 1; used > budget && k >= 0; ) {
    const i = bySizeDesc[k];
    if (alloc[i] > 1) { alloc[i]--; used--; } else { k--; }
  }
  // Too few → top up the biggest clusters while they have capacity
  let progressed = true;
  while (used < budget && progressed) {
    progressed = false;
    for (const i of bySizeDesc) {
      if (used >= budget) break;
      if (alloc[i] < pool[i].descriptors.length) { alloc[i]++; used++; progressed = true; }
    }
  }

  // Farthest-point sampling inside each cluster → maximum pose coverage
  const out = [];
  for (let i = 0; i < pool.length; i++) {
    const descs = pool[i].descriptors;
    const n = Math.min(alloc[i], descs.length);
    if (n >= descs.length) { out.push(...descs); continue; }

    const chosenIdx = new Set([0]);
    const chosen    = [descs[0]];
    while (chosen.length < n) {
      let bestJ = -1, bestScore = -1;
      for (let j = 0; j < descs.length; j++) {
        if (chosenIdx.has(j)) continue;
        let minD = Infinity;
        for (const c of chosen) {
          const d = _descDist(descs[j], c);
          if (d < minD) minD = d;
        }
        if (minD > bestScore) { bestScore = minD; bestJ = j; }
      }
      if (bestJ < 0) break;
      chosenIdx.add(bestJ);
      chosen.push(descs[bestJ]);
    }
    out.push(...chosen);
  }

  return out.slice(0, budget);
}

// ── Descriptor consolidation (character profile as a living folder) ────────
// Called when NEW scan descriptors are added to an EXISTING character.
// Keeps the profile bounded and maximally informative over years of use:
//   1. dedupe — descriptors within 0.12 of a kept one add no information,
//      they'd just crowd out rarer poses
//   2. farthest-point sampling down to the budget — keeps the most DIVERSE
//      subset (frontal + profiles + lighting + old video + new video)
function consolidateDescriptors(existing, incoming, budget) {
  const union = [...(existing || []), ...(incoming || [])]
    .filter((d) => Array.isArray(d) && d.length);
  if (!union.length) return [];

  // 1) dedupe near-identicals (greedy, order keeps older descriptors first)
  const DEDUPE_DIST = 0.12;
  const unique = [];
  for (const d of union) {
    let dup = false;
    for (const kept of unique) {
      if (kept.length === d.length && _descDist(d, kept) < DEDUPE_DIST) { dup = true; break; }
    }
    if (!dup) unique.push(d);
  }
  if (unique.length <= budget) return unique;

  // 2) farthest-point sampling — maximum pose/appearance coverage
  const chosen = [unique[0]];
  const chosenIdx = new Set([0]);
  while (chosen.length < budget) {
    let bestJ = -1, bestScore = -1;
    for (let j = 0; j < unique.length; j++) {
      if (chosenIdx.has(j)) continue;
      let minD = Infinity;
      for (const c of chosen) {
        const d = _descDist(unique[j], c);
        if (d < minD) minD = d;
      }
      if (minD > bestScore) { bestScore = minD; bestJ = j; }
    }
    if (bestJ < 0) break;
    chosenIdx.add(bestJ);
    chosen.push(unique[bestJ]);
  }
  return chosen;
}

// ── i18n ──────────────────────────────────────────────────────────────────────
const TRANSLATIONS = {
  en: {
    engineLoading: "Loading face recognition engine…",
    engineReady: "Engine ready! ✓",
    brandSub: "YouTube · AI Face Detection",
    detectionThreshold: "Detection Threshold",
    thresholdHint: "Lower = more aggressive match \u00b7 Higher = stricter match",
    library: "Library",
    libraryHint: "Activate saved characters here — no need to re-scan every video.",
    libraryEmpty: "No characters saved yet.",
    addCharacter: "Add Character",
    autoScanHint: "The extension scans the entire video and detects all characters.",
    scanAll: "Scan Entire Video",
    scanning: "Scanning…",
    scanProgressHint: "Scan in progress — video will return to original position when done.",
    detectedSuffix: "detected — click one to skip",
    selectedCharName: "Selected character's name…",
    save: "Save",
    mergeAndSave: "Merge & Save",
    autoLearnDesc: "Refine characters automatically while watching",
    saveTargetNew: "➕ Save as new character",
    addToCharacter: "Add to Character",
    addedToExisting: (name, n) => `Added to "${name}" — profile now holds ${n} descriptors`,
    orSingleFrame: "or scan single frame",
    scanCurrentFrame: "Scan Current Frame",
    rescanning: "Rescan",
    selected: "selected",
    clearSelection: "Clear selection",
    characterName: "Character name…",
    orUploadPhoto: "or upload photo",
    dragOrSelect: "Drag or select photo",
    maxFileSize: "JPEG \u00b7 PNG \u00b7 WebP \u00b7 max 5 MB",
    saveCharacter: "Save Character",
    characters: "Characters",
    autoLearn: "Auto learn",
    maxDescriptors: "Max descriptors / character",
    noCharacters: "No characters added yet.",
    samples: "samples",
    sensitivity: "Sensitivity",
    sensHint: "Low = more matches \u00b7 global = global threshold",
    active: "Active",
    inactive: "Inactive",
    rename: "\u270e Rename",
    disable: "Disable",
    enable: "Enable",
    delete: "Delete",
    newNamePlaceholder: "New name…",
    stats: "Statistics",
    detections: "Detections",
    skips: "Skips",
    timeSaved: "Time saved",
    score: "Score",
    lastDetected: "Last detected",
    detectionTime: "Detection time",
    reset: "Reset",
    debugRefresh: "Refresh Debug",
    characterPhotos: "Character photos",
    close: "Close",
    select: "Select",
    deselect: "Deselect",
    noPhotos: "No photos",
    unnamed: "Unnamed",
    libActive: "active",
    libInactive: "inactive",
    libRename: "Rename",
    libDelete: "Delete",
    libCancel: "Cancel",
    selectCharFirst: "Select a character first.",
    enterName: "Enter a character name.",
    nameTooLong: "Name cannot exceed 50 characters.",
    saving: "Saving…",
    processingPhotos: "Processing photos…",
    processingPhoto: (i, n) => `Processing photo ${i} / ${n}…`,
    noFaceInPhotos: "No face detected in the photos.",
    noFaceShort: "No face detected.",
    savedMsg: (name, n) => `"${name}" saved — ${n} samples.`,
    addedToast: (name) => `"${name}" added — skip active!`,
    saveFailed: "Save failed.",
    scanNoYTTab: "YouTube tab not found.",
    connectionLost: "Connection lost.",
    scanFailed: "Scan failed.",
    tryAgain: "Try Again",
    updateFailed: "Could not update.",
    scanStatusRunning: "Scanning…",
    sandboxLoadingModel: (sec) => `Loading face recognition model… (${sec}s)`,
    sandboxLoadingModelBase: "Loading face recognition model…",
    engineReadyStarting: "Engine ready! Starting scan…",
    extensionLoading: "Loading extension…",
    engineLoadFailed: "Face recognition model failed to load. Refresh with F5.",
    scanStartFailed: "Scan could not be started.",
    noYTTabError: "YouTube video tab not found. Open a YouTube video, refresh with F5, and try again.",
    scanProgressText: (cur, total) => `Scanning: frame ${cur} / ${total}…`,
    scanAlreadyRunning: "Scan already in progress…",
    scanReattach: "Scan in progress…",
    scanInitial: "Starting scan…",
    extensionRestarting: "Starting scan…",
    extensionLoopStopped: "\u26a0 Extension loop stopped. Refresh the YouTube tab with F5.",
    engineLoading2: "\u23f3 Face recognition engine loading… Wait a moment, model files may be downloading.",
    playVideo: "Play the video — faces will be scanned automatically. Wait a few seconds, then check again.",
    framesScannedNoFace: (n) => `${n} frames scanned but no face found. Try again when faces are clearly visible.`,
    facesClusteredFailed: "Faces scanned but couldn't pass clustering threshold. Keep watching the video.",
    scanCurrentFrameHint: "Scan the current frame and select a face.",
    faceFound: "Face found — click to select",
    facesFound: (n) => `${n} faces found — click one to skip`,
    galleryTitle: (i, n) => `Character ${i + 1} — ${n} frames`,
    galleryBtnTitle: (n) => n > 1 ? `View ${n} photos` : "",
    charAriaLabel: (i, n) => `Character ${i + 1}, seen in ${n} frames`,
    faceAriaLabel: (i) => `Face ${i + 1}`,
    detectedCount: (n) => `${n} people detected!`,
    mergeHint: (n) => `${n} different appearances will be merged into one character — better recognition`,
    mergeAndSaveN: (n) => `Merge & Save (${n})`,
    savedDescriptors: (name, saved, total) => `"${name}" saved! (${saved}/${total} descriptors — limit applied)`,
    savedDescriptorsSimple: (name, n) => `"${name}" saved! (${n} descriptors)`,
    renamed: (name) => `Renamed to "${name}".`,
    renameFailed: "Rename failed.",
    deleted: (name) => `"${name}" deleted.`,
    deleteFailed: "Could not delete.",
    statsReset: "Statistics reset.",
    scanErrorStatus: (msg) => msg || "Scan failed.",
    connectionError: "Connection lost.",
    libCardTitle: (name, active) => `${name} (${active ? "active" : "inactive"})`,
  },
  tr: {
    engineLoading: "Yüz tanıma motoru yükleniyor…",
    engineReady: "Motor hazır! ✓",
    brandSub: "YouTube · AI Yüz Tanıma",
    detectionThreshold: "Tespit Eşiği",
    thresholdHint: "Düşük = daha agresif eşleşme \u00b7 Yüksek = daha katı eşleşme",
    library: "Kütüphane",
    libraryHint: "Kayıtlı karakterleri buradan aktif et — her videoda yeniden taramana gerek yok.",
    libraryEmpty: "Henüz karakter kaydedilmedi.",
    addCharacter: "Karakter Ekle",
    autoScanHint: "Uzantı tüm videoyu tarayıp tüm karakterleri tespit eder.",
    scanAll: "Tüm Videoyu Tara",
    scanning: "Taranıyor…",
    scanProgressHint: "Tarama devam ediyor — bitince video eski konumuna döner.",
    detectedSuffix: "kişi tespit edildi — atlamak istediğine tıkla",
    selectedCharName: "Seçilen karakterin adı…",
    save: "Kaydet",
    mergeAndSave: "Birleştir ve Kaydet",
    autoLearnDesc: "İzlerken karakterleri otomatik iyileştir",
    saveTargetNew: "➕ Yeni karakter olarak kaydet",
    addToCharacter: "Karaktere Ekle",
    addedToExisting: (name, n) => `"${name}" profiline eklendi — artık ${n} descriptor var`,
    orSingleFrame: "veya tek kare tara",
    scanCurrentFrame: "Mevcut Kareyi Tara",
    rescanning: "Yeniden Tara",
    selected: "seçildi",
    clearSelection: "Seçimi temizle",
    characterName: "Karakter adı…",
    orUploadPhoto: "veya fotoğraf yükle",
    dragOrSelect: "Fotoğraf sürükle veya seç",
    maxFileSize: "JPEG \u00b7 PNG \u00b7 WebP \u00b7 maks 5 MB",
    saveCharacter: "Karakteri Kaydet",
    characters: "Karakterler",
    autoLearn: "Auto learn",
    maxDescriptors: "Maks descriptor / karakter",
    noCharacters: "Henüz karakter eklenmedi.",
    samples: "örnek",
    sensitivity: "Hassasiyet",
    sensHint: "Düşük = daha fazla eşleşme \u00b7 global = genel eşik",
    active: "Aktif",
    inactive: "Kapalı",
    rename: "\u270e Adı Değiştir",
    disable: "Kapat",
    enable: "Aç",
    delete: "Sil",
    newNamePlaceholder: "Yeni ad…",
    stats: "İstatistikler",
    detections: "Tespit",
    skips: "Atladı",
    timeSaved: "Kazanılan",
    score: "Skor",
    lastDetected: "Son tespit",
    detectionTime: "Tespit zamanı",
    reset: "Sıfırla",
    debugRefresh: "Debug Yenile",
    characterPhotos: "Karakter fotoğrafları",
    close: "Kapat",
    select: "Seç",
    deselect: "Seçimi Kaldır",
    noPhotos: "Fotoğraf yok",
    unnamed: "İsimsiz",
    libActive: "aktif",
    libInactive: "kapalı",
    libRename: "Yeniden Adlandır",
    libDelete: "Sil",
    libCancel: "İptal",
    selectCharFirst: "Önce bir karakter seç.",
    enterName: "Karakter adı gir.",
    nameTooLong: "Ad 50 karakterden uzun olamaz.",
    saving: "Kaydediliyor…",
    processingPhotos: "Fotoğraflar işleniyor…",
    processingPhoto: (i, n) => `Fotoğraf ${i} / ${n} işleniyor…`,
    noFaceInPhotos: "Fotoğraflarda yüz tespit edilemedi.",
    noFaceShort: "Yüz tespit edilemedi.",
    savedMsg: (name, n) => `"${name}" kaydedildi — ${n} örnek.`,
    addedToast: (name) => `"${name}" eklendi — atlama aktif!`,
    saveFailed: "Kaydetme başarısız.",
    scanNoYTTab: "YouTube sekmesi bulunamadı.",
    connectionLost: "Bağlantı kesildi.",
    scanFailed: "Tarama başarısız.",
    tryAgain: "Tekrar Dene",
    updateFailed: "Güncellenemedi.",
    scanStatusRunning: "Tarama devam ediyor…",
    sandboxLoadingModel: (sec) => `Yüz tanıma modeli yükleniyor… (${sec}s)`,
    sandboxLoadingModelBase: "Yüz tanıma modeli yükleniyor…",
    engineReadyStarting: "Motor hazır! Tarama başlatılıyor…",
    extensionLoading: "Uzantı yükleniyor…",
    engineLoadFailed: "Yüz tanıma modeli yüklenemedi. Sayfayı F5 ile yenile.",
    scanStartFailed: "Tarama başlatılamadı.",
    noYTTabError: "YouTube video sekmesi bulunamadı. Bir YouTube videosunu aç, sayfayı F5 ile yenile ve tekrar dene.",
    scanProgressText: (cur, total) => `Tarama: kare ${cur} / ${total}…`,
    scanAlreadyRunning: "Tarama zaten devam ediyor…",
    scanReattach: "Tarama devam ediyor…",
    scanInitial: "Tarama başlatılıyor…",
    extensionRestarting: "Tarama başlatılıyor…",
    extensionLoopStopped: "\u26a0 Uzantı döngüsü çalışmıyor. YouTube sekmesini F5 ile yenile.",
    engineLoading2: "\u23f3 Yüz tanıma motoru yükleniyor… Biraz bekle, model dosyaları indiriliyor olabilir.",
    playVideo: "Videoyu oynat — yüzler otomatik taranacak. Birkaç saniye bekle, sonra tekrar bak.",
    framesScannedNoFace: (n) => `${n} kare tarandı ama yüz bulunamadı. Yüzlerin ekranda net göründüğü bir anda tekrar dene.`,
    facesClusteredFailed: "Yüzler tarandı ama kümeleme eşiğini geçemedi. Video izlemeye devam et.",
    scanCurrentFrameHint: "Mevcut kareyi tarat ve yüzü seç.",
    faceFound: "Yüz bulundu — tıklayarak seç",
    facesFound: (n) => `${n} yüz bulundu — atlamak istediğine tıkla`,
    galleryTitle: (i, n) => `Karakter ${i + 1} — ${n} kare`,
    galleryBtnTitle: (n) => n > 1 ? `${n} fotoğrafı gör` : "",
    charAriaLabel: (i, n) => `Karakter ${i + 1}, ${n} karede görüldü`,
    faceAriaLabel: (i) => `Yüz ${i + 1}`,
    detectedCount: (n) => `${n} kişi tespit edildi!`,
    mergeHint: (n) => `${n} farklı görünüm tek karakterde birleştirilecek — daha iyi tanıma`,
    mergeAndSaveN: (n) => `Birleştir ve Kaydet (${n})`,
    savedDescriptors: (name, saved, total) => `"${name}" kaydedildi! (${saved}/${total} descriptor — limit uygulandı)`,
    savedDescriptorsSimple: (name, n) => `"${name}" kaydedildi! (${n} descriptor)`,
    renamed: (name) => `"${name}" olarak yeniden adlandırıldı.`,
    renameFailed: "Yeniden adlandırılamadı.",
    deleted: (name) => `"${name}" silindi.`,
    deleteFailed: "Silinemedi.",
    statsReset: "İstatistikler sıfırlandı.",
    scanErrorStatus: (msg) => msg || "Tarama başarısız.",
    connectionError: "Bağlantı kesildi.",
    libCardTitle: (name, active) => `${name} (${active ? "aktif" : "kapalı"})`,
  }
};

let currentLang = "en";

function t(key, ...args) {
  const val = (TRANSLATIONS[currentLang] || TRANSLATIONS.en)[key] ?? (TRANSLATIONS.en)[key] ?? key;
  return typeof val === "function" ? val(...args) : val;
}

function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    const attr = el.dataset.i18nAttr;
    const val  = t(key);
    if (attr) el.setAttribute(attr, val);
    else el.textContent = val;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  const extensionToggle = document.getElementById("extensionToggle");
  const thresholdInput = document.getElementById("thresholdInput");
  const thresholdValue = document.getElementById("thresholdValue");

  const statsTotalDetections = document.getElementById("statsTotalDetections");
  const statsTotalSkips = document.getElementById("statsTotalSkips");
  const statsTotalSkippedSeconds = document.getElementById("statsTotalSkippedSeconds");
  const statsLastDetected = document.getElementById("statsLastDetected");
  const statsLastScore = document.getElementById("statsLastScore");
  const statsLastDetectionAt = document.getElementById("statsLastDetectionAt");
  const resetStatsButton = document.getElementById("resetStatsButton");

  const characterNameInput = document.getElementById("characterNameInput");
  const characterImagesInput = document.getElementById("characterImagesInput");
  const addCharacterButton = document.getElementById("addCharacterButton");
  const addCharacterStatus = document.getElementById("addCharacterStatus");

  // ── Auto-scan UI elements ─────────────────────────────────────────────────
  const autoScanButton       = document.getElementById("autoScanButton");
  const autoScanIdle         = document.getElementById("autoScanIdle");
  const autoScanProgress     = document.getElementById("autoScanProgress");
  const autoScanProgressText = document.getElementById("autoScanProgressText");
  const autoScanProgressBar  = document.getElementById("autoScanProgressBar");
  const autoScanResults      = document.getElementById("autoScanResults");
  const autoScanResultCount  = document.getElementById("autoScanResultCount");
  const autoScanGrid         = document.getElementById("autoScanGrid");
  const autoScanRetry        = document.getElementById("autoScanRetry");
  const autoScanSaveRow      = document.getElementById("autoScanSaveRow");
  const autoScanNameInput    = document.getElementById("autoScanNameInput");
  const autoScanSaveButton   = document.getElementById("autoScanSaveButton");
  const autoScanSaveStatus   = document.getElementById("autoScanSaveStatus");

  // Currently selected clusters from auto-scan (multi-select + merge)
  let selectedClusters    = []; // [{descriptors, thumbnail, count}, ...]
  let currentScanClusters = []; // all clusters from last scan render

  // ── Scan Gallery overlay ──────────────────────────────────────────────────
  let galleryOpenCluster = null; // cluster currently shown in gallery

  function openGallery(cluster, clusterIndex) {
    const overlay   = document.getElementById("scanGalleryOverlay");
    const title     = document.getElementById("scanGalleryTitle");
    const grid      = document.getElementById("scanGalleryGrid");
    const toggleBtn = document.getElementById("scanGalleryToggleSelect");
    if (!overlay || !grid) return;

    galleryOpenCluster = cluster;

    // Title
    if (title) title.textContent = t("galleryTitle", clusterIndex, cluster.count);

    // Thumbnails
    grid.innerHTML = "";
    const thumbs = Array.isArray(cluster.thumbnails) && cluster.thumbnails.length
      ? cluster.thumbnails
      : (cluster.thumbnail ? [cluster.thumbnail] : []);

    if (thumbs.length === 0) {
      const empty = document.createElement("div");
      empty.style.cssText = "grid-column:1/-1;text-align:center;color:var(--ink-3);font-size:12px;padding:20px 0;";
      empty.textContent = t("noPhotos");
      grid.appendChild(empty);
    } else {
      thumbs.forEach((src) => {
        const wrap = document.createElement("div");
        wrap.className = "scan-gallery-thumb";
        wrap.setAttribute("role", "listitem");
        const img = document.createElement("img");
        img.src = src;
        img.alt = "";
        wrap.appendChild(img);
        grid.appendChild(wrap);
      });
    }

    // Toggle-select button state
    syncGallerySelectButton();

    overlay.style.display = "flex";
  }

  function closeGallery() {
    const overlay = document.getElementById("scanGalleryOverlay");
    if (overlay) overlay.style.display = "none";
    galleryOpenCluster = null;
  }

  function syncGallerySelectButton() {
    const toggleBtn = document.getElementById("scanGalleryToggleSelect");
    if (!toggleBtn || !galleryOpenCluster) return;
    const isSelected = selectedClusters.includes(galleryOpenCluster);
    toggleBtn.textContent = isSelected ? t("deselect") : t("select");
  }

  // ── Single-frame Scan UI elements ─────────────────────────────────────────
  const scanPanel        = document.getElementById("scanPanel");
  const scanFaces        = document.getElementById("scanFaces");
  const scanEmpty        = document.getElementById("scanEmpty");
  const scanEmptyIcon    = document.getElementById("scanEmptyIcon");
  const scanEmptyText    = document.getElementById("scanEmptyText");
  const scanVideoButton  = document.getElementById("scanVideoButton");
  const scanBtnLabel     = document.getElementById("scanBtnLabel");
  const scanCountRow     = document.getElementById("scanCountRow");
  const scanCountBadge   = document.getElementById("scanCountBadge");
  const clearScanButton  = document.getElementById("clearScanButton");

  // pendingDescriptors / pendingThumbnails from single-frame scan
  let pendingDescriptors = [];
  let pendingThumbnails  = []; // parallel array to pendingDescriptors
  let lastScanFaces      = null;

  const charactersCollapseButton = document.getElementById("charactersCollapseButton");
  const charactersCollapseIcon = document.getElementById("charactersCollapseIcon");
  const charactersSectionBody = document.getElementById("charactersSectionBody");
  const characterList = document.getElementById("characterList");
  const characterCount = document.getElementById("characterCount");
  const autoLearnToggle = document.getElementById("autoLearnToggle");
  const maxDescriptorsInput = document.getElementById("maxDescriptorsInput");
  const maxDescriptorsValue = document.getElementById("maxDescriptorsValue");

  const debugOverlayToggle = document.getElementById("debugOverlayToggle");
  const debugBoxesToggle = document.getElementById("debugBoxesToggle");
  const runtimeMode = document.getElementById("runtimeMode");
  const runtimeDetected = document.getElementById("runtimeDetected");
  const runtimeScore = document.getElementById("runtimeScore");
  const runtimeFaces = document.getElementById("runtimeFaces");
  const runtimeDetectMs = document.getElementById("runtimeDetectMs");
  const runtimeVideoTime = document.getElementById("runtimeVideoTime");
  const refreshDebugButton = document.getElementById("refreshDebugButton");

  // ── Inline status helper ─────────────────────────────────────────────────
  function setStatus(message, isError = false) {
    if (!addCharacterStatus) return;
    addCharacterStatus.textContent = message || "";
    addCharacterStatus.className = isError ? "status-msg is-error" : "status-msg";
  }

  // ── Engine loading bar ────────────────────────────────────────────────────
  let engineLoadBarTimer = null;

  function hideEngineLoadBar() {
    if (engineLoadBarTimer) { clearInterval(engineLoadBarTimer); engineLoadBarTimer = null; }
    const bar = document.getElementById("engineLoadBar");
    if (bar) { bar.style.opacity = "0"; setTimeout(() => { bar.style.display = "none"; bar.style.opacity = ""; }, 300); }
  }

  function startEngineLoadBar(tabId) {
    const bar  = document.getElementById("engineLoadBar");
    const fill = document.getElementById("engineLoadFill");
    const lbl  = document.getElementById("engineLoadLabel");
    if (!bar) return;

    bar.style.display = "block";
    bar.style.opacity = "1";
    if (fill) fill.style.width = "3%";
    if (lbl)  lbl.textContent  = t("engineLoading");

    const start = Date.now();
    if (engineLoadBarTimer) clearInterval(engineLoadBarTimer);

    engineLoadBarTimer = setInterval(async () => {
      const elapsed = Date.now() - start;
      const pct = 85 * (1 - Math.exp(-elapsed / 15000));
      if (fill) fill.style.width = pct.toFixed(1) + "%";

      try {
        const s = await sendTabMessage(tabId, { type: "GET_SCAN_STATUS" });
        if (s?.sandboxReady) {
          clearInterval(engineLoadBarTimer);
          engineLoadBarTimer = null;
          if (fill) fill.style.width = "100%";
          if (lbl)  lbl.textContent  = t("engineReady");
          setTimeout(hideEngineLoadBar, 1200);
        }
      } catch (_) {}
    }, 900);
  }

  // ── Toast notifications ──────────────────────────────────────────────────
  function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => toast.classList.add("toast--visible"));
    });

    setTimeout(() => {
      toast.classList.remove("toast--visible");
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // ── File → data URL conversion ────────────────────────────────────────────
  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Failed to read image file"));
      reader.readAsDataURL(file);
    });
  }

  // ── Input validation ─────────────────────────────────────────────────────
  const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024;
  const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

  function validateFiles(files) {
    for (const file of files) {
      if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
        return `"${file.name}" ${t("fileNotSupported") || (currentLang === "tr" ? "desteklenmiyor (JPEG, PNG, WebP)." : "is not supported (JPEG, PNG, WebP).")}`;
      }
      if (file.size > MAX_FILE_SIZE_BYTES) {
        return `"${file.name}" ${t("fileTooLarge") || (currentLang === "tr" ? "çok büyük (maks 5 MB)." : "is too large (max 5 MB).")}`;
      }
    }
    return null;
  }

  // ── Active tab helper ────────────────────────────────────────────────────
  // Prefers the active YouTube tab; falls back to any YouTube tab.
  async function getActiveTab() {
    // 1. Active tab in the window that opened the popup
    const w1 = await chrome.tabs.query({ active: true, currentWindow: true });
    const yt1 = w1.find((t) => t.id && t.url && t.url.includes("youtube.com"));
    if (yt1) return yt1;

    // 2. Active tab in the last focused window
    const w2 = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    const yt2 = w2.find((t) => t.id && t.url && t.url.includes("youtube.com"));
    if (yt2) return yt2;

    // 3. Any YouTube tab (pick the most recently accessed)
    const allYt = await chrome.tabs.query({ url: "*://*.youtube.com/*" });
    if (allYt.length) return allYt[0];

    // 4. Fallback: whatever active tab we found
    return w1[0] || w2[0] || null;
  }

  function friendlyError(raw) {
    if (
      raw.includes("Could not establish connection") ||
      raw.includes("Receiving end does not exist") ||
      raw.includes("message port closed")
    ) {
      return t("noYTTabError");
    }
    return raw;
  }

  // ── AUTO-SCAN ─────────────────────────────────────────────────────────────

  let scanPollTimer = null;

  function stopScanPoll() {
    if (scanPollTimer) { clearInterval(scanPollTimer); scanPollTimer = null; }
  }

  // Generic helper: send a message to a tab and return a promise
  function sendTabMessage(tabId, message) {
    return new Promise((resolve, reject) => {
      chrome.tabs.sendMessage(tabId, message, (resp) => {
        if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
        else resolve(resp);
      });
    });
  }

  // Inject content scripts programmatically (used when SPA navigation skipped auto-injection)
  async function tryInjectContentScripts(tabId) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["storage.js", "faceManager.js", "content.js"]
      });
      return true;
    } catch (e) {
      console.warn("[POPUP] script injection failed:", e?.message);
      return false;
    }
  }

  // Poll GET_SCAN_STATUS until sandboxReady=true or timeout.
  // Animates the progress bar with an exponential-approach fill (0 → 85%)
  // so the user can see loading progress, then snaps to 100% when ready.
  async function waitForSandboxReady(tabId, maxWaitMs = 60000) {
    const start = Date.now();

    // Drive the bar every 500 ms — the CSS transition (0.4s) smooths each step
    let barTimer = setInterval(() => {
      const elapsed = Date.now() - start;
      const pct = 85 * (1 - Math.exp(-elapsed / 15000)); // asymptote ≈ 85%
      if (autoScanProgressBar) autoScanProgressBar.style.width = `${pct.toFixed(1)}%`;
      const sec = Math.round(elapsed / 1000);
      if (autoScanProgressText)
        autoScanProgressText.textContent = t("sandboxLoadingModel", sec);
    }, 500);

    // Initial paint so bar isn't blank for the first 500 ms
    if (autoScanProgressBar) autoScanProgressBar.style.width = "2%";
    if (autoScanProgressText)
      autoScanProgressText.textContent = t("sandboxLoadingModelBase");

    try {
      let elapsed = 0;
      while (elapsed < maxWaitMs) {
        await new Promise((r) => setTimeout(r, 1000));
        elapsed = Date.now() - start;
        try {
          const s = await sendTabMessage(tabId, { type: "GET_SCAN_STATUS" });
          if (s?.sandboxReady) {
            clearInterval(barTimer);
            barTimer = null;
            if (autoScanProgressBar) autoScanProgressBar.style.width = "100%";
            if (autoScanProgressText)
              autoScanProgressText.textContent = t("engineReadyStarting");
            await new Promise((r) => setTimeout(r, 400)); // let user see 100%
            return true;
          }
        } catch (_) {}
      }
    } finally {
      if (barTimer) { clearInterval(barTimer); barTimer = null; }
    }
    return false;
  }

  function setAutoScanState(state) {
    autoScanIdle.style.display     = state === "idle"     ? "block" : "none";
    autoScanProgress.style.display = state === "scanning" ? "block" : "none";
    autoScanResults.style.display  = state === "results"  ? "block" : "none";
  }

  function resetAutoScan() {
    stopScanPoll();
    selectedClusters    = [];
    currentScanClusters = [];
    if (autoScanGrid) autoScanGrid.innerHTML = "";
    if (autoScanSaveRow) autoScanSaveRow.style.display = "none";
    if (autoScanNameInput) autoScanNameInput.value = "";
    if (autoScanSaveStatus) autoScanSaveStatus.textContent = "";
    setAutoScanState("idle");
  }

  // ── Multi-select + merge helpers ─────────────────────────────────────────

  function computeClusterCentroid(descriptors) {
    if (!descriptors?.length) return null;
    const dim = descriptors[0].length;
    const sum = new Array(dim).fill(0);
    for (const d of descriptors) for (let i = 0; i < dim; i++) sum[i] += d[i];
    return sum.map((v) => v / descriptors.length);
  }

  function euclideanDistArr(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
  }

  /** Add/remove .suggested class on unselected clusters close to any selected cluster */
  function updateSimilarityHints() {
    if (!autoScanGrid) return;
    const items = Array.from(autoScanGrid.querySelectorAll(".scan-face-item"));

    // Compute centroids of selected clusters
    const selectedCentroids = selectedClusters
      .map((c) => computeClusterCentroid(c.descriptors))
      .filter(Boolean);

    items.forEach((el, idx) => {
      const cluster = currentScanClusters[idx];
      if (!cluster) return;

      const isSelected = selectedClusters.includes(cluster);
      if (isSelected) {
        el.classList.remove("suggested");
        return;
      }

      if (!selectedCentroids.length) {
        el.classList.remove("suggested");
        return;
      }

      const c = computeClusterCentroid(cluster.descriptors);
      if (!c) { el.classList.remove("suggested"); return; }

      const closest = Math.min(...selectedCentroids.map((sc) => euclideanDistArr(sc, c)));
      if (closest < 0.62) {
        el.classList.add("suggested");
      } else {
        el.classList.remove("suggested");
      }
    });
  }

  /** Sync save-row visibility, button label, and merge hint */
  function updateAutoScanSaveRow() {
    const n = selectedClusters.length;
    if (n === 0) {
      if (autoScanSaveRow) autoScanSaveRow.style.display = "none";
      updateSimilarityHints();
      return;
    }
    if (autoScanSaveRow) autoScanSaveRow.style.display = "flex";
    if (autoScanSaveButton) {
      autoScanSaveButton.textContent =
        n === 1 ? t("save") : t("mergeAndSaveN", n);
    }
    const mergeHint = document.getElementById("autoScanMergeHint");
    if (mergeHint) {
      mergeHint.style.display = n >= 2 ? "block" : "none";
      mergeHint.textContent = t("mergeHint", n);
    }
    updateSimilarityHints();
    syncGallerySelectButton();
  }

  /** Render detected clusters into the grid */
  function renderAutoScanClusters(clusters, debugInfo = {}) {
    if (!autoScanGrid) return;
    autoScanGrid.innerHTML = "";
    currentScanClusters = clusters; // keep reference for similarity hints

    if (!clusters.length) {
      const msg = document.createElement("div");
      msg.className = "scan-empty-text";

      const { sandboxReady, discoveryAttempts = 0, discoveryFacesFound = 0, loopAlive } = debugInfo;

      if (!loopAlive) {
        msg.textContent = t("extensionLoopStopped");
      } else if (!sandboxReady) {
        msg.textContent = t("engineLoading2");
      } else if (discoveryAttempts === 0) {
        msg.textContent = t("playVideo");
      } else if (discoveryFacesFound === 0) {
        msg.textContent = t("framesScannedNoFace", discoveryAttempts);
      } else {
        msg.textContent = t("facesClusteredFailed");
      }

      autoScanGrid.appendChild(msg);
      return;
    }

    if (autoScanResultCount) {
      autoScanResultCount.textContent = String(clusters.length);
    }

    clusters.forEach((cluster, i) => {
      const item = document.createElement("div");
      item.className = "scan-face-item";
      item.setAttribute("role", "listitem");
      item.setAttribute("tabindex", "0");
      item.setAttribute("aria-label", t("charAriaLabel", i, cluster.count));

      // Thumbnail
      if (cluster.thumbnail) {
        const img = document.createElement("img");
        img.src = cluster.thumbnail;
        img.alt = t("charAriaLabel", i, cluster.count);
        item.appendChild(img);
      } else {
        // Fallback: show count
        const placeholder = document.createElement("div");
        placeholder.style.cssText = "width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:24px;color:#aaa;";
        placeholder.textContent = "👤";
        item.appendChild(placeholder);
      }

      // Count badge — clickable if cluster has >1 thumbnail
      const thumbs = Array.isArray(cluster.thumbnails) ? cluster.thumbnails : (cluster.thumbnail ? [cluster.thumbnail] : []);
      const badge = document.createElement("button");
      badge.className = "scan-face-count-btn";
      badge.type = "button";
      badge.textContent = `×${cluster.count}`;
      badge.title = t("galleryBtnTitle", thumbs.length);
      badge.addEventListener("click", (e) => {
        e.stopPropagation(); // don't also select/deselect the card
        openGallery(cluster, i);
      });
      item.appendChild(badge);

      // Checkmark
      const check = document.createElement("div");
      check.className = "scan-face-check";
      check.textContent = "✓";
      item.appendChild(check);

      item.addEventListener("click", () => {
        const idx = selectedClusters.indexOf(cluster);
        if (idx >= 0) {
          // Deselect
          selectedClusters.splice(idx, 1);
          item.classList.remove("selected");
        } else {
          // Select
          selectedClusters.push(cluster);
          item.classList.add("selected");
          if (selectedClusters.length === 1 && autoScanNameInput) autoScanNameInput.focus();
        }
        updateAutoScanSaveRow();
      });

      item.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); item.click(); }
      });

      autoScanGrid.appendChild(item);
    });
  }

  // Start polling GET_SCAN_STATUS from the given tab and update UI
  function startScanPolling(tabId) {
    stopScanPoll();
    scanPollTimer = setInterval(async () => {
      try {
        const status = await sendTabMessage(tabId, { type: "GET_SCAN_STATUS" });
        if (!status) { stopScanPoll(); showToast(t("connectionLost"), "error"); setAutoScanState("idle"); return; }

        if (status.status === "running") {
          const pct = status.total > 0 ? Math.round((status.progress / status.total) * 100) : 0;
          if (autoScanProgressText) autoScanProgressText.textContent = t("scanProgressText", status.progress, status.total);
          if (autoScanProgressBar) autoScanProgressBar.style.width = `${pct}%`;
          return; // keep polling
        }

        stopScanPoll();

        if (status.status === "error") {
          showToast(status.error || t("scanFailed"), "error");
          setAutoScanState("idle");
          return;
        }

        if (status.status === "done") {
          const clusters = status.clusters || [];
          renderAutoScanClusters(clusters, { sandboxReady: status.sandboxReady });
          setAutoScanState("results");
          if (clusters.length > 0) showToast(t("detectedCount", clusters.length), "success");
          return;
        }

        setAutoScanState("idle");
      } catch (err) {
        stopScanPoll();
        showToast(friendlyError(err?.message || t("scanFailed")), "error");
        setAutoScanState("idle");
      }
    }, 600);
  }

  // silent=true → auto-load on popup open (no new scan, just show existing results)
  async function doAutoScan(silent = false) {
    stopScanPoll();
    const tab = await getActiveTab();
    if (!tab?.id) {
      if (!silent) showToast(t("scanNoYTTab"), "error");
      setAutoScanState("idle");
      return;
    }

    if (silent) {
      // On popup open: show existing scan results if any, or idle state
      try {
        const status = await sendTabMessage(tab.id, { type: "GET_SCAN_STATUS" });
        if (status?.status === "done" && status.clusters?.length > 0) {
          renderAutoScanClusters(status.clusters, { sandboxReady: status.sandboxReady });
          setAutoScanState("results");
          return;
        }
        if (status?.status === "running") {
          // Scan was running before popup closed — re-attach polling
          setAutoScanState("scanning");
          if (autoScanProgressText) autoScanProgressText.textContent = t("scanReattach");
          startScanPolling(tab.id);
          return;
        }
      } catch (_) {}
      setAutoScanState("idle");
      return;
    }

    setAutoScanState("scanning");
    if (autoScanProgressText) autoScanProgressText.textContent = t("scanInitial");
    if (autoScanProgressBar) autoScanProgressBar.style.width = "0%";

    const attemptStartScan = async (withInjection) => {
      if (withInjection) {
        if (autoScanProgressText) autoScanProgressText.textContent = t("extensionLoading");
        const injected = await tryInjectContentScripts(tab.id);
        if (!injected) {
          showToast(t("noYTTabError"), "error");
          setAutoScanState("idle");
          return false;
        }
        if (autoScanProgressText) autoScanProgressText.textContent = t("sandboxLoadingModelBase");
        const ready = await waitForSandboxReady(tab.id, 60000);
        if (!ready) {
          showToast(t("engineLoadFailed"), "error");
          setAutoScanState("idle");
          return false;
        }
        if (autoScanProgressText) autoScanProgressText.textContent = t("extensionRestarting");
      }

      try {
        const startResp = await sendTabMessage(tab.id, { type: "START_SCAN" });
        if (!startResp?.ok) {
          showToast(t("scanStartFailed"), "error");
          setAutoScanState("idle");
          return false;
        }
        if (startResp.alreadyRunning) {
          if (autoScanProgressText) autoScanProgressText.textContent = t("scanAlreadyRunning");
        }
        startScanPolling(tab.id);
        return true;
      } catch (err) {
        return null; // signal: need injection
      }
    };

    const firstTry = await attemptStartScan(false);
    if (firstTry === null) {
      await attemptStartScan(true);
    }
  }

  // ── Save-target selector (new character vs. existing profile) ────────────
  function getScanTargetSelect() {
    return document.getElementById("autoScanTargetSelect");
  }

  function populateScanTargetSelect(characters) {
    const sel = getScanTargetSelect();
    if (!sel) return;
    const prev = sel.value;

    sel.innerHTML = "";
    const optNew = document.createElement("option");
    optNew.value = "";
    optNew.textContent = t("saveTargetNew");
    sel.appendChild(optNew);

    for (const ch of (characters || [])) {
      const opt = document.createElement("option");
      opt.value = ch.id;
      opt.textContent = ch.name || t("unnamed");
      sel.appendChild(opt);
    }
    // keep the user's selection across re-renders when still valid
    if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;

    syncScanTargetUI();
  }

  function syncScanTargetUI() {
    const sel = getScanTargetSelect();
    const isExisting = !!(sel && sel.value);
    if (autoScanNameInput) autoScanNameInput.style.display = isExisting ? "none" : "";
    if (autoScanSaveButton && !autoScanSaveButton.disabled) {
      autoScanSaveButton.textContent = isExisting ? t("addToCharacter") : t("save");
    }
  }

  document.getElementById("autoScanTargetSelect")
    ?.addEventListener("change", syncScanTargetUI);

  async function doAutoScanSave() {
    if (!selectedClusters.length) {
      showToast(t("selectCharFirst"), "error");
      return;
    }

    const targetId = getScanTargetSelect()?.value || "";

    // ── Path A: add selected clusters to an EXISTING character profile ─────
    if (targetId) {
      if (autoScanSaveButton) {
        autoScanSaveButton.disabled = true;
        autoScanSaveButton.innerHTML = `<span class="btn-spinner" aria-hidden="true"></span> ${t("saving")}`;
      }
      try {
        const data = await getAppData();
        const target = data.characters.find((c) => c.id === targetId);
        if (!target) throw new Error("target character not found");

        const maxDesc  = data.maxDescriptorsPerCharacter || 30;
        const incoming = selectRepresentativeDescriptors(selectedClusters, maxDesc);

        // Bounded, diversity-preserving merge — the "living folder" mechanism
        target.descriptors = consolidateDescriptors(target.descriptors, incoming, maxDesc);

        // Re-derive the threshold from the consolidated set + THIS scan's
        // imposters (unselected clusters). Stays bounded by the imposter
        // floor, so widening the profile can't make it match strangers.
        const otherClusters = currentScanClusters.filter(
          (c) => !selectedClusters.includes(c)
        );
        const newThr = computeAdaptiveThreshold(target.descriptors, otherClusters);
        if (typeof newThr === "number") target.threshold = newThr;
        if (!target.thumbnail && selectedClusters[0]?.thumbnail) {
          target.thumbnail = selectedClusters[0].thumbnail;
        }
        console.log("[POPUP] merged into character:", target.name,
          "descriptors:", target.descriptors.length, "threshold:", target.threshold);

        await saveAppData(data);

        showToast(t("addedToExisting", target.name, target.descriptors.length), "success");
        selectedClusters = [];
        if (autoScanSaveRow) autoScanSaveRow.style.display = "none";
        autoScanGrid?.querySelectorAll(".scan-face-item").forEach((el) => {
          el.classList.remove("selected", "suggested");
        });
        await render();
      } catch (error) {
        console.error("[POPUP] add-to-character failed:", error);
        showToast(t("saveFailed"), "error");
      } finally {
        if (autoScanSaveButton) {
          autoScanSaveButton.disabled = false;
          autoScanSaveButton.textContent = t("addToCharacter");
        }
      }
      return;
    }

    // ── Path B: save as a NEW character (original flow) ─────────────────────
    const name = (autoScanNameInput?.value || "").trim();
    if (!name) {
      if (autoScanSaveStatus) {
        autoScanSaveStatus.textContent = t("enterName");
        autoScanSaveStatus.className = "status-msg is-error";
      }
      autoScanNameInput?.focus();
      return;
    }
    if (name.length > 50) {
      if (autoScanSaveStatus) {
        autoScanSaveStatus.textContent = t("nameTooLong");
        autoScanSaveStatus.className = "status-msg is-error";
      }
      return;
    }

    if (autoScanSaveButton) {
      autoScanSaveButton.disabled = true;
      autoScanSaveButton.innerHTML = `<span class="btn-spinner" aria-hidden="true"></span> ${t("saving")}`;
    }
    if (autoScanSaveStatus) autoScanSaveStatus.textContent = "";

    try {
      const data = await getAppData();

      // Budget proportional to cluster size + diverse within-cluster picks —
      // see selectRepresentativeDescriptors for why equal-split was a bug.
      const maxDesc = data.maxDescriptorsPerCharacter || 30;
      const mergedDescriptors = selectRepresentativeDescriptors(selectedClusters, maxDesc);

      // Every scan cluster the user did NOT select = other people in this
      // video = measured imposter floor for the adaptive threshold.
      const otherClusters = currentScanClusters.filter(
        (c) => !selectedClusters.includes(c)
      );
      const adaptiveThr = computeAdaptiveThreshold(mergedDescriptors, otherClusters);
      console.log("[POPUP] character threshold:", adaptiveThr,
        "(selected", selectedClusters.length, "clusters,",
        otherClusters.length, "imposter clusters)");

      const thumbnail = selectedClusters[0]?.thumbnail || null;

      data.characters.push({
        id:          generateId(),
        name,
        enabled:     true,
        threshold:   adaptiveThr,
        descriptors: mergedDescriptors,
        thumbnail,
        stats:       createDefaultCharacterStats()
      });
      await saveAppData(data);

      selectedClusters = [];
      if (autoScanSaveRow) autoScanSaveRow.style.display = "none";
      if (autoScanNameInput) autoScanNameInput.value = "";
      const mergeHint = document.getElementById("autoScanMergeHint");
      if (mergeHint) mergeHint.style.display = "none";

      // Deselect all grid items + clear suggestions
      autoScanGrid?.querySelectorAll(".scan-face-item").forEach((el) => {
        el.classList.remove("selected", "suggested");
      });

      const totalRaw   = selectedClusters.reduce((s, c) => s + c.descriptors.length, 0);
      const savedCount = mergedDescriptors.length;
      const capped     = totalRaw > savedCount;
      if (autoScanSaveStatus) {
        autoScanSaveStatus.textContent = capped
          ? t("savedDescriptors", name, savedCount, totalRaw)
          : t("savedDescriptorsSimple", name, savedCount);
        autoScanSaveStatus.className = "status-msg";
      }
      showToast(t("addedToast", name), "success");
      await render();
    } catch (err) {
      const msg = err?.message || t("saveFailed");
      if (autoScanSaveStatus) {
        autoScanSaveStatus.textContent = msg;
        autoScanSaveStatus.className = "status-msg is-error";
      }
      showToast(msg, "error");
    } finally {
      if (autoScanSaveButton) {
        autoScanSaveButton.disabled = false;
        const n = selectedClusters.length;
        autoScanSaveButton.textContent =
          n >= 2 ? t("mergeAndSaveN", n) : t("save");
      }
    }
  }

  // ── Single-frame scan helpers ─────────────────────────────────────────────

  function cropFaceFromFrame(frameDataUrl, box, frameWidth, frameHeight) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const padding = Math.max(box.width, box.height) * 0.22;
        const sx = Math.max(0, box.x - padding);
        const sy = Math.max(0, box.y - padding);
        const sw = Math.min(frameWidth  - sx, box.width  + padding * 2);
        const sh = Math.min(frameHeight - sy, box.height + padding * 2);
        const size = Math.ceil(Math.max(sw, sh, 48));
        const canvas = document.createElement("canvas");
        canvas.width  = size;
        canvas.height = size;
        canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, size, size);
        resolve(canvas.toDataURL("image/jpeg", 0.75));
      };
      img.onerror = () => resolve(null);
      img.src = frameDataUrl;
    });
  }

  function setScanEmptyText(text, icon = "◎") {
    if (scanEmptyText) scanEmptyText.textContent = text;
    if (scanEmptyIcon) scanEmptyIcon.textContent = icon;
  }

  function updateScanCount() {
    const n = pendingDescriptors.length;
    if (scanCountBadge) scanCountBadge.textContent = String(n);
    if (scanCountRow)   scanCountRow.style.display = n > 0 ? "flex" : "none";
  }

  async function renderScanResults(scanData) {
    if (!scanFaces) return;
    lastScanFaces = scanData;

    const { faces, frameDataUrl, frameWidth, frameHeight } = scanData;

    if (scanEmpty) scanEmpty.style.display = "none";
    scanFaces.style.display = "flex";
    if (scanPanel) scanPanel.classList.add("has-results");

    if (scanBtnLabel) scanBtnLabel.textContent = t("rescanning");

    scanFaces.innerHTML = "";

    const thumbnails = await Promise.all(
      faces.map((f) => cropFaceFromFrame(frameDataUrl, f.box, frameWidth, frameHeight))
    );

    faces.forEach((face, i) => {
      const item = document.createElement("div");
      item.className = "scan-face-item";
      item.setAttribute("role", "listitem");
      item.setAttribute("aria-label", t("faceAriaLabel", i));
      item.setAttribute("tabindex", "0");

      if (thumbnails[i]) {
        const img = document.createElement("img");
        img.src = thumbnails[i];
        img.alt = t("faceAriaLabel", i);
        item.appendChild(img);
      }

      const check = document.createElement("div");
      check.className = "scan-face-check";
      check.setAttribute("aria-hidden", "true");
      check.textContent = "✓";
      item.appendChild(check);

      item.addEventListener("click", () => {
        const isSelected = item.classList.contains("selected");
        if (isSelected) {
          const idx = item.dataset.pendingIdx !== undefined ? Number(item.dataset.pendingIdx) : -1;
          if (idx >= 0) {
            pendingDescriptors.splice(idx, 1);
            pendingThumbnails.splice(idx, 1);
            item.removeAttribute("data-pending-idx");
          }
          item.classList.remove("selected");
          let p = 0;
          scanFaces.querySelectorAll(".scan-face-item.selected").forEach((el) => {
            el.dataset.pendingIdx = String(p++);
          });
        } else {
          const pendingIdx = pendingDescriptors.length;
          pendingDescriptors.push(face.descriptor);
          pendingThumbnails.push(thumbnails[i] || null);
          item.dataset.pendingIdx = String(pendingIdx);
          item.classList.add("selected");
        }
        updateScanCount();
      });

      item.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); item.click(); }
      });

      scanFaces.appendChild(item);
    });

    updateScanCount();
  }

  function resetScanUI() {
    pendingDescriptors = [];
    pendingThumbnails  = [];
    lastScanFaces = null;
    if (scanFaces) { scanFaces.innerHTML = ""; scanFaces.style.display = "none"; }
    if (scanEmpty) scanEmpty.style.display = "flex";
    if (scanPanel) scanPanel.classList.remove("has-results");
    if (scanBtnLabel) scanBtnLabel.textContent = t("scanCurrentFrame");
    setScanEmptyText(t("scanCurrentFrameHint"), "◎");
    updateScanCount();
  }

  function requestScanFromTab() {
    return new Promise(async (resolve, reject) => {
      try {
        const tab = await getActiveTab();
        if (!tab?.id) {
          reject(new Error(t("scanNoYTTab")));
          return;
        }
        chrome.tabs.sendMessage(tab.id, { type: "CAPTURE_FACES_REQUEST" }, (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(response);
          }
        });
      } catch (err) {
        reject(err);
      }
    });
  }

  async function doScan() {
    if (!scanVideoButton) return;
    document.getElementById("_scanSpinner")?.remove();

    scanVideoButton.disabled = true;
    if (scanBtnLabel) scanBtnLabel.textContent = t("scanning");
    scanVideoButton.insertAdjacentHTML(
      "afterbegin",
      '<span class="btn-spinner--capture" id="_scanSpinner" aria-hidden="true"></span>'
    );

    try {
      const response = await requestScanFromTab();
      document.getElementById("_scanSpinner")?.remove();

      if (!response || response.error) {
        const raw = response?.error || t("scanFailed");
        setScanEmptyText(friendlyError(raw), "⚠");
        if (scanEmpty) scanEmpty.style.display = "flex";
        if (scanFaces) scanFaces.style.display = "none";
        if (scanBtnLabel) scanBtnLabel.textContent = t("tryAgain");
        showToast(raw, "error");
        return;
      }

      await renderScanResults(response);
      showToast(
        response.faces.length > 1
          ? t("facesFound", response.faces.length)
          : t("faceFound"),
        "info"
      );
    } catch (err) {
      document.getElementById("_scanSpinner")?.remove();
      const raw = err?.message || t("scanFailed");
      setScanEmptyText(friendlyError(raw), "⚠");
      if (scanEmpty) scanEmpty.style.display = "flex";
      if (scanFaces) scanFaces.style.display = "none";
      if (scanBtnLabel) scanBtnLabel.textContent = t("tryAgain");
      showToast(friendlyError(raw), "error");
    } finally {
      scanVideoButton.disabled = false;
    }
  }

  // ── Formatting helpers ────────────────────────────────────────────────────
  function formatDateTime(timestamp) {
    if (typeof timestamp !== "number") return "-";
    return new Date(timestamp).toLocaleString();
  }

  function formatSeconds(value) {
    const secs = typeof value === "number" ? Math.max(0, value) : 0;
    if (secs < 60) return `${secs.toFixed(1)}s`;
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    return `${m}m ${s}s`;
  }

  // ── Smooth accordion helper ───────────────────────────────────────────────
  function setPanelOpen(bodyEl, open, btnEl) {
    if (!bodyEl) return;
    if (btnEl) btnEl.setAttribute("aria-expanded", open ? "true" : "false");

    if (open) {
      bodyEl.style.display = "block";
      bodyEl.style.overflow = "hidden";
      bodyEl.style.opacity  = "0";
      bodyEl.style.height   = "0";
      const targetH = bodyEl.scrollHeight;
      requestAnimationFrame(() => {
        bodyEl.style.transition = "height 0.42s cubic-bezier(0.22,1,0.36,1), opacity 0.32s ease";
        bodyEl.style.height  = targetH + "px";
        bodyEl.style.opacity = "1";
      });
      setTimeout(() => {
        bodyEl.style.removeProperty("height");
        bodyEl.style.removeProperty("overflow");
        bodyEl.style.removeProperty("opacity");
        bodyEl.style.removeProperty("transition");
      }, 450);
    } else {
      bodyEl.style.height   = bodyEl.scrollHeight + "px";
      bodyEl.style.overflow = "hidden";
      requestAnimationFrame(() => {
        bodyEl.style.transition = "height 0.36s cubic-bezier(0.4,0,0.2,1), opacity 0.26s ease";
        bodyEl.style.height  = "0";
        bodyEl.style.opacity = "0";
      });
      setTimeout(() => {
        bodyEl.style.display = "none";
        bodyEl.style.removeProperty("height");
        bodyEl.style.removeProperty("overflow");
        bodyEl.style.removeProperty("opacity");
        bodyEl.style.removeProperty("transition");
      }, 380);
    }
  }

  // ── Collapse helpers ──────────────────────────────────────────────────────
  function setCharactersCollapsed(collapsed) {
    // Guard: skip animation if the panel is already in the desired state
    const currentlyOpen = charactersCollapseButton?.getAttribute("aria-expanded") === "true";
    const shouldBeOpen  = !collapsed;
    if (currentlyOpen === shouldBeOpen) return;
    setPanelOpen(charactersSectionBody, shouldBeOpen, charactersCollapseButton);
    // chevron rotation is handled by CSS via aria-expanded
    if (charactersCollapseIcon) charactersCollapseIcon.textContent = "▾";
  }

  // ── Descriptor extraction (via content script) ─────────────────────────────
  // The face engine now runs directly in the content script (no iframe sandbox).
  // The popup sends the photo data URL to the content script for extraction.
  async function extractDescriptorViaContentScript(tabId, dataUrl) {
    return new Promise((resolve, reject) => {
      chrome.tabs.sendMessage(tabId, { type: "EXTRACT_DESCRIPTOR", dataUrl }, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!response) {
          reject(new Error("No response from content script"));
          return;
        }
        if (response.error) {
          reject(new Error(response.error));
          return;
        }
        resolve(response.descriptor || null);
      });
    });
  }

  // ── Stats rendering ───────────────────────────────────────────────────────
  function renderStats(stats) {
    const safe = stats || {};
    if (statsTotalDetections) statsTotalDetections.textContent = String(safe.totalDetections || 0);
    if (statsTotalSkips) statsTotalSkips.textContent = String(safe.totalSkips || 0);
    if (statsTotalSkippedSeconds) statsTotalSkippedSeconds.textContent = formatSeconds(safe.totalSkippedSeconds || 0);
    if (statsLastDetected) statsLastDetected.textContent = safe.lastDetectedCharacter || "-";
    if (statsLastScore) statsLastScore.textContent = typeof safe.lastMatchScore === "number" ? safe.lastMatchScore.toFixed(3) : "-";
    if (statsLastDetectionAt) statsLastDetectionAt.textContent = formatDateTime(safe.lastDetectionAt);
  }

  // ── Character card builder ────────────────────────────────────────────────
  function createCharacterCard(character) {
    const descriptorCount = Array.isArray(character.descriptors) ? character.descriptors.length : 0;
    const stats = character.stats || {};

    const card = document.createElement("div");
    card.className = "char-card" + (character.enabled ? "" : " char-card--inactive");
    card.setAttribute("role", "listitem");

    const head = document.createElement("div");
    head.className = "char-head";

    const avatar = document.createElement("div");
    avatar.className = "char-avatar";
    const avatarLetter = document.createElement("span");
    avatarLetter.className = "char-avatar-letter";
    avatarLetter.textContent = (character.name || "?")[0].toUpperCase();
    avatar.appendChild(avatarLetter);

    const info = document.createElement("div");
    info.className = "char-info";

    const nameEl = document.createElement("div");
    nameEl.className = "char-name";
    nameEl.textContent = character.name || t("unnamed");

    const metaEl = document.createElement("div");
    metaEl.className = "char-meta";
    metaEl.textContent = `${descriptorCount} ${t("samples")}`;

    info.appendChild(nameEl);
    info.appendChild(metaEl);

    const statusPill = document.createElement("span");
    statusPill.className = character.enabled
      ? "char-status-pill char-status-pill--active"
      : "char-status-pill char-status-pill--inactive";
    statusPill.textContent = character.enabled ? t("active") : t("inactive");

    head.appendChild(avatar);
    head.appendChild(info);
    head.appendChild(statusPill);
    card.appendChild(head);

    const statsEl = document.createElement("div");
    statsEl.className = "char-stats";

    const statRows = [
      [t("detections"), String(stats.detectCount || 0)],
      [t("skips"), String(stats.skipCount || 0)],
      [t("timeSaved"), formatSeconds(stats.skippedSeconds || 0)],
      [t("score"), typeof stats.lastMatchScore === "number" ? stats.lastMatchScore.toFixed(3) : "—"],
      ["Auto learn", String(stats.autoLearnCount || 0)],
      [t("lastDetected"), formatDateTime(stats.lastSeenAt)]
    ];

    for (const [label, value] of statRows) {
      const row = document.createElement("div");
      row.className = "char-stat-row";
      const strong = document.createElement("strong");
      strong.textContent = label;
      const span = document.createElement("span");
      span.textContent = value;
      row.appendChild(strong);
      row.appendChild(span);
      statsEl.appendChild(row);
    }

    card.appendChild(statsEl);

    const sensField = document.createElement("div");
    sensField.className = "char-sens";

    const sensHead = document.createElement("div");
    sensHead.className = "char-sens-head";

    const sensInputId = `sens-slider-${character.id}`;
    const sensLabelEl = document.createElement("label");
    sensLabelEl.htmlFor = sensInputId;
    sensLabelEl.className = "char-sens-label";
    sensLabelEl.textContent = t("sensitivity");

    const effectiveThreshold = typeof character.threshold === "number" ? character.threshold : null;

    const sensBadge = document.createElement("span");
    sensBadge.className = "value-chip";
    sensBadge.setAttribute("aria-live", "polite");
    sensBadge.textContent = effectiveThreshold !== null ? effectiveThreshold.toFixed(2) : "global";

    sensHead.appendChild(sensLabelEl);
    sensHead.appendChild(sensBadge);
    sensField.appendChild(sensHead);

    const sensHint = document.createElement("div");
    sensHint.className = "hint";
    sensHint.textContent = t("sensHint");
    sensField.appendChild(sensHint);

    const sensSlider = document.createElement("input");
    sensSlider.type = "range";
    sensSlider.className = "slider";
    sensSlider.id = sensInputId;
    sensSlider.min = "0.25";
    sensSlider.max = "0.60";
    sensSlider.step = "0.01";
    sensSlider.value = String(effectiveThreshold !== null ? effectiveThreshold : 0.42);

    sensSlider.addEventListener("input", async () => {
      const val = Number(sensSlider.value);
      sensBadge.textContent = val.toFixed(2);
      try {
        const data = await getAppData();
        data.characters = data.characters.map((c) => {
          if (c.id !== character.id) return c;
          return { ...c, threshold: val };
        });
        await saveAppData(data);
      } catch (err) {
        console.error("[POPUP] sensitivity save failed:", err);
      }
    });

    sensField.appendChild(sensSlider);
    card.appendChild(sensField);

    const actions = document.createElement("div");
    actions.className = "char-actions";

    // ── Rename button ────────────────────────────────────────────────────
    const renameBtn = document.createElement("button");
    renameBtn.className = "btn btn-sm";
    renameBtn.type = "button";
    renameBtn.textContent = t("rename");

    renameBtn.addEventListener("click", () => {
      // Toggle: if a rename row already exists on this card, remove it
      const existing = card.querySelector(".char-rename-row");
      if (existing) { existing.remove(); return; }

      const renameRow = document.createElement("div");
      renameRow.className = "char-rename-row";

      const renameInput = document.createElement("input");
      renameInput.type = "text";
      renameInput.className = "text-field";
      renameInput.value = character.name || "";
      renameInput.maxLength = 50;
      renameInput.autocomplete = "off";
      renameInput.placeholder = t("newNamePlaceholder");

      const saveBtnR = document.createElement("button");
      saveBtnR.className = "btn btn-sm btn-primary";
      saveBtnR.type = "button";
      saveBtnR.textContent = "✓";

      const cancelBtnR = document.createElement("button");
      cancelBtnR.className = "btn btn-sm";
      cancelBtnR.type = "button";
      cancelBtnR.textContent = "✕";

      renameRow.appendChild(renameInput);
      renameRow.appendChild(saveBtnR);
      renameRow.appendChild(cancelBtnR);
      actions.before(renameRow);
      renameInput.focus();
      renameInput.select();

      const doRenameChar = async () => {
        const newName = renameInput.value.trim();
        if (!newName) { renameInput.focus(); return; }
        renameRow.remove();
        try {
          const data = await getAppData();
          data.characters = data.characters.map((c) =>
            c.id !== character.id ? c : { ...c, name: newName }
          );
          await saveAppData(data);
          await render();
          showToast(t("renamed", newName), "success");
        } catch (_) {
          showToast(t("renameFailed"), "error");
        }
      };

      saveBtnR.addEventListener("click", doRenameChar);
      cancelBtnR.addEventListener("click", () => renameRow.remove());
      renameInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter")  doRenameChar();
        if (e.key === "Escape") renameRow.remove();
      });
    });

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "btn btn-sm";
    toggleBtn.type = "button";
    toggleBtn.textContent = character.enabled ? t("disable") : t("enable");

    toggleBtn.addEventListener("click", async () => {
      try {
        const data = await getAppData();
        data.characters = data.characters.map((c) => {
          if (c.id !== character.id) return c;
          return { ...c, enabled: !c.enabled };
        });
        await saveAppData(data);
        await render();
      } catch (err) {
        showToast(t("updateFailed"), "error");
      }
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "btn btn-sm btn-danger";
    deleteBtn.type = "button";
    deleteBtn.textContent = t("delete");

    deleteBtn.addEventListener("click", async () => {
      try {
        const data = await getAppData();
        data.characters = data.characters.filter((c) => c.id !== character.id);
        await saveAppData(data);
        await render();
        showToast(t("deleted", character.name || t("unnamed")), "info");
      } catch (err) {
        showToast(t("deleteFailed"), "error");
      }
    });

    actions.appendChild(renameBtn);
    actions.appendChild(toggleBtn);
    actions.appendChild(deleteBtn);
    card.appendChild(actions);

    return card;
  }

  // ── Library grid ──────────────────────────────────────────────────────────
  function renderLibrary(characters) {
    const grid    = document.getElementById("libraryGrid");
    const countEl = document.getElementById("libraryCount");
    if (!grid) return;

    grid.innerHTML = "";
    const chars = Array.isArray(characters) ? characters : [];
    if (countEl) countEl.textContent = String(chars.length);

    if (!chars.length) {
      const empty = document.createElement("div");
      empty.className = "scan-empty-text";
      empty.style.textAlign = "center";
      empty.style.padding = "8px 0";
      empty.textContent = t("libraryEmpty");
      grid.appendChild(empty);
      return;
    }

    chars.forEach((character) => {
      const card = document.createElement("div");
      card.className = "lib-card" + (character.enabled ? "" : " lib-card--off");
      card.setAttribute("role", "listitem");
      card.setAttribute("tabindex", "0");
      card.title = t("libCardTitle", character.name, character.enabled);

      // Thumbnail circle
      const thumb = document.createElement("div");
      thumb.className = "lib-card-thumb";
      if (character.thumbnail) {
        const img = document.createElement("img");
        img.src = character.thumbnail;
        img.alt = character.name;
        thumb.appendChild(img);
      } else {
        // Letter avatar fallback
        const letter = document.createElement("div");
        letter.className = "lib-card-fallback";
        letter.textContent = (character.name || "?")[0].toUpperCase();
        thumb.appendChild(letter);
      }
      card.appendChild(thumb);

      // Name label
      const name = document.createElement("div");
      name.className = "lib-card-name";
      name.textContent = character.name || t("unnamed");
      card.appendChild(name);

      // Enabled/disabled pill
      const pill = document.createElement("div");
      pill.className = "lib-card-pill " + (character.enabled ? "lib-card-pill--on" : "lib-card-pill--off");
      pill.textContent = character.enabled ? t("active") : t("inactive");
      card.appendChild(pill);

      // ── Action overlay (rename + delete, appears on hover) ──────────────
      const libActions = document.createElement("div");
      libActions.className = "lib-card-actions";

      const libRenameBtn = document.createElement("button");
      libRenameBtn.className = "lib-card-btn";
      libRenameBtn.type = "button";
      libRenameBtn.title = t("libRename");
      libRenameBtn.textContent = "✎";

      libRenameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (card.classList.contains("lib-card--renaming")) return;
        card.classList.add("lib-card--renaming");

        const rnInput = document.createElement("input");
        rnInput.type = "text";
        rnInput.className = "lib-card-rename-input";
        rnInput.value = character.name || "";
        rnInput.maxLength = 50;
        rnInput.autocomplete = "off";

        const rnSave = document.createElement("button");
        rnSave.className = "lib-card-btn";
        rnSave.type = "button";
        rnSave.title = "Kaydet";
        rnSave.textContent = "✓";

        const rnCancel = document.createElement("button");
        rnCancel.className = "lib-card-btn lib-card-btn--danger";
        rnCancel.type = "button";
        rnCancel.title = "İptal";
        rnCancel.textContent = "✕";

        const rnRow = document.createElement("div");
        rnRow.className = "lib-card-rename-row";
        rnRow.appendChild(rnInput);
        rnRow.appendChild(rnSave);
        rnRow.appendChild(rnCancel);
        card.appendChild(rnRow);
        rnInput.focus();
        rnInput.select();

        const doLibRename = async () => {
          const newName = rnInput.value.trim();
          if (!newName) { rnInput.focus(); return; }
          card.classList.remove("lib-card--renaming");
          rnRow.remove();
          try {
            const data = await getAppData();
            data.characters = data.characters.map((c) =>
              c.id !== character.id ? c : { ...c, name: newName }
            );
            await saveAppData(data);
            await render();
            showToast(t("renamed", newName), "success");
          } catch (_) {
            showToast(t("renameFailed"), "error");
          }
        };

        const doLibRenameCancel = () => {
          card.classList.remove("lib-card--renaming");
          rnRow.remove();
        };

        rnSave.addEventListener("click", (e) => { e.stopPropagation(); doLibRename(); });
        rnCancel.addEventListener("click", (e) => { e.stopPropagation(); doLibRenameCancel(); });
        rnInput.addEventListener("click", (e) => e.stopPropagation());
        rnInput.addEventListener("keydown", (e) => {
          e.stopPropagation();
          if (e.key === "Enter")  doLibRename();
          if (e.key === "Escape") doLibRenameCancel();
        });
      });

      const libDeleteBtn = document.createElement("button");
      libDeleteBtn.className = "lib-card-btn lib-card-btn--danger";
      libDeleteBtn.type = "button";
      libDeleteBtn.title = t("libDelete");
      libDeleteBtn.textContent = "✕";

      libDeleteBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          const data = await getAppData();
          data.characters = data.characters.filter((c) => c.id !== character.id);
          await saveAppData(data);
          await render();
          showToast(t("deleted", character.name || t("unnamed")), "info");
        } catch (_) {
          showToast(t("deleteFailed"), "error");
        }
      });

      libActions.appendChild(libRenameBtn);
      libActions.appendChild(libDeleteBtn);
      card.appendChild(libActions);

      // Click → toggle enabled
      const handleToggle = async () => {
        if (card.classList.contains("lib-card--renaming")) return;
        try {
          const data = await getAppData();
          data.characters = data.characters.map((c) =>
            c.id === character.id ? { ...c, enabled: !c.enabled } : c
          );
          await saveAppData(data);
          await render();
        } catch (_) {
          showToast(t("updateFailed"), "error");
        }
      };
      card.addEventListener("click", handleToggle);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleToggle(); }
      });

      grid.appendChild(card);
    });
  }

  function renderCharacters(characters) {
    if (!characterList) return;

    if (!Array.isArray(characters) || !characters.length) {
      characterList.innerHTML = "";
      const empty = document.createElement("div");
      empty.className = "empty-msg";
      empty.textContent = t("noCharacters");
      characterList.appendChild(empty);
      if (characterCount) characterCount.textContent = "0";
      return;
    }

    if (characterCount) characterCount.textContent = String(characters.length);
    characterList.innerHTML = "";
    for (const character of characters) {
      characterList.appendChild(createCharacterCard(character));
    }
  }

  function renderRuntimeDebug(runtimeDebug) {
    if (runtimeMode) runtimeMode.textContent = runtimeDebug.mode || "idle";
    if (runtimeDetected) runtimeDetected.textContent = runtimeDebug.lastDetectedCharacter || "-";
    if (runtimeScore) runtimeScore.textContent = typeof runtimeDebug.lastMatchScore === "number" ? runtimeDebug.lastMatchScore.toFixed(3) : "-";
    if (runtimeFaces) runtimeFaces.textContent = typeof runtimeDebug.lastFaceCount === "number" ? String(runtimeDebug.lastFaceCount) : "0";
    if (runtimeDetectMs) runtimeDetectMs.textContent = typeof runtimeDebug.lastDetectionTimeMs === "number" ? String(runtimeDebug.lastDetectionTimeMs) : "0";
    if (runtimeVideoTime) runtimeVideoTime.textContent = typeof runtimeDebug.lastVideoTime === "number" ? runtimeDebug.lastVideoTime.toFixed(2) + "s" : "0.00s";
  }

  async function render() {
    const data = await getAppData();

    if (extensionToggle) extensionToggle.checked = !!data.extensionEnabled;
    if (thresholdInput) thresholdInput.value = String(data.detectionThreshold);
    if (thresholdValue) thresholdValue.textContent = Number(data.detectionThreshold).toFixed(2);
    if (debugOverlayToggle) debugOverlayToggle.checked = !!data.debugOverlayEnabled;
    if (debugBoxesToggle) debugBoxesToggle.checked = !!data.debugBoxesEnabled;
    if (autoLearnToggle) autoLearnToggle.checked = !!data.autoLearnEnabled;
    if (maxDescriptorsInput) maxDescriptorsInput.value = String(data.maxDescriptorsPerCharacter || 30);
    if (maxDescriptorsValue) maxDescriptorsValue.textContent = String(data.maxDescriptorsPerCharacter || 30);

    setCharactersCollapsed(!!data.charactersCollapsed);
    renderStats(data.stats || {});
    renderRuntimeDebug(data.runtimeDebug || {});
    const chars = Array.isArray(data.characters) ? data.characters : [];
    renderLibrary(chars);
    populateScanTargetSelect(chars);
    renderCharacters(chars);
  }

  // ── Event listeners ───────────────────────────────────────────────────────

  if (extensionToggle) {
    extensionToggle.addEventListener("change", async () => {
      const data = await getAppData();
      data.extensionEnabled = extensionToggle.checked;
      await saveAppData(data);
      await render();
    });
  }

  if (thresholdInput) {
    thresholdInput.addEventListener("input", async () => {
      const value = Number(thresholdInput.value);
      const data = await getAppData();
      data.detectionThreshold = Number.isFinite(value) ? value : 0.52;
      await saveAppData(data);
      if (thresholdValue) thresholdValue.textContent = data.detectionThreshold.toFixed(2);
    });
  }

  if (debugOverlayToggle) {
    debugOverlayToggle.addEventListener("change", async () => {
      const data = await getAppData();
      data.debugOverlayEnabled = debugOverlayToggle.checked;
      await saveAppData(data);
      await render();
    });
  }

  if (debugBoxesToggle) {
    debugBoxesToggle.addEventListener("change", async () => {
      const data = await getAppData();
      data.debugBoxesEnabled = debugBoxesToggle.checked;
      await saveAppData(data);
      await render();
    });
  }

  if (autoLearnToggle) {
    autoLearnToggle.addEventListener("change", async () => {
      const data = await getAppData();
      data.autoLearnEnabled = autoLearnToggle.checked;
      await saveAppData(data);
      await render();
    });
  }

  if (maxDescriptorsInput) {
    maxDescriptorsInput.addEventListener("input", async () => {
      const value = Number(maxDescriptorsInput.value);
      const data = await getAppData();
      data.maxDescriptorsPerCharacter = Number.isFinite(value) ? value : 12;
      await saveAppData(data);
      if (maxDescriptorsValue) maxDescriptorsValue.textContent = String(data.maxDescriptorsPerCharacter);
    });
  }

  if (charactersCollapseButton) {
    charactersCollapseButton.addEventListener("click", async () => {
      const data = await getAppData();
      data.charactersCollapsed = !data.charactersCollapsed;
      await saveAppData(data);
      await render();
    });
  }

  // Library collapse (client-side only — no storage needed)
  const libraryCollapseButton = document.getElementById("libraryCollapseButton");
  const libraryCollapseIcon   = document.getElementById("libraryCollapseIcon");
  const libraryBody           = document.getElementById("libraryBody");
  let libraryCollapsed = false;

  if (libraryCollapseButton) {
    libraryCollapseButton.addEventListener("click", () => {
      libraryCollapsed = !libraryCollapsed;
      setPanelOpen(libraryBody, !libraryCollapsed, libraryCollapseButton);
    });
  }

  // ── Generic panel accordion handlers ────────────────────────────────────
  (function () {
    const panels = [
      { toggleId: "thresholdToggle", bodyId: "thresholdBody" },
      { toggleId: "addCharToggle",   bodyId: "addCharBody"   },
      { toggleId: "statsToggle",     bodyId: "statsBody"     },
      { toggleId: "debugToggle",     bodyId: "debugBody"     },
    ];
    panels.forEach(({ toggleId, bodyId }) => {
      const btn  = document.getElementById(toggleId);
      const body = document.getElementById(bodyId);
      if (!btn || !body) return;
      btn.addEventListener("click", () => {
        const isOpen = btn.getAttribute("aria-expanded") === "true";
        setPanelOpen(body, !isOpen, btn);
      });
    });
  })();

  if (refreshDebugButton) {
    refreshDebugButton.addEventListener("click", async () => { await render(); });
  }

  if (resetStatsButton) {
    resetStatsButton.addEventListener("click", async () => {
      await resetAllStats();
      await render();
      showToast(t("statsReset"), "info");
    });
  }

  // Auto-scan events
  if (autoScanButton) {
    autoScanButton.addEventListener("click", () => doAutoScan(false));
  }

  if (autoScanRetry) {
    autoScanRetry.addEventListener("click", () => {
      resetAutoScan();
      doAutoScan(false);
    });
  }

  // ── Gallery overlay event handlers ─────────────────────────────────────
  const scanGalleryClose = document.getElementById("scanGalleryClose");
  if (scanGalleryClose) {
    scanGalleryClose.addEventListener("click", closeGallery);
  }

  const scanGalleryToggleSelect = document.getElementById("scanGalleryToggleSelect");
  if (scanGalleryToggleSelect) {
    scanGalleryToggleSelect.addEventListener("click", () => {
      if (!galleryOpenCluster) return;
      const idx = selectedClusters.indexOf(galleryOpenCluster);
      if (idx >= 0) {
        selectedClusters.splice(idx, 1);
      } else {
        selectedClusters.push(galleryOpenCluster);
      }
      // Sync the cluster card's selected style in the grid
      const items = autoScanGrid ? autoScanGrid.querySelectorAll(".scan-face-item") : [];
      items.forEach((el, i) => {
        const cluster = currentScanClusters[i];
        if (cluster) el.classList.toggle("selected", selectedClusters.includes(cluster));
      });
      updateAutoScanSaveRow();
      syncGallerySelectButton();
    });
  }

  // Close gallery on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const overlay = document.getElementById("scanGalleryOverlay");
      if (overlay && overlay.style.display !== "none") { closeGallery(); }
    }
  });

  if (autoScanSaveButton) {
    autoScanSaveButton.addEventListener("click", () => doAutoScanSave());
  }

  if (autoScanNameInput) {
    autoScanNameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doAutoScanSave();
    });
  }

  // Single-frame scan events
  if (scanVideoButton) {
    scanVideoButton.addEventListener("click", () => doScan());
  }

  if (clearScanButton) {
    clearScanButton.addEventListener("click", () => resetScanUI());
  }

  // Save from single-frame scan
  if (addCharacterButton) {
    addCharacterButton.addEventListener("click", async () => {
      try {
        const name  = (characterNameInput?.value || "").trim();
        const files = Array.from(characterImagesInput?.files || []);

        if (!name) {
          setStatus(t("enterName"), true);
          characterNameInput?.focus();
          return;
        }
        if (name.length > 50) {
          setStatus(t("nameTooLong"), true);
          return;
        }

        const hasVideo = pendingDescriptors.length > 0;
        const hasFiles = files.length > 0;

        if (!hasVideo && !hasFiles) {
          setStatus(currentLang === "tr" ? "Yukarıdan bir yüz seç veya fotoğraf yükle." : "Select a face above or upload a photo.", true);
          return;
        }

        if (hasFiles) {
          const fileError = validateFiles(files);
          if (fileError) { setStatus(fileError, true); return; }
        }

        addCharacterButton.disabled = true;
        addCharacterButton.innerHTML = `<span class="btn-spinner" aria-hidden="true"></span> ${t("saving")}`;

        const allDescriptors = [...pendingDescriptors];

        if (hasFiles) {
          setStatus(t("processingPhotos"));
          const tab = await getActiveTab();
          if (!tab?.id) throw new Error("No active YouTube tab found.");

          for (let i = 0; i < files.length; i++) {
            setStatus(t("processingPhoto", i + 1, files.length));
            const dataUrl    = await fileToDataUrl(files[i]);
            const descriptor = await extractDescriptorViaContentScript(tab.id, dataUrl);
            if (descriptor) allDescriptors.push(descriptor);
          }
        }

        if (!allDescriptors.length) {
          setStatus(t("noFaceInPhotos"), true);
          showToast(t("noFaceShort"), "error");
          return;
        }

        const data = await getAppData();
        data.characters.push({
          id:          generateId(),
          name,
          enabled:     true,
          threshold:   computeAdaptiveThreshold(allDescriptors),
          descriptors: allDescriptors,
          thumbnail:   pendingThumbnails[0] || null, // first selected face as thumbnail
          stats:       createDefaultCharacterStats()
        });
        await saveAppData(data);

        if (characterNameInput)   characterNameInput.value = "";
        if (characterImagesInput) characterImagesInput.value = "";
        resetScanUI();

        setStatus(t("savedMsg", name, allDescriptors.length));
        showToast(t("addedToast", name), "success");
        await render();
      } catch (error) {
        console.error("[POPUP] add character failed:", error);
        const msg = error?.message || t("saveFailed");
        setStatus(msg, true);
        showToast(msg, "error");
      } finally {
        addCharacterButton.disabled = false;
        addCharacterButton.textContent = t("saveCharacter");
      }
    });
  }

  // ── Storage change listener ───────────────────────────────────────────────
  // Lightweight updates (runtimeDebug/stats) update only their panels.
  // Heavy keys (characters, settings) trigger a full render.
  const storageChangeHandler = async (changes, areaName) => {
    if (areaName !== "local") return;

    // runtimeDebug — update only the debug panel, no DOM rebuild
    if ("runtimeDebug" in changes) {
      const rd = changes.runtimeDebug.newValue;
      if (rd) renderRuntimeDebug(rd);
    }

    // stats — update only the stats panel, no DOM rebuild
    if ("stats" in changes) {
      const st = changes.stats.newValue;
      if (st) renderStats(st);
    }

    // Heavy keys — full render only when meaningful data changes
    const heavyKeys = [
      "characters", "extensionEnabled", "detectionThreshold",
      "debugOverlayEnabled", "debugBoxesEnabled",
      "autoLearnEnabled", "maxDescriptorsPerCharacter", "charactersCollapsed"
    ];
    if (heavyKeys.some((key) => key in changes)) {
      await render();
    }
  };

  chrome.storage.onChanged.addListener(storageChangeHandler);
  window.addEventListener("unload", () => {
    stopScanPoll();
    if (engineLoadBarTimer) { clearInterval(engineLoadBarTimer); engineLoadBarTimer = null; }
    chrome.storage.onChanged.removeListener(storageChangeHandler);
  });

  // ── Language init ─────────────────────────────────────────────────────────
  try {
    const stored = await chrome.storage.local.get("uiLanguage");
    if (stored.uiLanguage === "tr" || stored.uiLanguage === "en") {
      currentLang = stored.uiLanguage;
    }
  } catch (_) {}
  applyTranslations();

  const langBtn = document.getElementById("langToggleBtn");
  if (langBtn) {
    langBtn.textContent = currentLang === "en" ? "TR" : "EN";
    langBtn.addEventListener("click", async () => {
      currentLang = currentLang === "en" ? "tr" : "en";
      langBtn.textContent = currentLang === "en" ? "TR" : "EN";
      try { await chrome.storage.local.set({ uiLanguage: currentLang }); } catch (_) {}
      applyTranslations();
      await render();
      resetScanUI();
    });
  }

  // ── Initial render ────────────────────────────────────────────────────────
  await render();
  resetScanUI();

  // ── Show engine loading bar if face-api models aren't loaded yet ──────────
  setTimeout(async () => {
    try {
      const tab = await getActiveTab();
      if (!tab?.id) return;
      const s = await sendTabMessage(tab.id, { type: "GET_SCAN_STATUS" });
      if (s && !s.sandboxReady) startEngineLoadBar(tab.id);
    } catch (_) {}
  }, 300);

  // Auto-load discovered clusters on popup open — silent, small delay for content script to be ready
  setTimeout(() => doAutoScan(true).catch(() => setAutoScanState("idle")), 400);
});
