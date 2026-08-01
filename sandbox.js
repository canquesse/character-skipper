"use strict";
// ── sandbox.js — engine bridge (runs inside the hidden extension iframe) ──
//
// Why this exists:
//   • A content script runs in the PAGE origin (https://www.youtube.com),
//     so it cannot construct a Worker from a chrome-extension:// URL
//     ("cannot be accessed from origin") and blob: workers are blocked by
//     YouTube's strict CSP.
//   • This page IS extension-origin (loaded as a hidden iframe via
//     web_accessible_resources), so it can freely create the detection
//     worker, which importScripts human.js and fetches the models —
//     all same-origin, all under the extension's own CSP.
//
// Message flow (zero-copy pixel transfer end to end):
//   content script ──postMessage──▶ this iframe ──postMessage──▶ worker
//   content script ◀──postMessage── this iframe ◀──postMessage── worker
//
// All bridge messages carry  __ycs: true  so YouTube's own postMessage
// traffic never collides with ours.

let _worker       = null;
let _parentOrigin = "*";   // learned from the first INIT message

function replyToParent(payload, transfer) {
  window.parent.postMessage({ __ycs: true, ...payload }, _parentOrigin, transfer || []);
}

function ensureWorker() {
  if (_worker) return _worker;

  // Same-origin worker (chrome-extension://<id>/detection-worker.js) —
  // this is exactly what the content script itself is NOT allowed to do.
  _worker = new Worker("detection-worker.js");

  _worker.onmessage = (e) => {
    // Forward INIT_OK / INIT_ERROR / DETECT_RESULT straight to the parent.
    replyToParent(e.data);
  };

  _worker.onerror = (e) => {
    replyToParent({ type: "INIT_ERROR", error: "Worker error: " + (e.message || "unknown") });
    try { _worker.terminate(); } catch (_) {}
    _worker = null;
  };

  return _worker;
}

window.addEventListener("message", (event) => {
  if (event.source !== window.parent) return;

  const msg = event.data;
  if (!msg || msg.__ycs !== true) return;

  // ── INIT ────────────────────────────────────────────────────────────────
  if (msg.type === "INIT") {
    if (event.origin) _parentOrigin = event.origin;
    try {
      ensureWorker().postMessage({
        type:         "INIT",
        humanUrl:     chrome.runtime.getURL("libs/human.js"),
        modelBaseUrl: chrome.runtime.getURL("models/"),
        ortUrl:       chrome.runtime.getURL("libs/ort/ort.min.js"),
        ortBaseUrl:   chrome.runtime.getURL("libs/ort/"),
      });
    } catch (err) {
      replyToParent({ type: "INIT_ERROR", error: String(err?.message || err) });
    }
    return;
  }

  // ── DETECT ──────────────────────────────────────────────────────────────
  if (msg.type === "DETECT") {
    if (!_worker) {
      replyToParent({ type: "DETECT_RESULT", id: msg.id, error: "Engine not initialized" });
      return;
    }
    // Transfer the pixel buffer onward — zero-copy through the whole chain.
    _worker.postMessage(
      { type: "DETECT", id: msg.id, pixels: msg.pixels, width: msg.width, height: msg.height },
      [msg.pixels]
    );
    return;
  }
});

console.log("[SANDBOX] bridge ready (extension origin:", location.origin + ")");
