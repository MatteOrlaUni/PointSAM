import * as THREE from "three";
import { camera, points, renderer, origin_colors } from "/static/viewer.js";

// Colori fissi per le maschere di ogni tronco (cicla se tronchi > colori)
var MASK_COLORS = [
  [0, 0.4, 1],    // blu
  [1, 0.3, 0],    // arancione
  [0, 0.8, 0.2],  // verde
  [0.9, 0, 0.9],  // viola
  [1, 0.9, 0],    // giallo
  [0, 0.9, 0.9],  // ciano
  [1, 0, 0.3],    // rosso
  [0.5, 1, 0],    // lime
];
var alpha = 0.5;
var current_masks = [];   // Array di Bool[] — una per tronco
var status_el = null;

// ============================================================
// UI
// ============================================================

function bindButtons() {
  document.getElementById("btn-segment").onclick = onSegmentAll;
  document.getElementById("btn-clear").onclick   = onClear;
  document.getElementById("btn-save").onclick    = onSave;
  status_el = document.getElementById("status");
}

function setStatus(msg) {
  if (status_el) status_el.textContent = msg;
  console.log("[UI]", msg);
}

// ============================================================
// SEGMENTAZIONE AUTOMATICA (chiama /segment_all)
// ============================================================

async function onSegmentAll() {
  setStatus("Segmentazione in corso... attendere.");
  document.getElementById("btn-segment").disabled = true;

  try {
    var response = await fetch("/segment_all", { method: "POST" });
    var data = await response.json();
    current_masks = data.masks;

    applyMasksToViewer(current_masks);
    setStatus(`Completato. ${data.num_trunks} tronchi segmentati.`);
  } catch (err) {
    setStatus("Errore durante la segmentazione: " + err);
  } finally {
    document.getElementById("btn-segment").disabled = false;
  }
}

// ============================================================
// VISUALIZZAZIONE MASCHERE
// ============================================================

function applyMasksToViewer(masks) {
  var colors = points.geometry.attributes.color;

  // Reset ai colori originali
  for (var i = 0; i < origin_colors.count; i++) {
    colors.setXYZ(i, origin_colors.getX(i), origin_colors.getY(i), origin_colors.getZ(i));
  }

  // Applica ogni maschera con colore dedicato
  for (var m = 0; m < masks.length; m++) {
    var mask = masks[m];
    var mc = MASK_COLORS[m % MASK_COLORS.length];

    for (var i = 0; i < mask.length; i++) {
      if (mask[i]) {
        var r = origin_colors.getX(i) * (1 - alpha) + mc[0] * alpha;
        var g = origin_colors.getY(i) * (1 - alpha) + mc[1] * alpha;
        var b = origin_colors.getZ(i) * (1 - alpha) + mc[2] * alpha;
        colors.setXYZ(i, r, g, b);
      }
    }
  }

  // Evidenzia i punti prompt (giallo brillante)
  fetch("/prompts").then(r => r.json()).then(data => {
    // I punti prompt sono in spazio normalizzato: trovo il punto più vicino
    var positions = points.geometry.attributes.position;
    for (var pi = 0; pi < data.prompts.length; pi++) {
      var pp = data.prompts[pi].point;
      var bestIdx = -1, bestDist = Infinity;
      for (var i = 0; i < positions.count; i++) {
        var dx = positions.getX(i) - pp[0];
        var dy = positions.getY(i) - pp[1];
        var dz = positions.getZ(i) - pp[2];
        var d = dx*dx + dy*dy + dz*dz;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      if (bestIdx >= 0) {
        colors.setXYZ(bestIdx, 1, 1, 0); // giallo
      }
    }
    colors.needsUpdate = true;
  });

  colors.needsUpdate = true;
}

// ============================================================
// CLEAR e SAVE
// ============================================================

async function onClear() {
  await fetch("/clear", { method: "POST" });
  current_masks = [];

  var colors = points.geometry.attributes.color;
  for (var i = 0; i < origin_colors.count; i++) {
    colors.setXYZ(i, origin_colors.getX(i), origin_colors.getY(i), origin_colors.getZ(i));
  }
  colors.needsUpdate = true;
  setStatus("Pronto.");
}

async function onSave() {
  var resp = await fetch("/save", { method: "POST" });
  var data = await resp.json();
  if (data.status === "saved") {
    setStatus(`Salvato: ${data.path} (${data.num_masks} maschere)`);
  } else {
    setStatus("Nessuna maschera da salvare. Esegui prima la segmentazione.");
  }
}

// ============================================================
// INIT
// ============================================================

function main() {
  bindButtons();
  setStatus("Pronto. Premi 'Segmenta Tronchi' per avviare.");
}

main();
