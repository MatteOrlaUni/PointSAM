"""
app.py — PointSAM Demo Backend (integrazione PointCloud)
---------------------------------------------------------
Avvio:
    python demo/app.py --las /path/to/nuvola.las \\
                       --sam_checkpoint ./demo/model/sam_vit_h_4b8939.pth \\
                       --ckpt_path ./pretrained/model.safetensors

Flusso all'avvio:
  1. Pipeline PointCloud:
       - Generazione vista nadirale MAGMA dal file .LAS
       - SAM Classic sull'immagine MAGMA → maschere 2D
       - Calcolo centroide per maschera → back-projection → punto 3D
  2. Preparazione tensori PointSAM dalla nuvola del LAS (con normalizzazione)
  3. Avvio server Flask:
       GET  /             → index.html (visualizzazione browser)
       GET  /pointcloud   → nuvola normalizzata (xyz + rgb) per il viewer Three.js
       POST /segment_all  → esegue PointSAM su tutti i tronchi, restituisce tutte le maschere
       POST /clear        → reset stato
       POST /save         → salva risultati su disco

Modifica la logica di rilevamento tronchi in:
    demo/prompt_generator.py
    demo/trunk_detector/las_extractor.py
    demo/trunk_detector/sam_detector.py
"""

import hydra
from omegaconf import OmegaConf
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import torch
import os
import argparse
from safetensors.torch import load_model

from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm

# ---------------------------------------------------------------------------- #
# Pipeline automatica di rilevamento tronchi + generazione prompt
# Modifica la logica di selezione del punto in:  demo/prompt_generator.py
#                                                demo/trunk_detector/
# ---------------------------------------------------------------------------- #
from prompt_generator import (
    run_trunk_detection_pipeline,
    build_pc_tensors,
    project_trunk_point_to_normalized,
)

# ============================================================
# ARGOMENTI
# ============================================================
parser = argparse.ArgumentParser(description="PointSAM Demo — integrazione PointCloud")
parser.add_argument("--host",           type=str, default="localhost")
parser.add_argument("--port",           type=int, default=5000)
parser.add_argument("--las",            type=str, required=True,
                    help="Percorso del file .las di input")
parser.add_argument("--sam_checkpoint", type=str,
                    default="./demo/model/sam_vit_h_4b8939.pth",
                    help="Checkpoint SAM ViT-H per il rilevamento 2D dei tronchi")
parser.add_argument("--config",         type=str, default="large",
                    help="Nome configurazione PointSAM (cartella configs/)")
parser.add_argument("--config_dir",     type=str, default="../configs")
parser.add_argument("--ckpt_path",      type=str,
                    default="./pretrained/model.safetensors",
                    help="Checkpoint PointSAM")
parser.add_argument("--data_dir",       type=str, default="./demo/data",
                    help="Directory per i file intermedi della pipeline PointCloud")
parser.add_argument("--output_dir",     type=str, default="./results")
args = parser.parse_args()

# ============================================================
# STEP 1 — Pipeline PointCloud (pesante, eseguita una sola volta all'avvio)
# ============================================================
print("\n" + "=" * 60)
print("AVVIO PIPELINE TRUNK DETECTION (PointCloud → PointSAM)")
print("=" * 60)

points_original, colors_rgb, trunk_points_original = run_trunk_detection_pipeline(
    las_path=args.las,
    data_dir=args.data_dir,
    sam_checkpoint=args.sam_checkpoint,
)

# ============================================================
# STEP 2 — Preparazione nuvola per PointSAM (normalizzazione)
# ============================================================
print("\n[App] Preparazione tensori PointSAM dalla nuvola LAS...")
pc_xyz, pc_rgb, _shift, _scale = build_pc_tensors(points_original, colors_rgb)

# Pre-calcola tutti i prompt normalizzati
trunk_prompts = [
    project_trunk_point_to_normalized(pt, _shift, _scale)
    for pt in trunk_points_original
]
print(f"[App] {len(trunk_prompts)} prompt pronti per PointSAM.")

# Stato per salvataggio
output_dir = args.output_dir
all_segment_masks = []   # List[np.ndarray] — una maschera per tronco

# ============================================================
# STEP 3 — Caricamento PointSAM
# ============================================================
print("\n[App] Caricamento modello PointSAM...")
with hydra.initialize(args.config_dir, version_base=None):
    cfg = hydra.compose(config_name=args.config)
    OmegaConf.resolve(cfg)

sam_model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
sam_model.apply(replace_with_fused_layernorm)
load_model(sam_model, args.ckpt_path)
sam_model = sam_model.cuda()
sam_model.eval()
print("[App] PointSAM caricato.")

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__, static_folder="static")
CORS(app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/static/<path:path>")
def static_server(path):
    return app.send_static_file(path)


@app.route("/pointcloud")
def pointcloud_endpoint():
    """Restituisce la nuvola normalizzata (coordinate + colori) per il viewer Three.js."""
    xyz_flat = pc_xyz[0].cpu().numpy().flatten().tolist()
    rgb_flat = pc_rgb[0].cpu().numpy().flatten().tolist()
    return jsonify({"xyz": xyz_flat, "rgb": rgb_flat})


@app.route("/segment_all", methods=["POST"])
def segment_all():
    """Esegue PointSAM per tutti i tronchi rilevati e restituisce le maschere.

    Questo endpoint sostituisce il vecchio /segment basato sul click utente.
    Non richiede body JSON: i prompt sono già stati calcolati dalla pipeline
    all'avvio e sono disponibili in trunk_prompts.

    Risposta JSON:
        {
            "num_trunks": int,
            "masks": [ [bool, ...], ... ]   # una lista per tronco
        }
    """
    global all_segment_masks
    all_segment_masks = []

    results = []
    print(f"\n[Segment] Avvio segmentazione PointSAM su {len(trunk_prompts)} tronchi...")

    for i, (prompt_point, prompt_label) in enumerate(trunk_prompts):
        prompt_points = torch.tensor([[prompt_point]], dtype=torch.float32).cuda()  # [1, 1, 3]
        prompt_labels = torch.tensor([[prompt_label]], dtype=torch.long).cuda()      # [1, 1]

        with torch.no_grad():
            logits, scores = sam_model.predict_masks(
                pc_xyz, pc_rgb,
                prompt_points, prompt_labels,
                None,   # nessuna prompt_mask iniziale
                True,   # multimask_output
            )

        best_idx = torch.argmax(scores[0])
        mask = (logits[0][best_idx] > 0).cpu().numpy()
        all_segment_masks.append(mask)

        print(f"[Segment] Tronco {i + 1}/{len(trunk_prompts)} | punti segmentati: {mask.sum()}")
        results.append(mask.tolist())

    return jsonify({"num_trunks": len(results), "masks": results})


@app.route("/prompts")
def get_prompts():
    """Restituisce i punti prompt 3D normalizzati (utile per visualizzarli nel browser)."""
    return jsonify({
        "prompts": [{"point": pt, "label": lbl} for pt, lbl in trunk_prompts]
    })


@app.route("/clear", methods=["POST"])
def clear():
    global all_segment_masks
    all_segment_masks = []
    return jsonify({"status": "cleared"})


@app.route("/save", methods=["POST"])
def save():
    os.makedirs(output_dir, exist_ok=True)
    xyz = pc_xyz[0].cpu().numpy()
    rgb = pc_rgb[0].cpu().numpy()
    if all_segment_masks:
        masks_np = np.stack(all_segment_masks)
        las_name = os.path.splitext(os.path.basename(args.las))[0]
        out_path = os.path.join(output_dir, f"{las_name}_segmented.npy")
        np.save(out_path, {"xyz": xyz, "rgb": rgb, "masks": masks_np})
        print(f"[Save] Risultati salvati in {out_path}")
        return jsonify({"status": "saved", "path": out_path, "num_masks": len(masks_np)})
    else:
        return jsonify({"status": "nothing_to_save"})


if __name__ == "__main__":
    print(f"\n[App] Server Flask avviato su http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
