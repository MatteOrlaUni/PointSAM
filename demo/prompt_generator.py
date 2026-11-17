"""
prompt_generator.py
-------------------
Pipeline automatica di rilevamento tronchi e generazione prompt 3D per PointSAM.

Questa pipeline sostituisce completamente il click manuale dell'utente sul browser.

FLUSSO:
  1. Lettura del file .LAS e colorazione MAGMA (per quota) → immagine nadirale 2D
     [trunk_detector/las_extractor.py]

  2. SAM Classic applicato all'immagine MAGMA → maschere 2D dei tronchi
     [trunk_detector/sam_detector.py]

  3. Per ogni tronco: calcolo centroide 2D → back-projection → punto 3D
     sulla nuvola originale
     [trunk_detector/sam_detector.py]

  4. I punti 3D sono restituiti ad app.py, che li normalizza e li invia a
     PointSAM uno alla volta (predict_masks per ciascun tronco).

INTEGRAZIONE CON app.py:
  app.py chiama:
    trunk_points_original = run_trunk_detection_pipeline(las_path, data_dir, sam_checkpoint)
    pc_xyz, pc_rgb = load_las_as_tensors(las_path)
  Poi itera su trunk_points_original e per ciascuno chiama:
    prompt_point, prompt_label = project_point_to_normalized(point_3d, pc_xyz_np)
"""

import os
import numpy as np
import torch
from typing import List, Tuple

from trunk_detector.las_extractor import extract_magma_view
from trunk_detector.sam_detector import detect_trunks


# ============================================================
# FUNZIONE PRINCIPALE — chiamata da app.py all'avvio
# ============================================================

def run_trunk_detection_pipeline(
    las_path: str,
    data_dir: str,
    sam_checkpoint: str,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Esegue l'intera pipeline di rilevamento tronchi da un file LAS.

    1. Genera la vista nadirale MAGMA dal file LAS.
    2. Applica SAM Classic all'immagine MAGMA.
    3. Per ogni tronco accettato, calcola il punto 3D corrispondente
       nella nuvola originale (coordinate non normalizzate).

    Args:
        las_path       : Percorso del file .las di input.
        data_dir       : Directory per i file intermedi (immagini, npy, ecc.).
        sam_checkpoint : Percorso del checkpoint SAM ViT-H (.pth).

    Returns:
        points_original : np.ndarray [N, 3] — coordinate XYZ originali del LAS
        colors_rgb      : np.ndarray [N, 3] — colori RGB originali [0, 1]
        trunk_points    : List[np.ndarray]  — un punto 3D [3] per tronco rilevato
                          (coordinate nelle stesse unità del LAS, non normalizzate)
    """
    # Step 1: Generazione vista MAGMA
    print("\n[Pipeline] === Step 1: Generazione vista MAGMA ===")
    extracted = extract_magma_view(las_path, data_dir)

    # Step 2 + 3: SAM 2D → centroidi → punti 3D
    print("\n[Pipeline] === Step 2-3: Rilevamento tronchi con SAM Classic ===")
    trunk_points = detect_trunks(
        image_np=extracted["image_np"],
        points_original=extracted["points"],
        extrinsic=extracted["extrinsic"],
        intrinsic=extracted["intrinsic"],
        offset=extracted["offset"],
        checkpoint_path=sam_checkpoint,
        data_dir=data_dir,
    )

    print(f"\n[Pipeline] Completata. {len(trunk_points)} tronchi trovati.")
    return extracted["points"], extracted["colors_rgb"], trunk_points


# ============================================================
# UTILITY — usate da app.py per preparare i tensori PointSAM
# ============================================================

def normalize_pointcloud(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalizza la nuvola di punti in una sfera unitaria centrata in 0.

    Questa è la stessa normalizzazione applicata originariamente dal
    loader /pointcloud/<path> di app.py sul file .ply.

    Args:
        points: [N, 3] coordinate originali.

    Returns:
        points_norm : [N, 3] coordinate normalizzate in [-1, 1] circa
        shift       : [3] offset di centratura
        scale       : float — fattore di scala
    """
    shift = points.mean(axis=0)
    centered = points - shift
    scale = float(np.linalg.norm(centered, axis=1).max())
    return (centered / scale).astype(np.float32), shift, scale


def build_pc_tensors(
    points: np.ndarray,
    colors_rgb: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, float]:
    """Crea i tensori GPU pronti per PointSAM dalla nuvola del LAS.

    Args:
        points    : [N, 3] coordinate originali.
        colors_rgb: [N, 3] colori [0, 1].

    Returns:
        pc_xyz   : torch.Tensor [1, N, 3] su CUDA — coordinate normalizzate
        pc_rgb   : torch.Tensor [1, N, 3] su CUDA — colori
        shift    : np.ndarray [3]
        scale    : float
    """
    points_norm, shift, scale = normalize_pointcloud(points)
    pc_xyz = torch.from_numpy(points_norm).float().unsqueeze(0).cuda()
    pc_rgb = torch.from_numpy(colors_rgb.astype(np.float32)).unsqueeze(0).cuda()
    return pc_xyz, pc_rgb, shift, scale


def project_trunk_point_to_normalized(
    point_3d_original: np.ndarray,
    shift: np.ndarray,
    scale: float,
) -> Tuple[List[float], int]:
    """Converte un punto 3D (coordinate originali del LAS) nello spazio
    normalizzato di PointSAM, restituendo il prompt pronto.

    Args:
        point_3d_original : [3] coordinate originali (stesse unità del LAS).
        shift             : offset di centratura restituito da build_pc_tensors.
        scale             : fattore di scala restituito da build_pc_tensors.

    Returns:
        prompt_point : List[float] [x, y, z] nello spazio normalizzato
        prompt_label : int — 1 (foreground)
    """
    point_norm = (point_3d_original - shift) / scale
    return point_norm.tolist(), 1  # sempre foreground
