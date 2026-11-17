"""
sam_detector.py
---------------
Adattato da: PointCloud/src/sam_segmentation.py

Applica SAM Classic (SamAutomaticMaskGenerator) all'immagine MAGMA nadirale
e individua i tronchi presenti. Per ogni tronco accettato restituisce:
  - Il centroide 2D del tronco nell'immagine
  - Il punto 3D corrispondente ottenuto per back-projection sulla nuvola

NON esegue misure, confronti o salvataggio di file xyz/ply di tronchi
(quelle sono funzionalità del progetto PointCloud originale non necessarie qui).

La funzione principale `detect_trunks` può essere importata direttamente
da prompt_generator.py.

Fonte originale: Tommaso Ticci, Università di Firenze
"""

import numpy as np
import cv2
import torch
import os
import warnings
from typing import List, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Percorso default del checkpoint SAM ViT-H
# (copiato/linkato dalla directory model/ del progetto PointCloud)
_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "model", "sam_vit_h_4b8939.pth"
)


def _build_pixel_to_points_map(
    points_centered: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    img_h: int,
    img_w: int,
) -> Tuple[dict, np.ndarray]:
    """Costruisce la mappa pixel (u,v) → lista indici punti 3D.

    Adattato direttamente dalla sezione di pre-calcolo in sam_segmentation.py.

    Returns:
        pixel_to_points : dict (u,v) -> [idx_punto, ...]
        points_camera   : np.ndarray [N, 3] punti in spazio camera
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    points_homo = np.hstack([points_centered, np.ones((len(points_centered), 1))])
    points_camera = (extrinsic @ points_homo.T).T[:, :3]

    valid_camera = points_camera[:, 2] > 0
    pts_valid = points_camera[valid_camera]

    x_proj = (pts_valid[:, 0] * fx / pts_valid[:, 2]) + cx
    y_proj = (pts_valid[:, 1] * fy / pts_valid[:, 2]) + cy

    x_int = np.round(x_proj).astype(int)
    y_int = np.round(y_proj).astype(int)
    in_image = (x_int >= 0) & (x_int < img_w) & (y_int >= 0) & (y_int < img_h)
    x_int = x_int[in_image]
    y_int = y_int[in_image]
    indici_originali = np.where(valid_camera)[0][in_image]

    pixel_to_points = {}
    for x, y, idx_point in zip(x_int, y_int, indici_originali):
        key = (x, y)
        if key not in pixel_to_points:
            pixel_to_points[key] = []
        pixel_to_points[key].append(idx_point)

    return pixel_to_points, points_camera


def _centroid_3d_from_mask(
    mask: np.ndarray,
    pixel_to_points: dict,
    points_original: np.ndarray,
) -> np.ndarray | None:
    """Calcola il punto 3D più vicino al centroide 2D della maschera.

    Strategia:
      1. Calcola centroide (u_c, v_c) dei pixel della maschera.
      2. Cerca il pixel mappato più vicino al centroide.
      3. Tra i punti 3D mappati su quel pixel, prende il punto mediano in Z.

    Returns:
        Punto 3D [x, y, z] in coordinate originali (non centrate), o None.
    """
    v_pix, u_pix = np.where(mask > 0)
    if len(u_pix) == 0:
        return None

    u_c = float(u_pix.mean())
    v_c = float(v_pix.mean())

    # Cerca il punto 3D tra i pixel della maschera più vicino al centroide
    best_dist = float("inf")
    best_idx = None

    for u, v in zip(u_pix, v_pix):
        key = (int(u), int(v))
        if key in pixel_to_points:
            dist = (u - u_c) ** 2 + (v - v_c) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = pixel_to_points[key][0]  # primo punto mappato su quel pixel

    if best_idx is None:
        return None

    return points_original[best_idx]  # coordinate originali (non centrate)


def detect_trunks(
    image_np: np.ndarray,
    points_original: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    offset: np.ndarray,
    checkpoint_path: str = _DEFAULT_CHECKPOINT,
    data_dir: str = None,
) -> List[np.ndarray]:
    """Rileva i tronchi nell'immagine MAGMA e restituisce un punto 3D per ciascuno.

    Il punto è scelto come il punto 3D della nuvola più vicino al centroide
    2D della maschera del tronco.

    Args:
        image_np       : Immagine MAGMA [H, W, 3] uint8.
        points_original: Coordinate XYZ originali della nuvola [N, 3].
        extrinsic      : Matrice estrinseca camera [4, 4].
        intrinsic      : Matrice intrinseca camera [3, 3].
        offset         : Offset XYZ usato per centrare la nuvola [3].
        checkpoint_path: Percorso del checkpoint SAM ViT-H.
        data_dir       : Se fornito, salva overlay di debug (mask_check_all.png).

    Returns:
        Lista di array np.ndarray [3] — un punto 3D (coordinate originali)
        per ciascun tronco rilevato, nell'ordine di confidenza decrescente.
    """
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    h, w = image_np.shape[:2]
    points_centered = points_original - offset

    # --- Mappa pixel -> punti 3D ---
    print("[Detector] Costruzione mappa pixel → punti 3D...")
    pixel_to_points, points_camera = _build_pixel_to_points_map(
        points_centered, extrinsic, intrinsic, h, w
    )

    # --- SAM Classic ---
    print("[Detector] Avvio SAM Classic (Automatic Mask Generator)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(device=device)
    except RuntimeError:
        print("[Detector] Fallback su CPU...")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(device="cpu")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=48,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.90,
        min_mask_region_area=1800,
    )
    masks = mask_generator.generate(image_np)
    masks = sorted(masks, key=lambda x: x["predicted_iou"], reverse=True)
    print(f"[Detector] SAM ha prodotto {len(masks)} maschere candidate.")

    # --- Filtri (identici a sam_segmentation.py) ---
    max_area = (h * w) * 0.05
    min_area = (h * w) * 0.002

    overlay = image_np.copy() if data_dir else None
    canvas_occupato = np.zeros((h, w), dtype=bool)
    trunk_points_3d: List[np.ndarray] = []

    for mask_data in masks:
        mask = mask_data["segmentation"]
        area = mask_data["area"]

        if area > max_area or area < min_area:
            continue

        sovrapposizione = np.logical_and(canvas_occupato, mask).sum() / area
        if sovrapposizione > 0.25:
            continue

        v, u = np.where(mask > 0)

        # Raccoglie profondità dei punti 3D associati
        punti_indices, punti_profondita = [], []
        for u_pix, v_pix in zip(u, v):
            key = (int(u_pix), int(v_pix))
            if key in pixel_to_points:
                for idx_punto in pixel_to_points[key]:
                    punti_indices.append(idx_punto)
                    punti_profondita.append(points_camera[idx_punto, 2])

        if len(np.unique(punti_indices)) < 10:
            continue

        punti_profondita = np.array(punti_profondita)
        z_range = np.max(punti_profondita) - np.min(punti_profondita)
        z_std   = np.std(punti_profondita)
        z_media = np.mean(punti_profondita)
        soglia_range = max(0.05, z_media * 0.02)

        if z_range < soglia_range or z_std < 0.02:
            continue

        pixel_valori = image_np[mask]
        if np.mean(pixel_valori) < 80:
            continue

        # --- Maschera accettata: calcola il punto 3D centrale ---
        punto_3d = _centroid_3d_from_mask(mask, pixel_to_points, points_original)
        if punto_3d is None:
            continue

        trunk_points_3d.append(punto_3d)
        canvas_occupato = np.logical_or(canvas_occupato, mask)
        tid = len(trunk_points_3d)
        print(f"[Detector] Tronco {tid} rilevato | area={area:.0f} | punto 3D={np.round(punto_3d, 3)}")

        # Debug overlay
        if overlay is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            color = np.random.randint(0, 255, (3,)).tolist()
            cv2.drawContours(overlay, contours, -1, color, 3)
            cv2.putText(overlay, f"T{tid}",
                        (int(u.mean()), int(v.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Disegna il punto di prompt come cerchio
            u_c = int(u.mean())
            v_c = int(v.mean())
            cv2.circle(overlay, (u_c, v_c), 8, (255, 255, 0), -1)

    if data_dir and overlay is not None:
        cv2.imwrite(
            os.path.join(data_dir, "mask_check_all.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        print(f"[Detector] Overlay salvato in {data_dir}/mask_check_all.png")

    print(f"[Detector] Totale tronchi accettati: {len(trunk_points_3d)}")
    return trunk_points_3d

