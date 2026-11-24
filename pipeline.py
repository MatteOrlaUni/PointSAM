"""
Pipeline Automatica: Modello 1 (SAM 2D) → Modello 2 (Point-SAM 3D)
====================================================================
Autore: Matteo Orlandi - Tesi triennale, Ingegneria Informatica, Unifi

Flusso di esecuzione:
  1. Caricamento e sfoltimento della nuvola di punti (.las o .ply)
  2. Proiezione nadirale 2D (mappa di calore Z dall'alto)
  3. Modello 1 - SAM 2D individua i tronchi nella vista proiettata
  4. Per ogni tronco vengono estratti 3 punti 3D (estremo A, centro, estremo B)
  5. Modello 2 - Point-SAM produce la maschera densa 3D per ogni terna di prompt
  6. Salvataggio dei risultati in results/<nome_nuvola>/  (sovrascrittura ad ogni run)
"""

import os
import sys
import shutil
import argparse

import numpy as np
import cv2
import torch
import open3d as o3d
import laspy
import hydra
from omegaconf import OmegaConf
from safetensors.torch import load_model
from scipy.spatial import KDTree

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from hydra.core.global_hydra import GlobalHydra


# ==============================================================================
#  CONFIGURAZIONE - Modifica questi valori per personalizzare l'esecuzione
# ==============================================================================

# Soglia massima di punti per la nuvola (<=0 per attivare il calcolo ADATTIVO automatico)
MAX_POINTS = 0

# Dimensione voxel per downsampling geometrico (None per calcolo automatico adattivo)
VOXEL_SIZE = None

# Risoluzione pixel dell'immagine 2D generata dalla proiezione nadirale
RISOLUZIONE_PROIEZIONE = 1024

# Soglia minima di area in percentuale per considerare un tronco (0.4% per scartare frammenti spuri)
MIN_AREA_PERCENTUALE = 0.004

# Soglia massima di area in percentuale (15% per consentire tronchi completi anche in viste ravvicinate)
MAX_AREA_PERCENTUALE = 0.15

# Soglia massima di sovrapposizione consentita tra maschere (NMS: > 0.25 scartata, come PointCloud)
MAX_OVERLAP_SOGLIA = 0.25

# Soglia minima di elongazione via PCA (rapporto asse lungo / asse corto >= 2.0 per escludere macchie tonde)
MIN_ELONGAZIONE = 2.0

# Lunghezza fisica minima stimata in metri (adattata anche per tronchetti di piccola taglia >= 35 cm)
MIN_LUNGHEZZA_METRI = 0.35

# Soglia minima di variazione di quota Z per distinguere tronchi da suolo piatto (come PointCloud)
MIN_Z_STD = 0.012

# Numero di punti prompt estratti per ogni tronco (non modificare)
N_PUNTI_PER_TRONCO = 3

# Checkpoint Modello 1 (SAM 2D ViT-H)
SAM2D_CHECKPOINT = "pretrained/sam_vit_h_4b8939.pth"

# Checkpoint Modello 2 (Point-SAM 3D)
PCSAM_CHECKPOINT = "pretrained/model.safetensors"

# Config Hydra per Point-SAM
PCSAM_CONFIG = "large"

# Cartella di output
RESULTS_DIR = "results"


# ==============================================================================
#  STEP 1 - CARICAMENTO E SFOLTIMENTO DELLA NUVOLA
# ==============================================================================

def carica_nuvola(percorso: str, max_points: int = MAX_POINTS, voxel_size=VOXEL_SIZE):
    """
    Carica una nuvola di punti da file .las o .ply con campionamento adattivo e deterministico.
    Adatta la densità e il numero di punti alle dimensioni reali della scena:
    - Se max_points <= 0 o None: calcola dinamicamente i punti ottimali (1500 pts/m² di estensione XY, tra 60.000 e 120.000 punti).
    - Usa voxel downsampling per garantire omogeneità spaziale ed eliminare i buchi tra punti.
    - Fissa un seed casuale deterministico (seed=42) per garantire risultati identici a parità di dati.
    """
    print(f"\n{'='*60}")
    print("STEP 1 - Caricamento e campionamento adattivo della nuvola")
    print(f"{'='*60}")
    print(f"File: {percorso}")

    np.random.seed(42)
    torch.manual_seed(42)

    ext = os.path.splitext(percorso)[1].lower()

    if ext == ".las":
        las = laspy.read(percorso)
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            r = np.array(las.red, dtype=np.float64)
            g = np.array(las.green, dtype=np.float64)
            b = np.array(las.blue, dtype=np.float64)
            scala = 65535.0 if r.max() > 255 else 255.0
            rgb = np.vstack((r, g, b)).T / scala
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ones((len(xyz), 3)) * 0.5)

    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(percorso)
        if not pcd.has_colors():
            pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)) * 0.5)
    else:
        raise ValueError(f"Formato file non supportato: {ext}. Usa .las o .ply")

    n_originale = len(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    dx, dy, dz = float(extent[0]), float(extent[1]), float(extent[2])
    area_scena = dx * dy

    print(f"Punti caricati: {n_originale:,}")
    print(f"Dimensioni scena: {dx:.2f}m x {dy:.2f}m x {dz:.2f}m (Area piano XY: {area_scena:.1f} m²)")

    # 1. Calcolo adattivo del numero di punti target in base alle dimensioni reali della scena
    if max_points is None or max_points <= 0:
        # Per scene piccole (< 5 m²) mantiene densità fine, per scene grandi adatta proporzionalmente
        if area_scena < 5.0:
            target_pts = int(np.clip(n_originale, 40000, 80000))
        else:
            target_pts = int(np.clip(area_scena * 1500, 60000, 120000))
        print(f"Campionamento ADATTIVO: densità target calibrata -> budget: {target_pts:,} punti")
    else:
        target_pts = max_points
        print(f"Campionamento impostato da utente: {target_pts:,} punti massimi")

    # 2. Voxel Downsampling preliminare geometricamente omogeneo
    if voxel_size is not None and voxel_size > 0:
        v_size = voxel_size
    else:
        # Voxel size stimato per raggiungere la densità target preservando la geometria dei cilindri
        v_size = float(np.clip(np.sqrt(area_scena / max(1, target_pts)), 0.005, 0.035))

    if n_originale > target_pts:
        print(f"Voxel downsampling omogeneo (voxel_size = {v_size*100:.1f} cm)...")
        pcd = pcd.voxel_down_sample(voxel_size=v_size)
        print(f"Punti dopo voxel downsampling: {len(pcd.points):,}")

    # 3. Rifinitura numerica deterministica (con seed fisso per riproducibilità assoluta)
    if len(pcd.points) > target_pts:
        idx = np.random.choice(len(pcd.points), target_pts, replace=False)
        pcd = pcd.select_by_index(idx)

    print(f"Nuvola pronta per elaborazione: {len(pcd.points):,} punti")
    return pcd


# ==============================================================================
#  STEP 2 - PROIEZIONE NADIRALE 2D
# ==============================================================================

def proietta_2d(pcd: o3d.geometry.PointCloud, risoluzione: int = RISOLUZIONE_PROIEZIONE):
    """
    Proietta la nuvola dall'alto (vista nadirale = piano XY) generando:
    - immagine 2D colorata con mappa di calore Z (altezza) su sfondo nero (stile PointCloud)
    - versione RGB per SAM 2D e BGR per OpenCV
    - struttura dati per la ri-proiezione inversa 2D → 3D
    """
    print(f"\n{'='*60}")
    print("STEP 2 - Proiezione nadirale 2D")
    print(f"{'='*60}")

    xyz = np.asarray(pcd.points)
    rgb_norm = np.asarray(pcd.colors)  # [0, 1]

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Normalizzazione robusta con percentili 1% e 99% (come in PointCloud)
    # per evitare che singoli punti spuri schiaccino la scala cromatica
    z_min = float(np.percentile(z, 1))
    z_max = float(np.percentile(z, 99))

    delta_x = x_max - x_min
    delta_y = y_max - y_min

    # Mantieni le proporzioni reali della scena
    if delta_x >= delta_y:
        larghezza = risoluzione
        altezza = max(1, int(risoluzione * delta_y / delta_x))
    else:
        altezza = risoluzione
        larghezza = max(1, int(risoluzione * delta_x / delta_y))

    # Coordinate pixel per ogni punto
    u = ((x - x_min) / delta_x * (larghezza - 1)).astype(np.int32) if delta_x > 0 else np.zeros(len(x), dtype=np.int32)
    v = ((y_max - y) / delta_y * (altezza - 1)).astype(np.int32) if delta_y > 0 else np.zeros(len(y), dtype=np.int32)

    # Mappa di calore Z (MAGMA): suolo/basso scuro, tronchi/alto luminoso
    z_norm_uint8 = (np.clip((z - z_min) / (z_max - z_min + 1e-9), 0, 1) * 255).astype(np.uint8)

    # Sfondo NERO ad alto contrasto (come nel progetto PointCloud)
    immagine_bgr = np.zeros((altezza, larghezza, 3), dtype=np.uint8)
    ordine_z = np.argsort(z_norm_uint8)
    colori_jet = cv2.applyColorMap(z_norm_uint8, cv2.COLORMAP_MAGMA).squeeze()

    # Disegna ogni punto con raggio calibrato per chiudere micro-buchi tra punti vicini
    raggio_punto = max(2, int(risoluzione / 350))
    for idx_pt in ordine_z:
        pt_u = int(u[idx_pt])
        pt_v = int(v[idx_pt])
        colore = (int(colori_jet[idx_pt][0]), int(colori_jet[idx_pt][1]), int(colori_jet[idx_pt][2]))
        if raggio_punto > 1:
            cv2.circle(immagine_bgr, (pt_u, pt_v), raggio_punto, colore, -1)
        else:
            immagine_bgr[pt_v, pt_u] = colore

    # Converti in RGB nativo per SAM (SAM richiede RGB, cv2 produce BGR)
    immagine_rgb = cv2.cvtColor(immagine_bgr, cv2.COLOR_BGR2RGB)

    # KDTree per ritrovare il punto 3D più vicino dato un pixel 2D
    kdtree_2d = KDTree(np.vstack((u, v)).T.astype(np.float32))

    info = {
        "immagine": immagine_bgr,
        "immagine_rgb": immagine_rgb,
        "kdtree_2d": kdtree_2d,
        "xyz_originale": xyz,
        "u": u, "v": v,
        "x_min": x_min, "y_max": y_max,
        "delta_x": delta_x, "delta_y": delta_y,
        "larghezza": larghezza, "altezza": altezza,
    }

    print(f"Immagine 2D generata: {larghezza}x{altezza} px (sfondo nero, mappa MAGMA)")
    return info


# ==============================================================================
#  STEP 3 - MODELLO 1: SAM 2D - RILEVAMENTO TRONCHI
# ==============================================================================

def rileva_tronchi_sam2d(info_proiezione: dict, checkpoint: str = SAM2D_CHECKPOINT):
    """
    Applica SAM 2D (ViT-H Automatic Mask Generator) sull'immagine nadirale RGB.
    Implementa filtri completi per:
    - Ordinamento per confidenza decrescente (predicted_iou)
    - Filtro area min/max (0.4% - 12% dell'immagine totale)
    - NMS / controllo di sovrapposizione (max 25% con maschere già accettate)
    - Filtro colore/luminanza (elimina suolo scuro e sfondo)
    - Filtro geometrico di elongazione PCA (rapporto asse lungo / asse corto >= 2.2)
    - Filtro di lunghezza fisica minima in metri (>= 0.8 metri)
    - Filtro di profondità 3D (z_range >= 0.04m per scartare suolo piatto)
    """
    print(f"\n{'='*60}")
    print("STEP 3 - Modello 1: SAM 2D - Rilevamento tronchi")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint SAM 2D non trovato: {checkpoint}\n"
            f"Scarica sam_vit_h_4b8939.pth da:\n"
            f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
            f"e posizionalo in: pretrained/"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo SAM 2D: {device}")

    sam2d = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam2d.to(device=device)

    immagine_rgb = info_proiezione["immagine_rgb"]
    immagine_bgr = info_proiezione["immagine"]
    xyz_originale = info_proiezione["xyz_originale"]
    kdtree_2d = info_proiezione["kdtree_2d"]
    delta_x = info_proiezione["delta_x"]
    delta_y = info_proiezione["delta_y"]

    h, w = immagine_rgb.shape[:2]
    area_totale = h * w
    scale_x = delta_x / w
    scale_y = delta_y / h

    # Parametri ottimizzati conformi a PointCloud
    generatore = SamAutomaticMaskGenerator(
        model=sam2d,
        points_per_side=48,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.90,
        min_mask_region_area=1800,
    )

    print("Generazione maschere in corso (su immagine RGB, 48x48 griglia)...")
    maschere_raw = generatore.generate(immagine_rgb)
    print(f"Maschere generate dal modello: {len(maschere_raw)}")

    # 1. Ordina le maschere in base alla confidenza decrescente (come in PointCloud)
    maschere_raw.sort(key=lambda m: m["predicted_iou"], reverse=True)

    min_area = area_totale * MIN_AREA_PERCENTUALE
    max_area = area_totale * MAX_AREA_PERCENTUALE

    canvas_occupato = np.zeros((h, w), dtype=bool)
    maschere_filtrate = []

    for m in maschere_raw:
        area = m["area"]

        # 2. Filtro di area calibrato (0.4% - 12% dell'immagine)
        if area < min_area or area > max_area:
            continue

        # 3. Controllo sovrapposizione (NMS): tiene la maschera a confidenza maggiore
        sovrapposizione = np.logical_and(canvas_occupato, m["segmentation"]).sum() / area
        if sovrapposizione > MAX_OVERLAP_SOGLIA:
            continue

        # 4. Filtro di colore/luminanza (i tronchi in MAGMA contengono pixel arancio/giallo luminosi)
        pixel_valori = immagine_bgr[m["segmentation"]]
        if np.mean(pixel_valori) < 30 or np.percentile(pixel_valori, 80) < 50:
            continue

        # 5. Filtro geometrico di forma: un tronco e' un cilindro allungato (PCA su pixel in coordinate metriche)
        pixel_v, pixel_u = np.where(m["segmentation"])
        if len(pixel_u) < 20:
            continue

        pts_m = np.vstack((pixel_u * scale_x, pixel_v * scale_y)).T
        pts_c = pts_m - pts_m.mean(axis=0)
        cov = np.cov(pts_c, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)
        std_long = np.sqrt(max(1e-6, eigvals[1]))
        std_trans = np.sqrt(max(1e-6, eigvals[0]))
        elongazione = std_long / std_trans
        lunghezza_stimata = 3.5 * std_long

        # Scarta macchie tonde (rocce, terreno piatto) o frammenti troppo corti (< 0.8 metri)
        if elongazione < MIN_ELONGAZIONE or lunghezza_stimata < MIN_LUNGHEZZA_METRI:
            continue

        # 6. Filtro di rilievo 3D: campiona punti 3D nella maschera
        if len(pixel_u) > 100:
            sub_idx = np.random.choice(len(pixel_u), 100, replace=False)
            pu_sub, pv_sub = pixel_u[sub_idx], pixel_v[sub_idx]
        else:
            pu_sub, pv_sub = pixel_u, pixel_v

        pts_2d_sub = np.vstack((pu_sub, pv_sub)).T.astype(np.float32)
        _, idx_3d = kdtree_2d.query(pts_2d_sub)
        z_campioni = xyz_originale[idx_3d, 2]

        z_std = float(np.std(z_campioni))
        z_range = float(np.ptp(z_campioni))

        if z_std < MIN_Z_STD or z_range < 0.04:
            continue

        # Maschera confermata
        canvas_occupato = np.logical_or(canvas_occupato, m["segmentation"])
        maschere_filtrate.append(m)
        print(f"  Tronco accettato: area {area} ({area/area_totale*100:.2f}%) | Lunghezza ~{lunghezza_stimata:.2f}m | Elongazione {elongazione:.1f} | IoU SAM {m['predicted_iou']:.3f}")

    print(f"Maschere accettate dopo filtraggio rigoroso: {len(maschere_filtrate)}")

    # Libera la memoria GPU del Modello 1
    del sam2d
    torch.cuda.empty_cache()

    return maschere_filtrate


# ==============================================================================
#  STEP 4 - ESTRAZIONE TERNA DI PUNTI 3D PER TRONCO
# ==============================================================================

def estrai_terna_punti_3d(maschera_binaria: np.ndarray, info_proiezione: dict):
    """
    Data la maschera 2D di un tronco, estrae 3 punti 3D:
      - il punto centrale della maschera
      - i due punti agli estremi del tronco lungo il suo asse principale (PCA)
    Usa percentili (12% e 88%) lungo l'asse per garantire che i prompt cadano
    saldamente sul corpo solido del tronco e non su bordi sfocati o terreno.
    Restituisce: (punti_3d, candidati_2d)
    """
    u_tutti = info_proiezione["u"]
    v_tutti = info_proiezione["v"]
    xyz_originale = info_proiezione["xyz_originale"]
    kdtree_2d = info_proiezione["kdtree_2d"]

    # Coordinate pixel dei punti nella maschera
    pixel_v, pixel_u = np.where(maschera_binaria)
    if len(pixel_u) < 10:
        return None, None

    punti_2d = np.vstack((pixel_u, pixel_v)).T.astype(np.float32)

    # --- Centro della maschera ---
    centro_2d = punti_2d.mean(axis=0)

    # --- Asse principale tramite PCA sui pixel della maschera ---
    punti_centrati = punti_2d - centro_2d
    _, _, Vt = np.linalg.svd(punti_centrati, full_matrices=False)
    asse_principale = Vt[0]  # direzione dell'asse lungo del tronco

    # Proiezione scalare di ogni pixel lungo l'asse
    proiezioni = punti_centrati @ asse_principale

    # Usiamo i percentili (12% e 88%) per evitare il rumore dei bordi estremi
    p12 = np.percentile(proiezioni, 12)
    p88 = np.percentile(proiezioni, 88)
    idx_estremo_a = np.argmin(np.abs(proiezioni - p12))
    idx_estremo_b = np.argmin(np.abs(proiezioni - p88))

    candidati_2d = np.array([
        punti_2d[idx_estremo_a],
        centro_2d,
        punti_2d[idx_estremo_b],
    ])  # shape (3, 2)

    # --- Ri-proiezione inversa: pixel 2D → punto 3D più vicino nel KDTree ---
    punti_3d = []
    for pt_2d in candidati_2d:
        _, idx_vicino = kdtree_2d.query(pt_2d)
        punti_3d.append(xyz_originale[idx_vicino])

    return np.array(punti_3d), candidati_2d


# ==============================================================================
#  STEP 5 - MODELLO 2: POINT-SAM 3D - SEGMENTAZIONE
# ==============================================================================

def carica_pcsam(checkpoint: str = PCSAM_CHECKPOINT, config: str = PCSAM_CONFIG):
    """Carica e restituisce il modello Point-SAM 3D su GPU."""
    print(f"\n{'='*60}")
    print("STEP 5 - Modello 2: Point-SAM 3D - Caricamento")
    print(f"{'='*60}")

    # Pulisce lo stato globale di Hydra per evitare errori in esecuzioni multiple
    GlobalHydra.instance().clear()
    with hydra.initialize("configs", version_base=None):
        cfg = hydra.compose(config_name=config)
        OmegaConf.resolve(cfg)

    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)
    load_model(model, checkpoint)
    model = model.cuda()
    model.eval()
    print("Point-SAM 3D caricato su GPU.")
    return model


def segmenta_tronco(
    pcsam: PointCloudSAM,
    pc_xyz_norm: torch.Tensor,
    pc_rgb: torch.Tensor,
    terna_3d_norm: np.ndarray,
):
    """
    Applica Point-SAM sulla nuvola (già codificata in cache) usando 3 punti prompt.
    Restituisce la maschera booleana 3D del tronco [N] con il punteggio IoU migliore.

    :param pc_xyz_norm: coordinate normalizzate [1, N, 3] su GPU
    :param pc_rgb: colori [1, N, 3] su GPU
    :param terna_3d_norm: array (3, 3) con i 3 punti prompt nelle coordinate
                          normalizzate della nuvola
    """
    prompt_coords = torch.from_numpy(terna_3d_norm).cuda().float().unsqueeze(0)  # [1, 3, 3]
    prompt_labels = torch.ones(1, N_PUNTI_PER_TRONCO, dtype=torch.long).cuda()   # tutti positivi

    with torch.no_grad():
        logits, scores = pcsam.predict_masks(
            pc_xyz_norm, pc_rgb, prompt_coords, prompt_labels
        )

    # Prende la maschera con il punteggio IoU più alto tra le 3 proposte
    best = torch.argmax(scores[0])
    maschera = (logits[0][best] > 0).cpu().numpy()  # [N] bool
    return maschera, scores[0][best].item()


# ==============================================================================
#  OUTPUT - SALVATAGGIO RISULTATI
# ==============================================================================

TAVOLOZZA_COLORI = [
    [1.0, 0.2, 0.2],  # rosso
    [0.2, 0.8, 0.2],  # verde
    [0.2, 0.4, 1.0],  # blu
    [1.0, 0.8, 0.0],  # giallo
    [0.8, 0.2, 1.0],  # viola
    [0.0, 0.9, 0.9],  # ciano
    [1.0, 0.5, 0.0],  # arancione
    [0.5, 1.0, 0.0],  # lime
    [1.0, 0.0, 0.6],  # magenta
    [0.0, 0.5, 1.0],  # azzurro
]


def prepara_cartella_output(nome_nuvola: str) -> str:
    """
    Crea (o sovrascrive) la cartella results/<nome_nuvola>/.
    La sovrascrittura totale è intenzionale: l'esecuzione più recente è sempre valida.
    """
    cartella = os.path.join(RESULTS_DIR, nome_nuvola)
    if os.path.exists(cartella):
        shutil.rmtree(cartella)
    os.makedirs(cartella)
    print(f"\nCartella output: {cartella}")
    return cartella


def _snapshot_matplotlib(
    pcd_originale: o3d.geometry.PointCloud,
    colori_out: np.ndarray,
    percorso: str,
    larghezza: int = 1920,
    altezza: int = 1080,
    prompt_punti_3d: np.ndarray = None,
):
    """
    Genera snapshot ad alta risoluzione (1920x1080) usando matplotlib in pura modalità CPU headless.
    Garantisce proporzioni e visibilità dei punti identiche sia per la scena completa che per i singoli tronchi.
    Se fornita una terna di punti prompt 3D, li evidenzia con marker chiari e targhette informative.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(pcd_originale.points)
    n_punti = len(xyz)

    # Calibrazione adattiva della dimensione dei punti per risoluzione 1920x1080
    if n_punti <= 20000:
        point_size = 14.0
    elif n_punti <= 60000:
        point_size = 7.0
    elif n_punti <= 120000:
        point_size = 5.0
    else:
        point_size = max(2.0, 300000.0 / n_punti)

    dpi = 100
    fig, ax = plt.subplots(figsize=(larghezza / dpi, altezza / dpi), dpi=dpi)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Vista dall'alto (XY), ordinata per Z crescente per sovrascrivere con i punti più alti
    ordine_z = np.argsort(xyz[:, 2])
    ax.scatter(
        xyz[ordine_z, 0], xyz[ordine_z, 1],
        c=colori_out[ordine_z],
        s=point_size, linewidths=0,
    )

    # Se forniti i punti prompt (terna Point-SAM 3D), evidenziali visibilmente sul tronco
    if prompt_punti_3d is not None and len(prompt_punti_3d) > 0:
        pts_pr = np.asarray(prompt_punti_3d)

        # Cerchio esterno fluorescente con bordo nero spesso
        ax.scatter(
            pts_pr[:, 0], pts_pr[:, 1],
            c="#00FFFF", s=280, marker="o", edgecolors="#000000", linewidths=2.5,
            zorder=20, label="Prompt Point-SAM",
        )
        # Stella interna magenta di forte contrasto
        ax.scatter(
            pts_pr[:, 0], pts_pr[:, 1],
            c="#FF0055", s=90, marker="*", zorder=21,
        )

        # Targhette con etichetta leggibile
        etichette = ["P1 (Testa)", "P2 (Centro)", "P3 (Coda)"]
        for pt_i, pt_coord in enumerate(pts_pr):
            lbl = etichette[pt_i] if pt_i < len(etichette) else f"P{pt_i+1}"
            ax.annotate(
                lbl,
                (pt_coord[0], pt_coord[1]),
                textcoords="offset points",
                xytext=(12, 10),
                fontsize=11,
                fontweight="bold",
                color="#000000",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFF00", edgecolor="#000000", alpha=0.9),
                zorder=25,
            )

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(percorso, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="#FFFFFF")
    plt.close(fig)


def _prova_visualizer_o3d(pcd_vis: o3d.geometry.PointCloud, percorso: str, larghezza: int, altezza: int) -> bool:
    """
    In ambienti WSL/headless, l'inizializzazione di Open3D Visualizer con GLFW fallisce
    a livello di driver grafico C++ provocando un SIGSEGV (codice 139) non intercettabile da Python.
    Restituisce False per usare direttamente _snapshot_matplotlib che è affidabile e headless-safe.
    """
    return False


def salva_snapshot_tronco(
    pcd_originale: o3d.geometry.PointCloud,
    maschera: np.ndarray,
    colore: list,
    idx_tronco: int,
    cartella: str,
    score: float,
    prompt_terna_3d: np.ndarray = None,
):
    """
    Salva uno snapshot 3D ad alta risoluzione (1920x1080) con il tronco segmentato
    nel suo colore, la scena in grigio e la terna di prompt 3D evidenziata.
    """
    colori_out = np.ones((len(pcd_originale.points), 3)) * 0.75  # grigio chiaro su sfondo bianco
    colori_out[maschera] = colore

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = pcd_originale.points
    pcd_vis.colors = o3d.utility.Vector3dVector(colori_out)

    percorso = os.path.join(cartella, f"tronco_{idx_tronco:02d}_score{score:.2f}.png")

    # Usa esattamente 1920x1080 per garantire lo stesso rapporto immagine/punti di scena_completa_segmentata
    _snapshot_matplotlib(pcd_originale, colori_out, percorso, 1920, 1080, prompt_punti_3d=prompt_terna_3d)

    print(f"  Snapshot salvato: {os.path.basename(percorso)}")


def salva_scena_completa(
    pcd_originale: o3d.geometry.PointCloud,
    maschere: list,
    cartella: str,
):
    """
    Salva una vista 3D con tutti i tronchi distinguibili da colori diversi
    e il terreno/sfondo in grigio chiaro.
    Tenta Open3D Visualizer; se non disponibile (WSL headless) usa matplotlib.
    """
    colori_out = np.ones((len(pcd_originale.points), 3)) * 0.75  # grigio chiaro su sfondo bianco
    for i, maschera in enumerate(maschere):
        colore = TAVOLOZZA_COLORI[i % len(TAVOLOZZA_COLORI)]
        colori_out[maschera] = colore

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = pcd_originale.points
    pcd_vis.colors = o3d.utility.Vector3dVector(colori_out)

    percorso_png = os.path.join(cartella, "scena_completa_segmentata.png")
    if not _prova_visualizer_o3d(pcd_vis, percorso_png, 1920, 1080):
        _snapshot_matplotlib(pcd_originale, colori_out, percorso_png, 1920, 1080)

    # Salva anche il file PLY con i colori della segmentazione finale
    percorso_ply = os.path.join(cartella, "nuvola_segmentata.ply")
    o3d.io.write_point_cloud(percorso_ply, pcd_vis)
    print(f"  Nuvola segmentata PLY salvata: nuvola_segmentata.ply")


def salva_immagine_rilevamento(
    immagine_2d: np.ndarray,
    maschere_2d: list,
    terne_2d: list,
    cartella: str,
):
    """
    Salva l'immagine 2D con i contorni di ogni tronco rilevato da SAM 2D
    e le 3 terne di punti prompt evidenziate (cerchi colorati con bordo bianco).
    """
    overlay = immagine_2d.copy()

    for i, (maschera_dict, terna) in enumerate(zip(maschere_2d, terne_2d)):
        if terna is None:
            continue
        colore_bgr = [int(c * 255) for c in TAVOLOZZA_COLORI[i % len(TAVOLOZZA_COLORI)]][::-1]

        # Contorno della maschera 2D
        contorno_bin = (maschera_dict["segmentation"].astype(np.uint8) * 255)
        contorni, _ = cv2.findContours(contorno_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contorni, -1, colore_bgr, 2)

        # Disegna visibilmente i 3 punti prompt (estremo A, centro, estremo B)
        for pt in terna:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(overlay, (px, py), 6, (255, 255, 255), -1)
            cv2.circle(overlay, (px, py), 4, colore_bgr, -1)

        # Etichetta numerica
        y_m, x_m = np.where(maschera_dict["segmentation"])
        cx, cy = int(x_m.mean()), int(y_m.mean())
        cv2.putText(overlay, f"T{i+1}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(overlay, f"T{i+1}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colore_bgr, 2)

    percorso = os.path.join(cartella, "02_rilevamento_modello1_con_terne.png")
    cv2.imwrite(percorso, overlay)
    print(f"  Immagine rilevamento salvata: 02_rilevamento_modello1_con_terne.png")


# ==============================================================================
#  MAIN - ORCHESTRAZIONE DELLA PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Automatica SAM 2D + Point-SAM 3D per segmentazione tronchi"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Percorso al file sorgente della nuvola (.las o .ply)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=MAX_POINTS,
        help=f"Soglia massima di punti per lo sfoltimento (default: {MAX_POINTS}; <=0 disabilita)",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Dimensione voxel per il downsampling geometrico (default: disabilitato)",
    )
    parser.add_argument(
        "--risoluzione",
        type=int,
        default=RISOLUZIONE_PROIEZIONE,
        help=f"Risoluzione dell'immagine 2D nadirale (default: {RISOLUZIONE_PROIEZIONE})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERRORE] File non trovato: {args.input}")
        sys.exit(1)

    nome_nuvola = os.path.splitext(os.path.basename(args.input))[0]
    cartella_out = prepara_cartella_output(nome_nuvola)

    # ------------------------------------------------------------------
    # STEP 1 - Caricamento e sfoltimento
    # ------------------------------------------------------------------
    pcd = carica_nuvola(args.input, max_points=args.max_points, voxel_size=args.voxel_size)

    # ------------------------------------------------------------------
    # STEP 2 - Proiezione nadirale 2D
    # ------------------------------------------------------------------
    info_proj = proietta_2d(pcd, risoluzione=args.risoluzione)

    # Salva immagine nadirale grezza
    cv2.imwrite(os.path.join(cartella_out, "01_vista_nadirale_2d.png"), info_proj["immagine"])
    print(f"  Immagine nadirale salvata: 01_vista_nadirale_2d.png")

    # ------------------------------------------------------------------
    # STEP 3 - Modello 1: SAM 2D - Rilevamento tronchi
    # ------------------------------------------------------------------
    maschere_2d = rileva_tronchi_sam2d(info_proj)

    if not maschere_2d:
        print("\n[ATTENZIONE] Nessun tronco rilevato dal Modello 1. Pipeline terminata.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # STEP 4 - Estrazione terne di punti 3D
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 4 - Estrazione terne di punti 3D per tronco")
    print(f"{'='*60}")

    terne_originali = []
    terne_2d_valide = []
    maschere_2d_valide = []
    for i, m in enumerate(maschere_2d):
        terna_3d, terna_2d = estrai_terna_punti_3d(m["segmentation"], info_proj)
        if terna_3d is not None:
            terne_originali.append(terna_3d)
            terne_2d_valide.append(terna_2d)
            maschere_2d_valide.append(m)
            print(f"  Tronco {i+1}: 3 prompt estratti (estremo A, centro, estremo B)")
        else:
            print(f"  Tronco {i+1}: maschera non valida, ignorata")

    if not terne_originali:
        print("[ATTENZIONE] Nessuna terna valida estratta. Pipeline terminata.")
        sys.exit(0)

    # Salva immagine 2D con rilevamenti e prompt evidenziati
    salva_immagine_rilevamento(info_proj["immagine"], maschere_2d_valide, terne_2d_valide, cartella_out)

    print(f"\nTronchi validi da segmentare: {len(terne_originali)}")

    # ------------------------------------------------------------------
    # STEP 5 - Modello 2: Point-SAM 3D
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 5 - Modello 2: Point-SAM 3D - Segmentazione")
    print(f"{'='*60}")

    pcsam = carica_pcsam()

    # Preparazione tensori della nuvola (normalizzazione in [-1, 1])
    xyz = np.asarray(pcd.points).astype(np.float32)
    shift = xyz.mean(axis=0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    xyz_norm = (xyz - shift) / scale

    rgb = np.asarray(pcd.colors).astype(np.float32)  # già in [0,1]

    pc_xyz_t = torch.from_numpy(xyz_norm).cuda().float().unsqueeze(0)  # [1, N, 3]
    pc_rgb_t = torch.from_numpy(rgb).cuda().float().unsqueeze(0)        # [1, N, 3]

    # PRE-CALCOLO ENCODER - una sola volta per tutta la scena (caching)
    print("Pre-calcolo Encoder 3D (una sola volta per tutti i tronchi)...")
    pcsam.set_point_cloud(pc_xyz_t, pc_rgb_t)
    print("Encoder pronto. Avvio segmentazione...")

    maschere_3d = []

    for i, terna_xyz in enumerate(terne_originali):
        print(f"\n  Segmentazione tronco {i+1}/{len(terne_originali)}...")

        # Normalizza i punti prompt nelle stesse coordinate della nuvola
        terna_norm = (terna_xyz - shift) / scale

        maschera, score = segmenta_tronco(pcsam, pc_xyz_t, pc_rgb_t, terna_norm)
        n_punti_seg = maschera.sum()
        print(f"  Punti segmentati: {n_punti_seg:,} (IoU score: {score:.3f})")

        maschere_3d.append(maschera)

        colore = TAVOLOZZA_COLORI[i % len(TAVOLOZZA_COLORI)]
        salva_snapshot_tronco(pcd, maschera, colore, i + 1, cartella_out, score, prompt_terna_3d=terna_xyz)

    # ------------------------------------------------------------------
    # OUTPUT FINALE
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("OUTPUT - Salvataggio risultati finali")
    print(f"{'='*60}")

    salva_scena_completa(pcd, maschere_3d, cartella_out)

    print(f"\n{'='*60}")
    print(f"Pipeline completata con successo!")
    print(f"  Tronchi segmentati: {len(maschere_3d)}")
    print(f"  Output salvato in : {cartella_out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
