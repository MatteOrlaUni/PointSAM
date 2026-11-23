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

# Soglia massima di punti per la nuvola (<=0 per disabilitare lo sfoltimento)
MAX_POINTS = 50000

# Dimensione voxel per downsampling geometrico (None per disabilitare)
VOXEL_SIZE = None

# Risoluzione pixel dell'immagine 2D generata dalla proiezione nadirale
RISOLUZIONE_PROIEZIONE = 1024

# Soglia minima di area in pixel per considerare un oggetto rilevato da SAM 2D
MIN_AREA_MASCHERA = 500

# Soglia massima di area in pixel (evita maschere enormi che coprono lo sfondo)
MAX_AREA_MASCHERA_PERCENTUALE = 0.4  # 40% dell'immagine

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
    Carica una nuvola di punti da file .las o .ply, con sfoltimento automatico.
    Restituisce un open3d.PointCloud con coordinate XYZ e colori RGB [0,1].
    """
    print(f"\n{'='*60}")
    print("STEP 1 - Caricamento nuvola di punti")
    print(f"{'='*60}")
    print(f"File: {percorso}")

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
            # Nessun colore: usa grigio uniforme
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(xyz), 3)) * 0.5
            )

    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(percorso)
        if not pcd.has_colors():
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(pcd.points), 3)) * 0.5
            )
    else:
        raise ValueError(f"Formato file non supportato: {ext}. Usa .las o .ply")

    n_originale = len(pcd.points)
    print(f"Punti caricati: {n_originale:,}")

    # Sfoltimento con voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        print(f"Voxel downsampling (voxel_size={voxel_size})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Punti dopo voxel downsampling: {len(pcd.points):,}")

    # Sfoltimento numerico casuale
    if max_points is not None and max_points > 0 and len(pcd.points) > max_points:
        print(f"Sfoltimento: {len(pcd.points):,} -> {max_points:,} punti...")
        idx = np.random.choice(len(pcd.points), max_points, replace=False)
        pcd = pcd.select_by_index(idx)

    print(f"Nuvola pronta: {len(pcd.points):,} punti")
    return pcd


# ==============================================================================
#  STEP 2 - PROIEZIONE NADIRALE 2D
# ==============================================================================

def proietta_2d(pcd: o3d.geometry.PointCloud, risoluzione: int = RISOLUZIONE_PROIEZIONE):
    """
    Proietta la nuvola dall'alto (vista nadirale = piano XY) generando:
    - immagine 2D colorata con mappa di calore Z (altezza)
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
    z_min, z_max = z.min(), z.max()

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

    # Mappa di calore Z (più alto = più luminoso) dipinta dal basso verso l'alto
    z_norm_uint8 = ((z - z_min) / (z_max - z_min + 1e-9) * 255).astype(np.uint8)
    immagine = np.zeros((altezza, larghezza, 3), dtype=np.uint8)
    ordine_z = np.argsort(z_norm_uint8)
    colori_jet = cv2.applyColorMap(z_norm_uint8, cv2.COLORMAP_MAGMA).squeeze()
    immagine[v[ordine_z], u[ordine_z]] = colori_jet[ordine_z]

    # KDTree per ritrovare il punto 3D più vicino dato un pixel 2D
    kdtree_2d = KDTree(np.vstack((u, v)).T.astype(np.float32))

    info = {
        "immagine": immagine,
        "kdtree_2d": kdtree_2d,
        "xyz_originale": xyz,
        "u": u, "v": v,
        "x_min": x_min, "y_max": y_max,
        "delta_x": delta_x, "delta_y": delta_y,
        "larghezza": larghezza, "altezza": altezza,
    }

    print(f"Immagine 2D generata: {larghezza}x{altezza} px")
    return info


# ==============================================================================
#  STEP 3 - MODELLO 1: SAM 2D - RILEVAMENTO TRONCHI
# ==============================================================================

def rileva_tronchi_sam2d(immagine: np.ndarray, checkpoint: str = SAM2D_CHECKPOINT):
    """
    Applica SAM 2D (ViT-H Automatic Mask Generator) sull'immagine nadirale.
    Restituisce le maschere binarie filtrate per area, una per tronco rilevato.
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

    h, w = immagine.shape[:2]
    area_totale = h * w

    generatore = SamAutomaticMaskGenerator(
        model=sam2d,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=MIN_AREA_MASCHERA,
    )

    print("Generazione maschere in corso...")
    maschere_raw = generatore.generate(immagine)
    print(f"Maschere generate dal modello: {len(maschere_raw)}")

    # Filtraggio: elimina maschere troppo piccole o troppo grandi (sfondo)
    maschere_filtrate = [
        m for m in maschere_raw
        if MIN_AREA_MASCHERA <= m["area"] <= area_totale * MAX_AREA_MASCHERA_PERCENTUALE
    ]
    maschere_filtrate.sort(key=lambda m: m["area"], reverse=True)

    print(f"Maschere accettate dopo filtraggio: {len(maschere_filtrate)}")

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
    Usa la ri-proiezione inversa via KDTree 2D → punto 3D più vicino.
    """
    u_tutti = info_proiezione["u"]
    v_tutti = info_proiezione["v"]
    xyz_originale = info_proiezione["xyz_originale"]
    kdtree_2d = info_proiezione["kdtree_2d"]

    # Coordinate pixel dei punti nella maschera
    pixel_v, pixel_u = np.where(maschera_binaria)
    if len(pixel_u) < 10:
        return None

    punti_2d = np.vstack((pixel_u, pixel_v)).T.astype(np.float32)

    # --- Centro della maschera ---
    centro_2d = punti_2d.mean(axis=0)

    # --- Asse principale tramite PCA sui pixel della maschera ---
    punti_centrati = punti_2d - centro_2d
    _, _, Vt = np.linalg.svd(punti_centrati, full_matrices=False)
    asse_principale = Vt[0]  # direzione dell'asse lungo del tronco

    # Proiezione scalare di ogni pixel lungo l'asse
    proiezioni = punti_centrati @ asse_principale

    # I due estremi: punto con proiezione minima e massima
    idx_estremo_a = np.argmin(proiezioni)
    idx_estremo_b = np.argmax(proiezioni)

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

    return np.array(punti_3d)  # shape (3, 3)


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
    larghezza: int = 1280,
    altezza: int = 720,
):
    """
    Fallback: genera uno snapshot 2D nadirale usando matplotlib quando
    Open3D Visualizer non è disponibile (es. WSL headless senza server X).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(pcd_originale.points)
    dpi = 100
    fig, ax = plt.subplots(figsize=(larghezza / dpi, altezza / dpi), dpi=dpi)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    # Vista dall'alto (XY), ordinata per Z crescente per sovrascrivere con i punti più alti
    ordine_z = np.argsort(xyz[:, 2])
    ax.scatter(
        xyz[ordine_z, 0], xyz[ordine_z, 1],
        c=colori_out[ordine_z],
        s=0.5, linewidths=0,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(percorso, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _prova_visualizer_o3d(pcd_vis: o3d.geometry.PointCloud, percorso: str, larghezza: int, altezza: int) -> bool:
    """
    Tenta di usare Open3D Visualizer in modalità offscreen.
    Restituisce True se riesce, False altrimenti (WSL headless senza X).
    """
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=larghezza, height=altezza)
        vis.add_geometry(pcd_vis)
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_up([0, 1, 0])
        ctr.set_lookat(np.asarray(pcd_vis.points).mean(axis=0))
        ctr.set_zoom(0.5)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(percorso)
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"  [INFO] Open3D Visualizer non disponibile ({e}), uso matplotlib.")
        return False


def salva_snapshot_tronco(
    pcd_originale: o3d.geometry.PointCloud,
    maschera: np.ndarray,
    colore: list,
    idx_tronco: int,
    cartella: str,
    score: float,
):
    """
    Salva uno snapshot 3D con il tronco segmentato evidenziato nel suo colore
    e il resto della scena in grigio neutro.
    Tenta Open3D Visualizer; se non disponibile (WSL headless) usa matplotlib.
    """
    colori_out = np.ones((len(pcd_originale.points), 3)) * 0.3  # grigio
    colori_out[maschera] = colore

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = pcd_originale.points
    pcd_vis.colors = o3d.utility.Vector3dVector(colori_out)

    percorso = os.path.join(cartella, f"tronco_{idx_tronco:02d}_score{score:.2f}.png")

    if not _prova_visualizer_o3d(pcd_vis, percorso, 1280, 720):
        _snapshot_matplotlib(pcd_originale, colori_out, percorso, 1280, 720)

    print(f"  Snapshot salvato: {os.path.basename(percorso)}")


def salva_scena_completa(
    pcd_originale: o3d.geometry.PointCloud,
    maschere: list,
    cartella: str,
):
    """
    Salva una vista 3D con tutti i tronchi distinguibili da colori diversi
    e il terreno/sfondo in grigio.
    Tenta Open3D Visualizer; se non disponibile (WSL headless) usa matplotlib.
    """
    colori_out = np.ones((len(pcd_originale.points), 3)) * 0.25  # grigio scuro
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
    terne_punti: list,
    cartella: str,
):
    """
    Salva l'immagine 2D con i contorni di ogni tronco rilevato da SAM 2D
    e le 3 terne di punti prompt evidenziate (cerchi colorati).
    """
    overlay = immagine_2d.copy()

    for i, (maschera_dict, terna) in enumerate(zip(maschere_2d, terne_punti)):
        if terna is None:
            continue
        colore_bgr = [int(c * 255) for c in TAVOLOZZA_COLORI[i % len(TAVOLOZZA_COLORI)]][::-1]

        # Contorno della maschera 2D
        contorno_bin = (maschera_dict["segmentation"].astype(np.uint8) * 255)
        contorni, _ = cv2.findContours(contorno_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contorni, -1, colore_bgr, 2)

        # Etichetta numerica
        y_m, x_m = np.where(maschera_dict["segmentation"])
        cx, cy = int(x_m.mean()), int(y_m.mean())
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
    maschere_2d = rileva_tronchi_sam2d(info_proj["immagine"])

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
    maschere_2d_valide = []
    for i, m in enumerate(maschere_2d):
        terna = estrai_terna_punti_3d(m["segmentation"], info_proj)
        if terna is not None:
            terne_originali.append(terna)
            maschere_2d_valide.append(m)
            print(f"  Tronco {i+1}: 3 prompt estratti (estremo A, centro, estremo B)")
        else:
            print(f"  Tronco {i+1}: maschera troppo piccola, ignorata")

    if not terne_originali:
        print("[ATTENZIONE] Nessuna terna valida estratta. Pipeline terminata.")
        sys.exit(0)

    # Salva immagine 2D con rilevamenti e prompt evidenziati
    salva_immagine_rilevamento(info_proj["immagine"], maschere_2d_valide, terne_originali, cartella_out)

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
        salva_snapshot_tronco(pcd, maschera, colore, i + 1, cartella_out, score)

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
