"""
las_extractor.py
----------------
Adattato da: PointCloud/src/las_data_extractor.py

Genera la vista 2D nadirale della nuvola di punti LAS colorata con la
mappa MAGMA (colorazione per quota), salvando:
  - color_data.npy     : immagine RGB [H, W, 3] uint8
  - depth_data.npy     : mappa di profondità [H, W] float32
  - camera_params.npz  : matrice intrinseca + estrinseca della camera virtuale
  - offset.npy         : offset XYZ usato per centrare la nuvola
  - color_preview.png  : anteprima PNG della vista

La funzione principale `extract_magma_view` può essere chiamata direttamente
da prompt_generator.py senza passare per subprocess.

Fonte originale: Tommaso Ticci, Università di Firenze
"""

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import laspy
import cv2
import os


def extract_magma_view(
    las_path: str,
    data_dir: str,
    width: int = 1920,
    height: int = 1080,
) -> dict:
    """Genera la vista nadirale MAGMA dalla nuvola LAS e la salva in data_dir.

    Args:
        las_path : Percorso del file .las di input.
        data_dir : Directory in cui salvare i file intermedi.
        width    : Larghezza della finestra di rendering virtuale.
        height   : Altezza della finestra di rendering virtuale.

    Returns:
        Dizionario con le chiavi:
            'image_np'   : np.ndarray [H, W, 3] uint8 — immagine MAGMA
            'depth_np'   : np.ndarray [H, W] float32  — mappa profondità
            'intrinsic'  : np.ndarray [3, 3]           — matrice intrinseca
            'extrinsic'  : np.ndarray [4, 4]           — matrice estrinseca
            'offset'     : np.ndarray [3]              — offset di centratura
            'points'     : np.ndarray [N, 3]           — coordinate XYZ originali
            'colors_rgb' : np.ndarray [N, 3] float [0,1] — colori originali del LAS
    """
    os.makedirs(data_dir, exist_ok=True)

    # --- Lettura file LAS ---
    print(f"[Extractor] Lettura {os.path.basename(las_path)}...")
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)

    # Colori originali dal LAS (per usarli successivamente in PointSAM)
    if hasattr(las, 'red'):
        r = np.array(las.red) / 65535.0
        g = np.array(las.green) / 65535.0
        b = np.array(las.blue) / 65535.0
        colors_rgb = np.vstack((r, g, b)).transpose().astype(np.float32)
    else:
        print("[Extractor] ATTENZIONE: colori RGB non trovati, uso bianco.")
        colors_rgb = np.ones((len(points), 3), dtype=np.float32)

    # --- Centratura nuvola ---
    offset = np.mean(points, axis=0)
    points_centered = points - offset

    # --- Colorazione MAGMA per quota ---
    z_values = points_centered[:, 2]
    z_min = np.percentile(z_values, 1)
    z_max = np.percentile(z_values, 99)
    z_norm = np.clip((z_values - z_min) / (z_max - z_min + 1e-7), 0, 1)
    magma_colors = plt.get_cmap('magma')(z_norm)[:, :3]

    # --- Creazione nuvola Open3D ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_centered)
    pcd.colors = o3d.utility.Vector3dVector(magma_colors)

    # --- Rendering nadirale (vista dall'alto) ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Cattura Nadirale - MAGMA",
        width=width, height=height,
        visible=False,  # headless: nessuna finestra visibile
    )
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])

    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    render_option.point_size = max(0.5, 20.0 / max(bbox.get_extent()))
    ctr.set_front([0, 0, 1])
    ctr.set_lookat(center)
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()

    print("[Extractor] Generazione vista MAGMA...")
    image_float = vis.capture_screen_float_buffer(True)
    depth_float = vis.capture_depth_float_buffer(True)
    params = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()

    image_np = (np.asarray(image_float) * 255).astype(np.uint8)
    depth_np = np.asarray(depth_float)
    intrinsic = params.intrinsic.intrinsic_matrix
    extrinsic = params.extrinsic

    # --- Salvataggio su disco ---
    np.save(os.path.join(data_dir, "color_data.npy"), image_np)
    np.save(os.path.join(data_dir, "depth_data.npy"), depth_np)
    np.savez(os.path.join(data_dir, "camera_params.npz"),
             intrinsic=intrinsic, extrinsic=extrinsic)
    np.save(os.path.join(data_dir, "offset.npy"), offset)
    np.save(os.path.join(data_dir, "points_original.npy"), points)
    np.save(os.path.join(data_dir, "colors_rgb.npy"), colors_rgb)
    cv2.imwrite(
        os.path.join(data_dir, "color_preview.png"),
        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    )

    print(f"[Extractor] Vista salvata in {data_dir}.")
    return {
        "image_np":   image_np,
        "depth_np":   depth_np,
        "intrinsic":  intrinsic,
        "extrinsic":  extrinsic,
        "offset":     offset,
        "points":     points,
        "colors_rgb": colors_rgb,
    }

