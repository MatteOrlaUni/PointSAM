import numpy as np
import open3d as o3d


def downsample_point_cloud(pcd, max_points=50000, voxel_size=None):
    """
    Sfoltisce la nuvola di punti se troppo densa/pesante per alleggerire
    il rendering 3D nel browser e l'inferenza del modello Point-SAM.

    :param pcd: open3d.geometry.PointCloud
    :param max_points: int, numero massimo di punti desiderati. Se None o <= 0, nessun limite numerico.
    :param voxel_size: float (opzionale), dimensione del voxel per voxel_down_sample.
    :return: open3d.geometry.PointCloud sfoltita
    """
    num_points = len(pcd.points)

    # 1. Applicazione facoltativa del voxel downsampling (riduzione geometrica uniforme)
    if voxel_size is not None and voxel_size > 0:
        print(f"[INFO Sfoltimento] Applicazione Voxel Downsampling (voxel_size={voxel_size})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        num_points = len(pcd.points)
        print(f"[INFO Sfoltimento] Punti dopo voxel downsampling: {num_points}")

    # 2. Controllo della soglia massima di punti
    if max_points is not None and max_points > 0 and num_points > max_points:
        print(f"[INFO Sfoltimento] Nuvola troppo pesante: {num_points} punti (soglia massima: {max_points}). Sfoltimento in corso...")
        indices = np.random.choice(num_points, max_points, replace=False)
        pcd = pcd.select_by_index(indices)
        print(f"[INFO Sfoltimento] Nuvola sfoltita con successo a {len(pcd.points)} punti.")
    else:
        print(f"[INFO Sfoltimento] Nuvola idonea: {num_points} punti (soglia: {max_points}). Nessun campionamento necessario.")

    return pcd


def load_ply(filename, max_points=50000, voxel_size=None):
    """
    Carica una nuvola di punti da file .ply, applicando un eventuale sfoltimento
    se supera la soglia di punti massima consentita.

    :param filename: percorso del file .ply
    :param max_points: numero massimo di punti consentiti (default: 50000)
    :param voxel_size: dimensione del voxel per downsampling volumetrico (default: None)
    :return: array numpy di forma (N, 6) con [x, y, z, r, g, b] (RGB su scala 0-255)
    """
    pcd = o3d.io.read_point_cloud(filename)

    # Controllo e sfoltimento della nuvola
    pcd = downsample_point_cloud(pcd, max_points=max_points, voxel_size=voxel_size)

    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Se il file non ha colori, forziamo il bianco
    if colors.shape[0] == 0:
        colors = np.ones_like(xyz) * 255.0
    else:
        # Open3D restituisce i colori normalizzati tra 0 e 1
        # app.py li dividerà nuovamente per 255, quindi li riportiamo alla scala 0-255
        colors = colors * 255.0

    return np.concatenate((xyz, colors), axis=1)
