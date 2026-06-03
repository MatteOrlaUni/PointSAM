import numpy as np

# def load_ply(filename):
#     with open(filename, "r") as rf:
#         while True:
#             try:
#                 line = rf.readline()
#             except:
#                 raise NotImplementedError
#             if "end_header" in line:
#                 break
#             if "element vertex" in line:
#                 arr = line.split()
#                 num_of_points = int(arr[2])

#         # print("%d points in ply file" %num_of_points)
#         points = np.zeros([num_of_points, 6])
#         for i in range(points.shape[0]):
#             point = rf.readline().split()
#             assert len(point) == 6
#             points[i][0] = float(point[0])
#             points[i][1] = float(point[1])
#             points[i][2] = float(point[2])
#             points[i][3] = float(point[3])
#             points[i][4] = float(point[4])
#             points[i][5] = float(point[5])
#     rf.close()
#     del rf
#     return points

import open3d as o3d

def load_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
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