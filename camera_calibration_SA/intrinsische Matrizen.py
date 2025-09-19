import numpy as np
import cv2
import glob
import os

########################################
# Lade die intrinsischen Parameter für alle Kameras
########################################

save_dir = "X:/wiese/Mansoureh/Result_Aruco_small"
camera_dirs = ['000325420812', '000436120812', '000365930112', '000368930112', '000409930112', '000068700312']

# Definiere Kamera 4 (000365930112) als Zielkamera für PnP-Solve
target_camera_dir = '000325420812'
other_cameras = [camera for camera in camera_dirs if camera != target_camera_dir]

# Lade die intrinsischen Parameter für alle Kameras
camera_matrices = {}
distortion_coeffs = {}
for camera_dir in camera_dirs:
    camera_matrices[camera_dir] = np.load(os.path.join(save_dir, f'{camera_dir}_camera_matrix.npy'))
    distortion_coeffs[camera_dir] = np.load(os.path.join(save_dir, f'{camera_dir}_distortion_coeffs.npy'))
    print("Kameramatrix: ", camera_matrices[camera_dir])
    print("distortion_coeffs: ", distortion_coeffs[camera_dir])
    print(f"Kameramatrix und Verzerrungskoeffizienten geladen für Kamera {camera_dir}")