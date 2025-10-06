import cv2
import numpy as np
import os
import glob

# === Basisordner ===
base_dir = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112"

# Kalibrierungsergebnisse laden
rgb_calib_dir = r"D:\CameraCalibrationProjekt\camera_calibration\CalibrationResults_RGB"
tof_calib_dir = os.path.join(base_dir, "CalibrationResults_TOF_7")

rgb_K = np.load(os.path.join(rgb_calib_dir, "000369930112_rgb_camera_matrix.npy"))
rgb_dist = np.load(os.path.join(rgb_calib_dir, "000369930112_rgb_distortion_coeffs.npy"))

tof_K = np.load(os.path.join(tof_calib_dir, "000369930112_tof_camera_matrix.npy"))
tof_dist = np.load(os.path.join(tof_calib_dir, "000369930112_tof_distortion_coeffs.npy"))

print("RGB K:", rgb_K)
print("TOF K:", tof_K)

# Undistort Funktion
def undistort_and_save(img_path, K, dist, out_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Bild nicht gefunden: {img_path}")
        return

    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    undistorted = cv2.undistort(img, K, dist, None, new_K)
    cv2.imwrite(out_path, undistorted)
    print(f"Gespeichert: {out_path}")

# Zielordner
out_dir = os.path.join(base_dir, "undistorted")
os.makedirs(out_dir, exist_ok=True)

# --- RGB Bilder ---
for f in glob.glob(os.path.join(base_dir, "color_*.png")):
    fname = os.path.basename(f).replace("color_", "color_undistorted_")
    out_path = os.path.join(out_dir, fname)
    undistort_and_save(f, rgb_K, rgb_dist, out_path)

# --- IR Bilder ---
for f in glob.glob(os.path.join(base_dir, "ir_*.png")):  # nur die PNG, nicht *_raw.npy
    fname = os.path.basename(f).replace("ir_", "ir_undistorted_")
    out_path = os.path.join(out_dir, fname)
    undistort_and_save(f, tof_K, tof_dist, out_path)

print("Fertig! Alle color_*.png und ir_*.png wurden entzerrt.")
