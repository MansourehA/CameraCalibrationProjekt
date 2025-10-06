import cv2
import numpy as np
import glob
import os

# --- Board Parameter (von deinem Board) ---
num_cols = 10   # innere Spalten (X)
num_rows = 7    # innere Reihen (Y)
square_length_m = 0.065   # 65 mm
marker_length_m = 0.048   # 48 mm

# --- Dictionary und Board ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard(
    (num_cols, num_rows),
    square_length_m,
    marker_length_m,
    dictionary
)

# --- Detector Parameter ---
detector_params = cv2.aruco.DetectorParameters()
charuco_params = cv2.aruco.CharucoParameters()
detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

# --- Ordner mit entzerrten Bildern ---
image_dir = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112\undistorted"

images = glob.glob(os.path.join(image_dir, "color_undistorted_*.png"))
print(f"Gefundene Bilder: {len(images)}")

for i, fname in enumerate(sorted(images)):
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[{i}] Bild konnte nicht gelesen werden: {fname}")
        continue

    # Falls Graustufen nötig
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # --- Marker + ChArUco erkennen ---
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # --- Visualisierung ---
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        print(f"[{i}] ChArUco erkannt: {len(charuco_ids)} Ecken ({os.path.basename(fname)})")
    else:
        print(f"[{i}] Nur ArUco Marker erkannt, keine ChArUco-Ecken ({os.path.basename(fname)})")

    # --- Bild kleiner anzeigen (scale 0.4) ---
    scale = 0.4
    vis_small = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))

    cv2.imshow("Debug", vis_small)
    key = cv2.waitKey(0)
    if key == 27:  # ESC zum Abbrechen
        break

cv2.destroyAllWindows()
