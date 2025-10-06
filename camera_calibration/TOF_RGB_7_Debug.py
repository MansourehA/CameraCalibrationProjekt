import cv2
import numpy as np
import glob
import os

# -----------------------------
# 🔧 Board-Parameter anpassen!
# -----------------------------
num_rows = 7     # innere Reihen (Charuco squares)
num_cols = 10    # innere Spalten
square_length_mm = 65   # Seitenlänge Schachfeld [mm]
marker_length_mm = 48   # Seitenlänge Marker [mm]

# Dictionary und Board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard(
    (num_cols, num_rows),
    square_length_mm / 1000.0,
    marker_length_mm / 1000.0,
    dictionary
)

# Detector-Parameter
detector_params = cv2.aruco.DetectorParameters()

# -----------------------------
# Ordner mit Bildern
# -----------------------------
image_dir = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112"
images = glob.glob(os.path.join(image_dir, "*.png"))

print(f"Gefundene Bilder: {len(images)}")

# -----------------------------
# Bilder durchgehen
# -----------------------------
for i, fname in enumerate(sorted(images)):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[{i}] Konnte Bild nicht laden: {fname}")
        continue

    # --- Schritt 1: ArUco Marker erkennen ---
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=detector_params)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        # --- Schritt 2: ChArUco-Ecken interpolieren ---
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, img, board
        )

        if retval is not None and retval > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            print(f"[{i}] {retval} ChArUco-Ecken erkannt")
        else:
            print(f"[{i}] Nur ArUco Marker erkannt, keine ChArUco-Ecken")
    else:
        print(f"[{i}] Keine Marker erkannt")

    # --- Anzeige ---
    cv2.imshow("Debug", cv2.resize(vis, (1280, 960)))
    key = cv2.waitKey(0)
    if key == 27:  # ESC -> abbrechen
        break

cv2.destroyAllWindows()
