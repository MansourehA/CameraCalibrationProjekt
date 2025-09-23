import numpy as np
import cv2
import glob
import os

print(cv2.__version__)

def calibrate_tof_camera(image_dir, num_rows, num_cols, square_length_mm, marker_length_mm):
    # --- Charuco-Board Definition ---
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        (num_cols, num_rows),           # (Spalten, Reihen)
        square_length_mm / 1000.0,      # mm -> m
        marker_length_mm / 1000.0,
        dictionary
    )
    board.setLegacyPattern(True)

    # --- Aruco-Detektor ---
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    all_corners = []
    all_ids = []
    img_size = None

    # --- Alle IR-Bilder laden ---
    images = glob.glob(os.path.join(image_dir, "ir_*.png"))
    print(f"Found {len(images)} IR frames in {image_dir}")

    for i, fname in enumerate(images):
        print(f"Processing IR frame {i+1}/{len(images)}: {fname}")

        # Bild laden (16-bit Grauwert von Azure Kinect IR)
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Konnte {fname} nicht laden.")
            continue

        # Normalisierung auf 8-bit, falls 16-bit IR
        if img.dtype != np.uint8:
            maxv = float(np.max(img))
            if maxv > 0:
                gray = cv2.convertScaleAbs(img, alpha=255.0/maxv)
            else:
                print(f"Frame {fname} leer.")
                continue
        else:
            gray = img

        # Marker erkennen
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if charuco_retval is not None and charuco_retval > 4:
                print(f"Charuco erkannt in Frame {i+1}")
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                if img_size is None:
                    img_size = gray.shape[::-1]

    # --- Kalibrierung nur, wenn genügend Ecken erkannt ---
    if len(all_corners) < 5:
        raise RuntimeError("Nicht genügend gültige Charuco-Erkennungen für Kalibrierung gefunden!")

    print("\nStarting ToF calibration...")

    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None
    )

    # --- Reprojektion-Fehler berechnen ---
    total_error = 0
    total_points = 0
    obj_points = board.getChessboardCorners()

    for i in range(len(all_corners)):
        ids = all_ids[i].flatten()
        obj_pts = obj_points[ids]
        proj, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], camera_matrix, distortion_coeffs)
        err = cv2.norm(all_corners[i], proj, cv2.NORM_L2)
        total_error += err ** 2
        total_points += len(ids)

    reprojection_error = np.sqrt(total_error / total_points)

    # --- Ergebnisse ausgeben ---
    print("\n Kalibrierung (ToF/IR) abgeschlossen")
    print("Kameramatrix (K):\n", camera_matrix)
    print("Verzerrungskoeffizienten (Dist):\n", distortion_coeffs.ravel())
    print(f"Reprojektion Fehler (RMS): {reprojection_error:.4f}")

    return camera_matrix, distortion_coeffs, reprojection_error


# --------------------------
# Anwendung
if __name__ == "__main__":
    image_folder = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\TofRGB"

    calibrate_tof_camera(
        image_dir=image_folder,
        num_rows=7,
        num_cols=10,
        square_length_mm=80,
        marker_length_mm=60
    )
