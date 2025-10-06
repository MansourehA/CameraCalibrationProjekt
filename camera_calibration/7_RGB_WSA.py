import numpy as np
import cv2
import glob
import os

print(cv2.__version__)

# Hinweis: Charuco-Board Parameter für dein Setup
square_Vertical = 7     # Anzahl Reihen (squaresY)
square_Horizontal = 10  # Anzahl Spalten (squaresX)
square_size = 0.080     # Feldgröße in Meter (80 mm)
marker_size = 0.060     # Markergröße in Meter (60 mm)

# Hinweis: ArUco Dictionary und Charuco-Board definieren
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((square_Horizontal, square_Vertical),
                               square_size, marker_size, aruco_dict)
board.setLegacyPattern(True)

# Hinweis: Detektorparameter
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# Hinweis: Seriennummer der Kamera (für Ergebnis-Dateinamen)
camera_serial = "000369930112"

# Hinweis: Pfad mit RGB-Bildern dieser Kamera
image_dir = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\RGB_7"
images = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))

print(f"Processing RGB images for Camera {camera_serial}...")
print(f"Found {len(images)} images in {image_dir}")

all_corners = []
all_ids = []
img_size = None

# Hinweis: Schleife über alle Bilder
for i, fname in enumerate(images):
    print(f"Processing image {i+1}/{len(images)}: {fname}")

    # Bild laden und in Grau umwandeln
    img = cv2.imread(fname)
    if img is None:
        print(f"Warnung: konnte {fname} nicht laden.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Marker erkennen
    corners, ids, _ = detector.detectMarkers(gray)

    if len(corners) > 0:
        # Charuco-Ecken interpolieren
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        # Nur verwenden, wenn genügend Ecken erkannt wurden
        if charuco_retval is not None and charuco_retval > 20:
            print(f"Found Charuco corners in image {i+1}")
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            if img_size is None:
                img_size = gray.shape[::-1]  # (width, height)

print(f"Starting RGB calibration for Camera {camera_serial}...")

# Hinweis: Kalibrierung durchführen, wenn genügend Daten vorhanden
if len(all_corners) > 0 and img_size is not None:
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None
    )

    # Ergebnisse ausgeben
    print(f'Kalibrierungsergebnisse für RGB-Kamera {camera_serial}:')
    print("Kameramatrix:\n", camera_matrix)
    print("Verzerrungskoeffizienten:\n", distortion_coeffs)
    print("Reprojektion Fehler:", ret)

    # Ergebnisse speichern
    save_dir = "CalibrationResults_RGB"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f'{camera_serial}_rgb_camera_matrix.npy'), camera_matrix)
    np.save(os.path.join(save_dir, f'{camera_serial}_rgb_distortion_coeffs.npy'), distortion_coeffs)
    np.save(os.path.join(save_dir, f'{camera_serial}_rgb_reprojection_error.npy'), ret)

print(f"Finished calibration for RGB Camera {camera_serial}.\n")
print("RGB camera calibration done.")
