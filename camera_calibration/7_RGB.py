import cv2
import numpy as np
import glob
import os

def calibrate_camera_from_images(image_dir, num_rows, num_cols, square_length_mm, marker_length_mm):
    # Board definition
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        (num_cols, num_rows),  # Reihen x Spalten
        square_length_mm / 1000.0,  # mm → m
        marker_length_mm / 1000.0,
        dictionary
    )

    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    # Bilder laden
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))  # oder *.jpg
    print(f"{len(image_paths)} Bilder gefunden in: {image_dir}")

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret and charuco_corners is not None and charuco_ids is not None:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                if image_size is None:
                    image_size = gray.shape[::-1]

    if len(all_charuco_corners) < 5:
        raise RuntimeError("Nicht genug gültige Bilder für Kalibrierung!")

    # Kalibrierung durchführen
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # Reprojektion Error
    total_error = 0
    total_points = 0
    for i in range(len(all_charuco_corners)):
        projected, _ = cv2.projectPoints(
            board.getChessboardCorners(), rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(all_charuco_corners[i], projected, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(all_charuco_corners[i])

    reprojection_error = np.sqrt(total_error / total_points)

    print("\n Kalibrierung abgeschlossen")
    print("Kameramatrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print(f"Reprojektion Error: {reprojection_error:.4f}")

    return camera_matrix, dist_coeffs, reprojection_error


# --------------------------
# Anwendung:
if __name__ == "__main__":
    image_folder = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\RGB_7"

    calibrate_camera_from_images(
        image_dir=image_folder,
        num_rows=7,
        num_cols=10,
        square_length_mm=80,
        marker_length_mm=60
    )
