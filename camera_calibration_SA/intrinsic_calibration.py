import cv2
import numpy as np

def calibrate_with_charuco(image_list,num_rows,num_cols,square_length,marker_length):
    ### definieren der charuco parameter
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((num_rows, num_cols), square_length, marker_length, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    all_corners = []  # 2D-Bildpunkte
    all_ids = []  # Marker-IDs
    all_charuco_corners = []  # Interpolierte Charuco-Eckpunkte
    all_charuco_ids = []  # Interpolierte Charuco-IDs
    img_size = None  # Bildgröße, wird beim ersten Bild festgelegt
    debug = False

    for idx, image in enumerate(image_list):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if debug:
            debug_image_1 = cv2.aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 255, 0))
            debug_image_1 = cv2.resize(debug_image_1, (900, 675))
            cv2.imshow('img', debug_image_1)
            cv2.waitKey(0)

        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

            if ret:
                all_corners.append(corners)
                all_ids.append(ids)
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

        # Festlegen der Bildgröße beim ersten Bild
        if img_size is None:
            img_size = gray.shape[::-1]
        print(f"Image {idx}: {len(corners)} ArUco markers found")

    # Kalibrierung durchführen
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # Reprojektionsfehler berechnen
    total_error = 0
    total_points = 0
    berechenbar = False

    for i in range(len(all_charuco_corners)):
        print(f"Image {i}: {len(all_charuco_corners[i])} Charuco corners")
        if len(all_charuco_corners[i]) == 40:
            img_points_2, _ = cv2.projectPoints(board.getChessboardCorners(), rvecs[i], tvecs[i], camera_matrix,
                                                dist_coeffs)
            error = cv2.norm(all_charuco_corners[i], img_points_2, cv2.NORM_L2)
            total_error += error ** 2
            total_points += len(all_charuco_corners[i])
            berechenbar = True
        else:
            print(f"Image {i} has not enough Charuco corners for reprojection error calculation.")
    if berechenbar:
        reprojection_error = np.sqrt(total_error / total_points)
    else:
        reprojection_error = None

    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error




