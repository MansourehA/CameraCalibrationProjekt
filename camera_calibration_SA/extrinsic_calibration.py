import glob
import cv2
import numpy as np

def extrinsic_charuco_calibration(images_cam_1, images_cam_2, int_mtx_1, int_mtx_2, dist_1, dist_2,num_rows,num_cols,square_length,marker_length):
    print(cv2.__version__)
    # Create dictionary and Charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((num_rows, num_cols), square_length, marker_length, dictionary)
    if num_rows == 9:
        board.setLegacyPattern(True)
    #board = cv2.aruco.CharucoBoard((7,4),0.055,0.041, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    # image_board = board.generateImage((900, 600))
    # cv2.namedWindow("Charuco board")
    # cv2.imshow("Charuco board", image_board)
    # cv2.waitKey(0)

    # images_cam_1[0] = cv2.undistort(images_cam_1[0], int_mtx_1, dist_1)
    # images_cam_2[0] = cv2.undistort(images_cam_2[0], int_mtx_2, dist_2)

    points_cam_1 = []
    points_cam_2 = []
    obj_points = []  # List to store the 3D object points

    for frame1, frame2 in zip(images_cam_1, images_cam_2):
        frame1  = cv2.undistort(frame1, int_mtx_1, dist_1)
        frame2  = cv2.undistort(frame2, int_mtx_2, dist_2)

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        corners1, ids1, _ = detector.detectMarkers(frame1)
        corners2, ids2, _ = detector.detectMarkers(frame2)
        print(f'len corners cam_1 {len(corners1)} | len ids cam_1  {len(ids1)}')
        print(f'len corners cam_2 {len(corners2)} | len ids cam_2  {len(ids2)}')

        # cv2.aruco.drawDetectedMarkers(frame1, corners1, ids1)
        # # resize the window
        # cv2.namedWindow("Detected Markers", cv2.WINDOW_NORMAL)
        # cv2.imshow("Detected Markers", frame1)
        # cv2.waitKey(0)

        if ids1 is not None and ids2 is not None:
            try:
                ret1, charuco_corners1, charuco_ids1 = cv2.aruco.interpolateCornersCharuco(corners1, ids1, frame1,
                                                                                           board)
            except Exception as e:
                print(f"Fehler bei interpolateCornersCharuco cam 1: {e}")
            try:
                ret1, charuco_corners2, charuco_ids2 = cv2.aruco.interpolateCornersCharuco(corners2, ids2, frame2,
                                                                                           board)
            except Exception as e:
                print(f"Fehler bei interpolateCornersCharuco cam 2: {e}")

            if charuco_ids1 is not None and charuco_ids2 is not None:
                print('Charuco corners found')
                common_ids = np.intersect1d(charuco_ids1.flatten(), charuco_ids2.flatten())
                if len(common_ids) > 0:
                    obj_pts = []
                    img_pts1 = []
                    img_pts2 = []

                    for cid in common_ids:
                        # Index des aktuellen IDs in beiden ID-Arrays finden
                        idx1 = np.where(charuco_ids1 == cid)[0][0]
                        idx2 = np.where(charuco_ids2 == cid)[0][0]

                        # Entsprechende Bildpunkte sammeln
                        img_pts1.append(charuco_corners1[idx1][0])
                        img_pts2.append(charuco_corners2[idx2][0])

                        # Entsprechende Objektpunkte sammeln
                        obj_pts.append(board.getChessboardCorners()[cid])

                    if len(obj_pts) >= 4:
                        obj_points.append(np.array(obj_pts, dtype=np.float32))
                        points_cam_1.append(np.array(img_pts1, dtype=np.float32))
                        points_cam_2.append(np.array(img_pts2, dtype=np.float32))
                    else:
                        print(f"Weniger als 4 gemeinsame Punkte gefunden ({len(obj_pts)}). Bildpaar wird übersprungen.")
    print(f'Anzahl der gültigen Bildpaare: {len(obj_points)}')

    # Überprüfen, ob genügend Punkte für die Kalibrierung vorhanden sind
    if len(obj_points) < 1:
        raise ValueError(
            "Nicht genügend Punkte für die Kalibrierung. Stellen Sie sicher, dass Marker in beiden Kamerabildern erkannt werden.")

    # Stereo-Kalibrierungskriterien
    criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
    image_size = (images_cam_1[0].shape[1], images_cam_1[0].shape[0])  # (Breite, Höhe)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # Stereo-Kalibrierung durchführen
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, points_cam_1, points_cam_2,
        int_mtx_1, dist_1, int_mtx_2, dist_2,
        image_size, criteria=criteria, flags=flags
    )

    return R, T, ret, points_cam_1, points_cam_2

    return None, None, None

if __name__ == '__main__':

    cam_1 = '000068700312'
    cam_2 = '000436120812'

    images_cam_1 = glob.glob(rf'X:\wiese\Mansoureh\leon_swpa\extrinsisch_big\{cam_1}\*.png')
    images_cam_2 = glob.glob(rf'X:\wiese\Mansoureh\leon_swpa\extrinsisch_big\{cam_2}\*.png')

    images_cam_1 = [cv2.imread(image) for image in images_cam_1]
    images_cam_2 = [cv2.imread(image) for image in images_cam_2]

    int_mtx_1 = np.load(rf'X:\wiese\Mansoureh\leon_swpa\Result_chess_small\{cam_1}_camera_matrix.npy')
    dist_1 = np.load(rf'X:\wiese\Mansoureh\leon_swpa\Result_chess_small\{cam_1}_distortion_coeffs.npy')

    int_mtx_2 = np.load(rf'X:\wiese\Mansoureh\leon_swpa\Result_chess_small\{cam_2}_camera_matrix.npy')
    dist_2 = np.load(rf'X:\wiese\Mansoureh\leon_swpa\Result_chess_small\{cam_2}_distortion_coeffs.npy')

    # R und T geben die Lage und Orientierung der zweiten Kamera im Koordinatensystem der ersten Kamera an.
    R, T, ret = extrinsic_charuco_calibration(images_cam_1, images_cam_2, int_mtx_1, int_mtx_2, dist_1, dist_2,10,7,0.08,0.06)
    print(f'Ret: {ret}')
    print(f'R: {R}')
    print(f'T: {T}')






















