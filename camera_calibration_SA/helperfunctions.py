from operator import index
import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def charuco_solve_pnp_coord_change(images_cam, int_mtx, dist, cam_id, num_cols, num_rows, squar_len, marker_len, pnp_method='EPNP'):
    #print(cv2.__version__)
    # Erstelle das Dictionary und das Charuco-Board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((num_cols, num_rows), squar_len, marker_len, dictionary)
    if num_cols == 9:
        board.setLegacyPattern(True)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    points_cam = []
    all_ids = []
    obj_points = []  # Liste zur Speicherung der 3D-Objektpunkte

    undistorted = cv2.undistort(images_cam, int_mtx, dist)
    corners, ids, _ = detector.detectMarkers(undistorted)
    print(f'Anzahl Ecken Kamera_{cam_id}: {len(corners)} | Anzahl IDs Kamera_{cam_id}: {len(ids)}')

    if len(corners) > 0:
        ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            corners, ids, undistorted, board, cameraMatrix=int_mtx, distCoeffs=dist
        )
        if ret is not None and len(charucoCorners) > 0:
            points_cam.append(charucoCorners)
            all_ids.append(charucoIds)
            # get object points for found charuco corners
            obj_points.append(board.getChessboardCorners()[all_ids[0].flatten()])

    if len(points_cam) == 0:
        print("Keine gültigen Charuco-Eckpunkte erkannt.")
        return None

    # Zeichne die erkannten Ecken und Marker
    pattern_with_detected_points = cv2.aruco.drawDetectedCornersCharuco(undistorted, charucoCorners, charucoIds)
    pattern_with_detected_points = cv2.aruco.drawDetectedMarkers(pattern_with_detected_points, corners, ids)

    # Initialisiere rvec und tvec
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners, charucoIds, board, int_mtx, dist, rvec, tvec
    )
    # calculate the reprojection error
    reprojection_errors = []  # Liste zur Speicherung der Reprojektion-Fehler für jedes Bild

    for i, charucoCorners in enumerate(points_cam):
        # Hole die entsprechenden Objektpunkte und Bildpunkte
        object_points = obj_points[i].reshape(-1, 3)  # 3D Objektpunkte für die Charuco-Ecken
        image_points = charucoCorners.reshape(-1, 2)  # 2D Bildpunkte für die Charuco-Ecken

        # Verwende cv2.projectPoints, um die 3D-Punkte in das Bild zu projizieren
        _, rvec, tvec = cv2.solvePnP(object_points, image_points, int_mtx, dist)
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, int_mtx, dist)

        # Berechne den Reprojektion-Fehler
        error = np.sqrt(np.sum((image_points - projected_points.reshape(-1, 2)) ** 2, axis=1))
        mean_error = np.mean(error)
        reprojection_errors.append(mean_error)

    # Ausgabe des durchschnittlichen Reprojektion-Fehlers für alle Bilder
    overall_error = np.mean(reprojection_errors)
    print(f'Durchschnittlicher Reprojektion-Fehler: {overall_error:.2f} Pixel')




    if success:
        # Zeichne die Achsen für die Pose des Charuco-Boards
        cv2.drawFrameAxes(pattern_with_detected_points, int_mtx, dist, rvec, tvec, 0.1)  # Länge der Achsen: 10 cm
        cv2.imwrite(f'charuco_corners_{cam_id}.png', pattern_with_detected_points)
        x,y,z, trans_vec_inv = invert_transformation(rvec, tvec)
        return [x, y, z],trans_vec_inv, [charucoCorners,charucoIds,overall_error]
        #print(f"Euler-Winkel (in Grad): X: {x}, Y: {y}, Z: {z}")
        #print(f"Transvektor (in cm): {trans_vec_inv}")



    # Konvertiere Punkte in das richtige Format
    img_points = np.array(points_cam).reshape(-1, 2).astype('float32')
    obj_points = np.array(obj_points).reshape(-1, 3).astype('float32')

    print('Anzahl img_points:', len(img_points))
    print('Anzahl obj_points:', len(obj_points))

    # Überprüfe, ob genügend Punkte vorhanden sind
    if len(img_points) >= 4 and len(obj_points) >= 4:
        # Mapping der Methoden-Strings zu OpenCV-Flags
        method_flags = {
            'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
            'EPNP': cv2.SOLVEPNP_EPNP,
            'P3P': cv2.SOLVEPNP_P3P,
            'DLS': cv2.SOLVEPNP_DLS,
            'UPNP': cv2.SOLVEPNP_UPNP,
            'AP3P': cv2.SOLVEPNP_AP3P,
            'IPPE': cv2.SOLVEPNP_IPPE,
            'IPPE_SQUARE': cv2.SOLVEPNP_IPPE_SQUARE,
            'RANSAC': None  # Spezielle Behandlung für solvePnPRansac
        }

        method = pnp_method.upper()
        if method == 'RANSAC':
            # Verwende solvePnPRansac
            ret, R, T, inliers = cv2.solvePnPRansac(obj_points, img_points, int_mtx, dist)
        else:
            flag = method_flags.get(method, cv2.SOLVEPNP_ITERATIVE)
            ret, R, T = cv2.solvePnP(obj_points, img_points, int_mtx, dist, flags=flag)

        print(f"Erfolg: {ret}")



        if ret:
            # Konvertiere die Rotationsmatrix in einen Rotationsvektor
            x,y,z, trans_vec_inv = invert_transformation(R, T)
            print(f"Euler-Winkel (in Grad): X: {x}, Y: {y}, Z: {z}")

            return [x, y, z], trans_vec_inv, charucoCorners
        else:
            print("solvePnP fehlgeschlagen.")
            return None
    else:
        print("Nicht genügend Punkte für solvePnP gefunden.")
        return None


def invert_transformation(R, T):
    # Konvertiere die Rotationsmatrix in einen Rotationsvektor
    rot_matrix, _ = cv2.Rodrigues(R)  # Konvertiere in Rotationsmatrix

    # Invertiere die Rotationsmatrix
    rot_matrix_inv = rot_matrix.T

    # Invertiere den Translationsvektor
    trans_vec_inv = -np.dot(rot_matrix_inv, T)

    # Konvertiere die invertierte Rotationsmatrix zurück in einen Rotationsvektor
    rot_vec_inv, _ = cv2.Rodrigues(rot_matrix_inv)

    # Umwandlung der invertierten Rotationsmatrix in Euler-Winkel
    sy = np.sqrt(rot_matrix_inv[0, 0] ** 2 + rot_matrix_inv[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rot_matrix_inv[2, 1], rot_matrix_inv[2, 2])
        y = np.arctan2(-rot_matrix_inv[2, 0], sy)
        z = np.arctan2(rot_matrix_inv[1, 0], rot_matrix_inv[0, 0])
    else:
        x = np.arctan2(-rot_matrix_inv[1, 2], rot_matrix_inv[1, 1])
        y = np.arctan2(-rot_matrix_inv[2, 0], sy)
        z = 0

    # Umwandlung in Grad (optional)
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)

    return x, y, z, trans_vec_inv


def euler_to_rotation_matrix(R):
    # Konvertiere die Winkel von Grad zu Radiant
    x = np.radians(R[0])
    y = np.radians(R[1])
    z = np.radians(R[2])

    # Rotationsmatrix für die X-Achse
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    # Rotationsmatrix für die Y-Achse
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    # Rotationsmatrix für die Z-Achse
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    # Kombiniere die Rotationsmatrizen: R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def plot_camera_in_3d(ax, R, t, scale, camera_name=None,color = 'blue'):
        # Define the points of a camera model in its local coordinate system
        # A simple pyramid shape with apex at the origin (camera center)
        # and the base forming the image plane.
        pyramid_points = np.array([
            [0, 0, 0],  # Camera center (apex of the pyramid)
            [-1, -1, 2],  # Bottom-left of the image plane
            [1, -1, 2],  # Bottom-right of the image plane
            [1, 1, 2],  # Top-right of the image plane
            [-1, 1, 2]  # Top-left of the image plane
        ]) * scale

        # Transform pyramid points using the rotation and translation
        transformed_points = (R @ pyramid_points.T).T + t

        # Extract the vertices for the pyramid
        vertices = [
            [transformed_points[0], transformed_points[1], transformed_points[2]],  # Side 1
            [transformed_points[0], transformed_points[2], transformed_points[3]],  # Side 2
            [transformed_points[0], transformed_points[3], transformed_points[4]],  # Side 3
            [transformed_points[0], transformed_points[4], transformed_points[1]],  # Side 4
            [transformed_points[1], transformed_points[2], transformed_points[3], transformed_points[4]]
            # Base (image plane)
        ]

        # Plot the camera as a collection of triangles (pyramid sides)
        ax.add_collection3d(Poly3DCollection(vertices, color=color, edgecolor='black', linewidths=1, alpha=0.5))

        # Also plot the camera center
        if camera_name:
            ax.scatter(*t.flatten(), color=color, s=50)
            ax.text(*t.flatten(), f'  {camera_name}', color=color, zdir=(0, 0, 0))

        ax.scatter(*t.flatten(), color=color, s=50)


def perform_stereo_triangulation(points1, points2, P1, P2):
    """
    Führt die Stereo-Triangulation auf zwei Sets von Punkten mit gegebenen Projektionsmatrizen aus.
    """

    # Umwandlung in homogene Koordinaten
    def DLT(P1, P2, point1, point2):
        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))


        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    for uv1, uv2 in zip(points1, points2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)

    return p3ds


def convert_to_homogeneous(points):
    """
    Konvertiert 2D-Punkte in homogene Koordinaten.
    """
    return np.vstack([points.T, np.ones(points.shape[0])])