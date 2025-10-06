import os
import re
import glob
import cv2
import numpy as np

# ------------------------ Konfiguration ------------------------
# Board (GENAU zum Ausdruck passend!)
BOARD_COLS = 10            # Spalten (squaresX)
BOARD_ROWS = 7             # Reihen  (squaresY)
CHECKER_MM = 65.0          # Seitenlänge eines weißen/ schwarzen Felds
MARKER_MM  = 48.0          # Seitenlänge des ArUco-Markers innerhalb des Felds
DICT       = cv2.aruco.DICT_4X4_250

# Pfade
IMAGE_DIR  = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112"
RGB_INTR   = r"D:\CameraCalibrationProjekt\camera_calibration\CalibrationResults_RGB"
TOF_INTR   = r"D:\CameraCalibrationProjekt\camera_calibration\CalibrationResults_TOF"
SAVE_DIR   = os.path.join(os.path.dirname(RGB_INTR), "CalibrationResults_Extrinsic_PnP")

# Mindestpunkte / Qualitätsfilter
MIN_CHARUCO = 12     # pro Bild (RGB oder IR)
MIN_PAARE   = 6      # wie viele gültige Paare wir mindestens brauchen

# ---------------------------------------------------------------

def numeric_key(p):
    m = re.search(r'_(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0

def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS),
        CHECKER_MM / 1000.0,   # mm -> m
        MARKER_MM  / 1000.0,
        dictionary
    )
    # für OpenCV 4.7+ (gleiche Nummerierung wie Generator)
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, dictionary

def detector_tuned(dictionary):
    params = cv2.aruco.DetectorParameters()
    # Tuning (robuster für IR)
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.02
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 5
    return cv2.aruco.ArucoDetector(dictionary, params)

def preprocess_ir(ir):
    if ir is None:
        return None
    # Azure IR ist 12-bit -> stabil auf 0..4095 normalisieren
    if ir.dtype != np.uint8:
        ir8 = cv2.convertScaleAbs(ir, alpha=255.0/4095.0)
    else:
        ir8 = ir.copy()
    # Kontrast lokal steigern
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ir8 = clahe.apply(ir8)
    # leicht glätten
    ir8 = cv2.GaussianBlur(ir8, (3,3), 0)
    return ir8

def detect_charuco(gray, board, detector):
    """ChArUco erkennen + Visualisierung zurückgeben"""
    corners, ids, _ = detector.detectMarkers(gray)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    charuco_corners = None
    charuco_ids     = None
    n = 0

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if retval and ch_corners is not None and ch_ids is not None:
            n = len(ch_ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
            charuco_corners, charuco_ids = ch_corners, ch_ids

    return charuco_corners, charuco_ids, vis, n

def pose_from_charuco(K, dist, ch_corners, ch_ids, board):
    # passende 3D-Objektpunkte für genau diese IDs
    obj = board.getChessboardCorners()[ch_ids.flatten()]  # (N,3) in Metern
    img = ch_corners.reshape(-1, 2).astype(np.float32)    # (N,2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def average_poses(R_list, t_list):
    # Rotationen: über Rotationsvektoren mitteln (robust: Median)
    rvecs = [cv2.Rodrigues(R)[0].ravel() for R in R_list]
    r_med = np.median(np.stack(rvecs, axis=0), axis=0)
    R_med, _ = cv2.Rodrigues(r_med.reshape(3,1))
    # Translation: Median
    t_med = np.median(np.stack(t_list, axis=0), axis=0).reshape(3,1)
    return R_med, t_med

def project_debug(rgb_img, K_rgb, dist_rgb, board, R_rgb_tof, t_rgb_tof):
    """Projiziere einige Board-Punkte aus dem ToF in das RGB-Bild (nur zur Kontrolle)."""
    # Nehmen wir einen dichten Gitter-Satz (alle Ecken):
    obj_all = board.getChessboardCorners()
    # Welt -> ToF-Kamera -> RGB-Kamera:
    # Wir brauchen die Pose der ToF-Kamera relativ zur RGB-Kamera = (R_rgb_tof, t_rgb_tof).
    # Um Punkte aus ToF in RGB zu projizieren, genügt: rvec,tvec aus R,t.
    rvec, _ = cv2.Rodrigues(R_rgb_tof)
    img_pts, _ = cv2.projectPoints(obj_all, rvec, t_rgb_tof, K_rgb, dist_rgb)

    out = rgb_img.copy()
    for p in img_pts.reshape(-1,2):
        cv2.circle(out, tuple(np.round(p).astype(int)), 3, (0,0,255), -1)
    return out

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Intrinsik laden
    K_rgb   = np.load(os.path.join(RGB_INTR, "camera_matrix.npy"))
    dist_rgb= np.load(os.path.join(RGB_INTR, "dist_coeffs.npy"))
    K_tof   = np.load(os.path.join(TOF_INTR, "camera_matrix.npy"))
    dist_tof= np.load(os.path.join(TOF_INTR, "dist_coeffs.npy"))

    board, dictionary = make_board()
    detector = detector_tuned(dictionary)

    # Dateien numerisch paaren (color_N mit ir_N)
    rgb_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "color_*.png")), key=numeric_key)
    ir_paths  = sorted(glob.glob(os.path.join(IMAGE_DIR, "ir_*.png")),    key=numeric_key)
    n_pairs = min(len(rgb_paths), len(ir_paths))
    print(f"{n_pairs} Bildpaare gefunden.")

    R_list = []
    t_list = []
    used = 0

    for i in range(n_pairs):
        rgb = cv2.imread(rgb_paths[i])
        ir  = cv2.imread(ir_paths[i], cv2.IMREAD_UNCHANGED)

        if rgb is None or ir is None:
            continue

        gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_pre   = preprocess_ir(ir)

        # Erkennung + Speichern wie im OpenCV-Tutorial
        cr_rgb, id_rgb, vis_rgb, n_rgb = detect_charuco(gray_rgb, board, detector)
        cr_ir,  id_ir,  vis_ir,  n_ir  = detect_charuco(ir_pre,   board, detector)

        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_rgb_{i+1:03d}.png"), vis_rgb)
        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_ir_{i+1:03d}.png"),  vis_ir)

        if n_rgb < MIN_CHARUCO or n_ir < MIN_CHARUCO:
            print(f"[{i+1}] Zu wenige ChArUco-Ecken (RGB={n_rgb}, IR={n_ir}) -> übersprungen.")
            continue

        # Board-Pose in jeder Kamera
        R_rgb, t_rgb = pose_from_charuco(K_rgb, dist_rgb, cr_rgb, id_rgb, board)
        R_tof, t_tof = pose_from_charuco(K_tof, dist_tof, cr_ir,  id_ir,  board)
        if R_rgb is None or R_tof is None:
            print(f"[{i+1}] PnP fehlgeschlagen -> übersprungen.")
            continue

        # Relativpose:  T_rgb_tof = T_rgb_board * inv(T_tof_board)
        R_rgb_tof = R_rgb @ R_tof.T
        t_rgb_tof = (t_rgb - R_rgb_tof @ t_tof)

        R_list.append(R_rgb_tof)
        t_list.append(t_rgb_tof)
        used += 1

    if used < MIN_PAARE:
        raise RuntimeError(f"Nicht genug gültige Paare ({used}/{MIN_PAARE}).")

    # robuste Mitte über alle Paare
    R_med, t_med = average_poses(R_list, t_list)
    print("\nExtrinsik RGB <- ToF (Median über Paare)")
    print("R =\n", R_med)
    print("t =\n", t_med.ravel())

    # speichern
    np.save(os.path.join(SAVE_DIR, "R_rgb_from_tof.npy"), R_med)
    np.save(os.path.join(SAVE_DIR, "t_rgb_from_tof.npy"), t_med)

    # Visualisierung auf erstem RGB-Bild
    rgb0 = cv2.imread(rgb_paths[0])
    vis  = project_debug(rgb0, K_rgb, dist_rgb, board, R_med, t_med)
    cv2.imwrite(os.path.join(SAVE_DIR, "reprojection_debug.png"), vis)
    print(f"\nErgebnisse gespeichert in: {SAVE_DIR}")
    print("• detect_rgb_###.png / detect_ir_###.png (Erkennungen)\n• R_rgb_from_tof.npy, t_rgb_from_tof.npy\n• reprojection_debug.png")

if __name__ == "__main__":
    main()
