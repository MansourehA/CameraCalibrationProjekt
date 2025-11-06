import os, re, glob, json
import numpy as np
import cv2

# ========================= Pfade anpassen =========================
IMAGE_DIR    = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\pvc_grau\000369930112"  # RAW-Ordner (NICHT 'undistorted')
RGB_INTR_DIR = r"D:\CameraCalibrationProjekt\camera_calibration\CalibrationResults_RGB"
TOF_INTR_DIR = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112\CalibrationResults_TOF_7"
SAVE_DIR     = os.path.join(IMAGE_DIR, "CalibrationResults_Extrinsic_RGB_TOF_RAW")

# === Falls die Intrinsic-Kalibrierung mit einer anderen Auflösung gemacht wurde,
#     kannst du sie hier eintragen. Wenn None, wird NICHT skaliert.
RGB_CALIB_SIZE = None        # z.B. (1920,1080)  (Breite,Höhe)
TOF_CALIB_SIZE = None        # z.B. (1024,1024) (Breite,Höhe)

# ========================= Board-Parameter =========================
BOARD_COLS = 10
BOARD_ROWS = 7
CHECKER_MM = 80.0            # <- Wenn du gemessen hast: hier realen Wert eintragen!
MARKER_MM  = 60.0
DICT_NAME  = cv2.aruco.DICT_4X4_250

# Qualitätsfilter
MIN_CHARUCO = 10             # min. ChArUco-Ecken pro Bild
MIN_PAARE   = 6              # min. gültige Paare, um Ergebnis zu bilden

# ========================= Hilfsfunktionen =========================
def find_file_by_patterns(folder, patterns):
    for pat in patterns:
        hits = glob.glob(os.path.join(folder, pat))
        if hits:
            hits.sort(key=lambda p: (len(os.path.basename(p)), p.lower()))
            return hits[0]
    return None

def load_intrinsics_any(folder):
    Kp = find_file_by_patterns(folder, ["camera_matrix.npy","K.npy","*camera*matrix*.npy","*K*.npy"])
    Dp = find_file_by_patterns(folder, ["dist_coeffs.npy","distortion_coeffs.npy","*dist*.npy","*D*.npy"])
    if Kp and Dp:
        print(f"[OK] Intrinsics (NPY) aus:\n  K: {Kp}\n  D: {Dp}")
        return np.load(Kp), np.load(Dp)
    Jp = find_file_by_patterns(folder, ["*.json"])
    if Jp:
        with open(Jp,"r") as f: data = json.load(f)
        K = np.array(data.get("camera_matrix", data.get("K")), dtype=np.float64)
        D = np.array(data.get("distortion_coeffs", data.get("dist_coeffs", data.get("D"))), dtype=np.float64).reshape(1,-1)
        if K.size and D.size:
            print(f"[OK] Intrinsics (JSON) aus:\n  {Jp}")
            return K, D
    raise FileNotFoundError(f"Keine Intrinsics in {folder} gefunden.")

def scale_K(K, from_size, to_size):
    """Skaliert K von ursprünglicher Kalibriergröße (w,h) auf aktuelle Bildgröße (w,h)."""
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    K2 = K.copy().astype(np.float64)
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2

def last_int_in_name(path):
    nums = re.findall(r'(\d+)', os.path.basename(path))
    return int(nums[-1]) if nums else -1

def collect_pairs(img_dir):
    # RGB RAW: color_#.png (aus deinem Aufnahme-Script)
    rgb_files = sorted(glob.glob(os.path.join(img_dir, "color_*.png")),
                       key=last_int_in_name)

    # IR RAW: ir_#.png (8-bit Norm) ODER ir_#_raw.npy (uint16)
    ir_png = glob.glob(os.path.join(img_dir, "ir_*.png"))
    ir_npy = glob.glob(os.path.join(img_dir, "ir_*_raw.npy"))

    ir_map = {}
    for p in ir_png:
        idx = last_int_in_name(p)
        # PNG bevorzugen
        if idx not in ir_map:
            ir_map[idx] = ("png", p)
    for p in ir_npy:
        idx = last_int_in_name(p)
        if idx not in ir_map:
            ir_map[idx] = ("npy", p)

    pairs = []
    for prgb in rgb_files:
        idx = last_int_in_name(prgb)
        if idx in ir_map:
            pairs.append((prgb, ir_map[idx], idx))
    return pairs

def preprocess_ir_raw_to_gray(tag_path):
    """IR-Rohdaten (uint16 .npy) oder PNG in 8-bit-Gray mit leichtem Kontrastboost wandeln."""
    tag, path = tag_path
    if tag == "png":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        if gray.dtype == np.uint16:
            gray = cv2.convertScaleAbs(gray, alpha=255.0/max(1, int(gray.max())))
        gray = gray.astype(np.uint8)
    else:
        arr = np.load(path)  # uint16
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo: lo, hi = 0, float(arr.max() or 1)
        gray = np.clip((arr - lo) * (255.0/(hi-lo)), 0, 255).astype(np.uint8)

    # etwas robuster für IR
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def preprocess_rgb_to_gray(rgb_bgr):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    # leichte, symmetrische Vorverarbeitung wie IR (erhöht Konsistenz)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def make_board_and_detector():
    dic = cv2.aruco.getPredefinedDictionary(DICT_NAME)
    board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS),
                                   CHECKER_MM/1000.0, MARKER_MM/1000.0, dic)
    if hasattr(board, "setLegacyPattern"): board.setLegacyPattern(True)

    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 5
    p.adaptiveThreshConstant = 7
    detector = cv2.aruco.ArucoDetector(dic, p)
    return board, detector

def detect_charuco(gray, board, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n = 0; ch_c=None; ch_id=None
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        retval, ch_c, ch_id = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if retval and ch_c is not None and ch_id is not None:
            n = len(ch_id)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_id)
    return ch_c, ch_id, vis, n

def pose_from_charuco(K, dist, ch_c, ch_id, board):
    obj = board.getChessboardCorners()[ch_id.flatten()]
    img = ch_c.reshape(-1,2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None, None
    R,_ = cv2.Rodrigues(rvec)
    return R, tvec

def se3_median(Rs, ts):
    rvecs = [cv2.Rodrigues(R)[0].ravel() for R in Rs]
    r_med = np.median(np.stack(rvecs,0),0).reshape(3,1)
    R_med,_ = cv2.Rodrigues(r_med)
    t_med = np.median(np.stack(ts,0),0).reshape(3,1)
    return R_med, t_med

def project_debug(rgb_bgr, K, dist, board, R, t):
    obj = board.getChessboardCorners()
    rvec,_ = cv2.Rodrigues(R)
    pts,_  = cv2.projectPoints(obj, rvec, t, K, dist)
    out = rgb_bgr.copy()
    for p in pts.reshape(-1,2):
        cv2.circle(out, tuple(np.round(p).astype(int)), 3, (0,0,255), -1)
    return out

# ========================= Hauptteil =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Intrinsics laden
    K_rgb, D_rgb = load_intrinsics_any(RGB_INTR_DIR)
    K_tof, D_tof = load_intrinsics_any(TOF_INTR_DIR)

    board, detector = make_board_and_detector()
    pairs = collect_pairs(IMAGE_DIR)
    print(f"{len(pairs)} Paare in {IMAGE_DIR} gefunden.")
    for p in pairs[:10]:
        print("PAIR:", os.path.basename(p[0]), "<->", os.path.basename(p[1][1]), "[idx=", p[2], "]")
    if not pairs:
        raise RuntimeError("Keine passenden Dateien gefunden.")

    # Falls Kalibrierauflösung ≠ aktuelle RAW-Auflösung, skalieren wir K passend
    # (wird pro Bild geprüft und nur bei Bedarf angewandt)
    R_list, t_list, used = [], [], 0

    for (rgb_path, ir_tagpath, idx) in pairs:
        rgb = cv2.imread(rgb_path)                      # BGR (RAW)
        irg = preprocess_ir_raw_to_gray(ir_tagpath)     # Gray (8-bit)
        if rgb is None or irg is None:
            print(f"[{idx}] Lesen fehlgeschlagen."); continue

        gray_rgb = preprocess_rgb_to_gray(rgb)

        # ggf. K auf die tatsächliche Bildgröße skalieren (nur wenn Größen bekannt sind)
        h_rgb, w_rgb = gray_rgb.shape[:2]
        h_ir,  w_ir  = irg.shape[:2]
        K_rgb_use = scale_K(K_rgb, RGB_CALIB_SIZE, (w_rgb, h_rgb)) if RGB_CALIB_SIZE else K_rgb
        K_tof_use = scale_K(K_tof, TOF_CALIB_SIZE, (w_ir,  h_ir )) if TOF_CALIB_SIZE else K_tof
        D_rgb_use, D_tof_use = D_rgb, D_tof  # RAW -> Distortion benutzen!

        # Erkennung & Debug
        cr_rgb, id_rgb, vis_rgb, n_rgb = detect_charuco(gray_rgb, board, detector)
        cr_ir,  id_ir,  vis_ir,  n_ir  = detect_charuco(irg,       board, detector)
        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_rgb_{idx:03d}.png"), vis_rgb)
        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_ir_{idx:03d}.png"),  vis_ir)

        if n_rgb < MIN_CHARUCO or n_ir < MIN_CHARUCO:
            print(f"[{idx}] zu wenige Ecken (RGB={n_rgb}, IR={n_ir})"); continue

        # PnP je Kamera (RAW -> mit K und D!)
        R_rgb, t_rgb = pose_from_charuco(K_rgb_use, D_rgb_use, cr_rgb, id_rgb, board)
        R_tof, t_tof = pose_from_charuco(K_tof_use, D_tof_use, cr_ir,  id_ir,  board)
        if R_rgb is None or R_tof is None:
            print(f"[{idx}] solvePnP fehlgeschlagen"); continue

        # Relativpose: T_rgb_tof = T_rgb_board * inv(T_tof_board)
        R_rgb_tof = R_rgb @ R_tof.T
        t_rgb_tof = t_rgb - R_rgb_tof @ t_tof

        R_list.append(R_rgb_tof)
        t_list.append(t_rgb_tof)
        used += 1

    if used < MIN_PAARE:
        raise RuntimeError(f"Nicht genug gültige Paare ({used}/{MIN_PAARE}).")

    R_med, t_med = se3_median(R_list, t_list)
    print("\nExtrinsik (RGB <- ToF) [RAW]:")
    print("R =\n", R_med)
    print("t =\n", t_med.ravel())
    print(f"|t| = {float(np.linalg.norm(t_med)):.4f} m  (Azure-Kinect-Physik ca. 0.032 m)")

    # speichern
    np.save(os.path.join(SAVE_DIR, "R_rgb_from_tof_raw.npy"), R_med)
    np.save(os.path.join(SAVE_DIR, "t_rgb_from_tof_raw.npy"), t_med)
    with open(os.path.join(SAVE_DIR, "extrinsic_rgb_from_tof_raw.json"), "w") as f:
        json.dump({"R": R_med.tolist(), "t": t_med.ravel().tolist()}, f, indent=2)

    # Reprojektionstest auf erstem RGB
    first_rgb_path = sorted(glob.glob(os.path.join(IMAGE_DIR, "color_*.png")),
                            key=last_int_in_name)[0]
    first_rgb = cv2.imread(first_rgb_path)
    # Debug-Projektion mit originalen K,D auf RAW:
    reproj = project_debug(first_rgb, K_rgb, D_rgb, board, R_med, t_med)
    cv2.imwrite(os.path.join(SAVE_DIR, "reprojection_debug_raw.png"), reproj)
    print(f"\nErgebnisse gespeichert in:\n{SAVE_DIR}")

if __name__ == "__main__":
    main()
