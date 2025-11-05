import os, re, glob, json
import numpy as np
import cv2

# ---------------- Pfade ----------------
IMAGE_DIR    = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\pvc_grau\000369930112\undistorted"
RGB_INTR_DIR = r"D:\CameraCalibrationProjekt\camera_calibration\CalibrationResults_RGB"
TOF_INTR_DIR = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112\CalibrationResults_TOF_7"
SAVE_DIR     = os.path.join(IMAGE_DIR, "CalibrationResults_Extrinsic_RGB_TOF")

# ---------------- Board ----------------
BOARD_COLS = 10
BOARD_ROWS = 7
CHECKER_MM = 80.0
MARKER_MM  = 60.0
DICT_NAME  = cv2.aruco.DICT_4X4_250

MIN_CHARUCO = 12
MIN_PAARE   = 6

# ---------- Intrinsics laden (flexibel) ----------
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

# ---------- Dateisammlung & Paarung ----------
def last_int_in_name(path):
    nums = re.findall(r'(\d+)', os.path.basename(path))
    return int(nums[-1]) if nums else -1

def collect_pairs(img_dir):
    # RGB: alles was wie color_undistorted*.png aussieht
    rgb_files = sorted(
        glob.glob(os.path.join(img_dir, "color_undistorted*.png")),
        key=last_int_in_name
    )
    # IR: png + npy (raw)
    ir_png = glob.glob(os.path.join(img_dir, "ir_undistorted*.png"))
    ir_npy = glob.glob(os.path.join(img_dir, "ir_undistorted*raw.npy"))

    # Map: index -> (tag, path)  (PNG hat Vorrang)
    ir_map = {}
    for p in ir_png:
        ir_map[last_int_in_name(p)] = ("png", p)
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

# ---------- IR laden & aufbereiten ----------
def load_ir_gray(tag_path):
    tag, path = tag_path
    if tag == "png":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        if gray.dtype==np.uint16:
            gray = cv2.convertScaleAbs(gray, alpha=255.0/max(1,int(gray.max())))
        gray = gray.astype(np.uint8)
    else:
        arr = np.load(path)
        lo, hi = np.percentile(arr, [1,99])
        if hi<=lo: lo,hi = 0, float(arr.max() or 1)
        gray = np.clip((arr-lo)*(255.0/(hi-lo)), 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

# ---------- Board & Detector ----------
def make_board_and_detector():
    dic = cv2.aruco.getPredefinedDictionary(DICT_NAME)
    board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS),
                                   CHECKER_MM/1000.0, MARKER_MM/1000.0, dic)
    if hasattr(board, "setLegacyPattern"): board.setLegacyPattern(True)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.02
    p.adaptiveThreshWinSizeMin = 5
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 5
    detector = cv2.aruco.ArucoDetector(dic, p)
    return board, detector

def detect_charuco(gray, board, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n = 0; ch_c=None; ch_id=None
    if ids is not None and len(ids)>0:
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

def project_debug(rgb, K, dist, board, R, t):
    obj = board.getChessboardCorners()
    rvec,_ = cv2.Rodrigues(R)
    pts,_  = cv2.projectPoints(obj, rvec, t, K, dist)
    out = rgb.copy()
    for p in pts.reshape(-1,2):
        cv2.circle(out, tuple(np.round(p).astype(int)), 3, (0,0,255), -1)
    return out

# ------------------------ Main ------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    K_rgb, D_rgb = load_intrinsics_any(RGB_INTR_DIR)
    K_tof, D_tof = load_intrinsics_any(TOF_INTR_DIR)

    # Achtung: Bilder sind bereits entverzerrt -> Dist = 0
    D_rgb = np.zeros((1,5), dtype=np.float32)
    D_tof = np.zeros((1,5), dtype=np.float32)

    board, detector = make_board_and_detector()
    pairs = collect_pairs(IMAGE_DIR)
    print(f"{len(pairs)} Paare gefunden in {IMAGE_DIR}.")
    # Debug-Liste (erste 10)
    for p in pairs[:10]:
        print("PAIR:", os.path.basename(p[0]), " <-> ", os.path.basename(p[1][1]), " [idx=", p[2], "]")
    if not pairs:
        raise RuntimeError("Keine passenden Dateien gefunden.")

    R_list, t_list, used = [], [], 0
    for (rgb_path, ir_tagpath, idx) in pairs:
        rgb = cv2.imread(rgb_path)
        irg = load_ir_gray(ir_tagpath)
        if rgb is None or irg is None:
            print(f"[{idx}] Lesen fehlgeschlagen.");
            continue

        cr_rgb, id_rgb, vis_rgb, n_rgb = detect_charuco(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), board, detector)
        cr_ir,  id_ir,  vis_ir,  n_ir  = detect_charuco(irg, board, detector)

        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_rgb_{idx:03d}.png"), vis_rgb)
        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_ir_{idx:03d}.png"),  vis_ir)

        if n_rgb < MIN_CHARUCO or n_ir < MIN_CHARUCO:
            print(f"[{idx}] zu wenige Ecken (RGB={n_rgb}, IR={n_ir})");
            continue

        R_rgb, t_rgb = pose_from_charuco(K_rgb, D_rgb, cr_rgb, id_rgb, board)
        R_tof, t_tof = pose_from_charuco(K_tof, D_tof, cr_ir,  id_ir,  board)
        if R_rgb is None or R_tof is None:
            print(f"[{idx}] solvePnP fehlgeschlagen");
            continue

        R_rgb_tof = R_rgb @ R_tof.T
        t_rgb_tof = t_rgb - R_rgb_tof @ t_tof

        R_list.append(R_rgb_tof)
        t_list.append(t_rgb_tof)
        used += 1

    if used < MIN_PAARE:
        raise RuntimeError(f"Nicht genug gültige Paare ({used}/{MIN_PAARE}).")

    R_med, t_med = se3_median(R_list, t_list)
    print("\nExtrinsik (RGB <- ToF):")
    print("R =\n", R_med)
    print("t =\n", t_med.ravel())

    np.save(os.path.join(SAVE_DIR, "R_rgb_from_tof.npy"), R_med)
    np.save(os.path.join(SAVE_DIR, "t_rgb_from_tof.npy"), t_med)
    with open(os.path.join(SAVE_DIR, "extrinsic_rgb_from_tof.json"), "w") as f:
        json.dump({"R": R_med.tolist(), "t": t_med.ravel().tolist()}, f, indent=2)

    # Reprojektionstest
    first_rgb = cv2.imread(sorted(glob.glob(os.path.join(IMAGE_DIR, "color_undistorted*.png")),
                                  key=last_int_in_name)[0])
    reproj = project_debug(first_rgb, K_rgb, D_rgb, board, R_med, t_med)
    cv2.imwrite(os.path.join(SAVE_DIR, "reprojection_debug.png"), reproj)
    print(f"\nErgebnisse gespeichert in:\n{SAVE_DIR}")

if __name__ == "__main__":
    main()
