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
def scale_K(K, from_size, to_size):
    """Skaliert Kameramatrix K von ursprünglicher Kalibriergröße auf neue Bildgröße."""
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2
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
    # Ursprüngliche Auflösungen aus den Intrinsic-Kalibrierungen
    # (Falls du sie nicht gespeichert hast, schätz sie manuell!)
    RGB_CALIB_SIZE = (1920, 1080)  # (Breite, Höhe) z.B. für Azure RGB
    TOF_CALIB_SIZE = (1024, 1024)  # (Breite, Höhe) z.B. für Azure IR

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
    R_list, t_list, used = [], [], 0
    for (rgb_path, ir_tagpath, idx) in pairs:
        rgb = cv2.imread(rgb_path)
        irg = load_ir_gray(ir_tagpath)
        if rgb is None or irg is None:
            print(f"[{idx}] Lesen fehlgeschlagen.")
            continue

        gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # --- NEU: K-Matrizen skalieren auf tatsächliche Bildgröße ---
        h_rgb, w_rgb = gray_rgb.shape[:2]
        h_ir, w_ir = irg.shape[:2]

        K_rgb_use = scale_K(K_rgb, RGB_CALIB_SIZE, (w_rgb, h_rgb))
        K_tof_use = scale_K(K_tof, TOF_CALIB_SIZE, (w_ir, h_ir))

        # Verzerrung deaktivieren (weil Bilder schon undistorted sind)
        D_rgb_use = np.zeros((1, 5), np.float32)
        D_tof_use = np.zeros((1, 5), np.float32)
        # -------------------------------------------------------------

        cr_rgb, id_rgb, vis_rgb, n_rgb = detect_charuco(gray_rgb, board, detector)
        cr_ir, id_ir, vis_ir, n_ir = detect_charuco(irg, board, detector)

        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_rgb_{idx:03d}.png"), vis_rgb)
        cv2.imwrite(os.path.join(SAVE_DIR, f"detect_ir_{idx:03d}.png"), vis_ir)

        if n_rgb < MIN_CHARUCO or n_ir < MIN_CHARUCO:
            print(f"[{idx}] zu wenige Ecken (RGB={n_rgb}, IR={n_ir})")
            continue

        # --- NEU: PnP mit skalierten Intrinsics ---
        R_rgb, t_rgb = pose_from_charuco(K_rgb_use, D_rgb_use, cr_rgb, id_rgb, board)
        R_tof, t_tof = pose_from_charuco(K_tof_use, D_tof_use, cr_ir, id_ir, board)
        # -------------------------------------------

        if R_rgb is None or R_tof is None:
            print(f"[{idx}] solvePnP fehlgeschlagen")
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


import os, glob, re, json
import numpy as np
import cv2

# === Pfade anpassen ===
IMAGE_DIR    = r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\pvc_grau\000369930112\undistorted"
SAVE_DIR     = os.path.join(IMAGE_DIR, "CalibrationResults_Extrinsic_RGB_TOF")
K_rgb = np.load(r"D:\CameraCalibrationProjekt\camera_calibration\CalibrationResults_RGB\000369930112_rgb_camera_matrix.npy")
D_rgb = np.zeros((1,5), np.float32)  # undistorted
K_tof = np.load(r"D:\CameraCalibrationProjekt\camera_calibration\Camera7_Captures\Holz\000369930112\CalibrationResults_TOF_7\000369930112_tof_camera_matrix.npy")
D_tof = np.zeros((1,5), np.float32)  # undistorted

# Board
BOARD_COLS, BOARD_ROWS = 10, 7
CHECKER_MM, MARKER_MM  = 80.0, 60.0
dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), CHECKER_MM/1000.0, MARKER_MM/1000.0, dic)
if hasattr(board, "setLegacyPattern"): board.setLegacyPattern(True)

# Extrinsik laden
R = np.load(os.path.join(SAVE_DIR, "R_rgb_from_tof.npy"))
t = np.load(os.path.join(SAVE_DIR, "t_rgb_from_tof.npy")).reshape(3,1)

def last_int(p):
    m = re.findall(r'(\d+)', os.path.basename(p));
    return int(m[-1]) if m else -1

def load_pairs():
    rgbs = sorted(glob.glob(os.path.join(IMAGE_DIR, "color_undistorted*.png")), key=last_int)
    irps = { last_int(p):("png",p) for p in glob.glob(os.path.join(IMAGE_DIR, "ir_undistorted*.png")) }
    for p in glob.glob(os.path.join(IMAGE_DIR, "ir_undistorted*raw.npy")):
        idx = last_int(p);
        if idx not in irps: irps[idx]=("npy",p)
    pairs=[]
    for prgb in rgbs:
        idx = last_int(prgb)
        if idx in irps: pairs.append((prgb, irps[idx], idx))
    return pairs

def load_ir_gray(tag_path):
    tag, path = tag_path
    if tag=="png":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        if g.dtype==np.uint16: g = cv2.convertScaleAbs(g, alpha=255.0/max(1,int(g.max())))
        g = g.astype(np.uint8)
    else:
        arr = np.load(path); lo,hi = np.percentile(arr,[1,99]);
        if hi<=lo: lo,hi = 0, float(arr.max() or 1)
        g = np.clip((arr-lo)*(255.0/(hi-lo)),0,255).astype(np.uint8)
    return g

def repro_error(gray, K, D, obj3d, img2d):
    # projiziere obj3d und vergleiche
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    # hier rvec,tvec werden nicht benutzt; wir geben proj direkt
    proj,_ = cv2.projectPoints(obj3d, rvec, tvec, K, D)  # nur zum Typziehen
    # tatsächliche Projektion wird außerhalb berechnet
    pass

def evaluate():
    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    obj_all = board.getChessboardCorners()   # (N,3) in m

    rms_rgb, rms_tof = [], []
    R_list, t_list = [], []

    pairs = load_pairs()
    print(f"{len(pairs)} Paare für Evaluation.")

    for prgb, ir_tag, idx in pairs:
        rgb = cv2.imread(prgb)
        irg = load_ir_gray(ir_tag)
        if rgb is None or irg is None: continue

        g_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # gleiche Vorverarbeitung wie zuvor (empfohlen):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        g_rgb = clahe.apply(g_rgb); g_rgb = cv2.GaussianBlur(g_rgb,(3,3),0)

        corners_rgb, ids_rgb, _ = detector.detectMarkers(g_rgb)
        corners_ir,  ids_ir,  _ = detector.detectMarkers(irg)

        if ids_rgb is None or ids_ir is None or len(ids_rgb)==0 or len(ids_ir)==0:
            continue

        ok1, ch_rgb, id_rgb = cv2.aruco.interpolateCornersCharuco(corners_rgb, ids_rgb, g_rgb, board)
        ok2, ch_ir,  id_ir  = cv2.aruco.interpolateCornersCharuco(corners_ir,  ids_ir,  irg,   board)
        if not(ok1 and ok2) or ch_rgb is None or ch_ir is None:
            continue

        # Pose je Kamera
        obj_rgb = obj_all[id_rgb.flatten()]
        obj_ir  = obj_all[id_ir.flatten()]
        ok, rvec_rgb, tvec_rgb = cv2.solvePnP(obj_rgb, ch_rgb, K_rgb, np.zeros((1,5)))
        ok, rvec_tof, tvec_tof = cv2.solvePnP(obj_ir,  ch_ir,  K_tof, np.zeros((1,5)))
        R_rgb,_ = cv2.Rodrigues(rvec_rgb); R_tof,_ = cv2.Rodrigues(rvec_tof)
        R_list.append(R_rgb @ R_tof.T); t_list.append(tvec_rgb - (R_rgb @ R_tof.T) @ tvec_tof)

        # Reprojektion RGB (Board->RGB direkt aus der PnP dieses Frames)
        proj_rgb,_ = cv2.projectPoints(obj_rgb, rvec_rgb, tvec_rgb, K_rgb, np.zeros((1,5)))
        err_rgb = cv2.norm(ch_rgb, proj_rgb, cv2.NORM_L2) / len(id_rgb)
        rms_rgb.append(float(err_rgb))

        # Reprojektion TOF->RGB via Extrinsik (Board in TOF, dann nach RGB)
        # 1) Board-Punkte im TOF des jeweiligen Frames:
        proj_ir,_ = cv2.projectPoints(obj_ir, rvec_tof, tvec_tof, np.eye(3), np.zeros((1,5)))  # 3D im TOF-Frame holen
        # 2) Diese 3D-Punkte (obj_ir) zuerst in RGB mittels globaler Extrinsik projizieren:
        rvec_RT,_ = cv2.Rodrigues(R)     # Extrinsik global
        proj_to_rgb,_ = cv2.projectPoints(obj_ir, rvec_RT, t, K_rgb, np.zeros((1,5)))
        # gegen die detektierten charuco in RGB mit gleichen IDs matchen:
        # (Achtung: Id-Schnittmenge bilden)
        ids_int = np.intersect1d(id_rgb.flatten(), id_ir.flatten())
        if len(ids_int)>0:
            idx_rgb = np.nonzero(np.in1d(id_rgb.flatten(), ids_int))[0]
            idx_ir  = np.nonzero(np.in1d(id_ir.flatten(),  ids_int))[0]
            err = cv2.norm(ch_rgb[idx_rgb], proj_to_rgb[idx_ir], cv2.NORM_L2) / len(ids_int)
            rms_tof.append(float(err))

    # Statistik
    def deg(a): return a*180.0/np.pi
    if len(R_list)>0:
        rvecs = [cv2.Rodrigues(Ri)[0].ravel() for Ri in R_list]
        r_std = np.std(np.stack(rvecs,0), axis=0)
        t_std = np.std(np.stack(t_list,0), axis=0).ravel()
        print("\nStreuung (über alle Frames):")
        print(f"Rotation-Std [deg]: {deg(r_std)}")
        print(f"Translation-Std [m]: {t_std}")

    if rms_rgb:
        print(f"\nRMS reprojection (RGB, px): mean {np.mean(rms_rgb):.3f} | std {np.std(rms_rgb):.3f}")
    if rms_tof:
        print(f"RMS reprojection (TOF→RGB via Extrinsik, px): mean {np.mean(rms_tof):.3f} | std {np.std(rms_tof):.3f}")

    # Länge von t:
    print(f"\n|t| = {np.linalg.norm(t):.4f} m")

if __name__ == "__main__":
    evaluate()
