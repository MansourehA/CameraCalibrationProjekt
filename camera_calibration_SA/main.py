import numpy as np
import pandas as pd
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

mpl.use('Qt5Agg')
from extrinsic_calibration import extrinsic_charuco_calibration
from intrinsic_calibration import calibrate_with_charuco
import helperfunctions as hf


# Definieren Sie Ihre Kamera-IDs
cam_ids = ['000365930112', '000409930112', '000368930112', '000436120812', '000325420812','000068700312']
colors_for_cams = ['dimgray', 'firebrick','darkorange','lightgreen','royalblue','darkviolet']
stereo_calibrate = True  # Setzen Sie dies auf True, um die Kalibrierung durchzuführen
if stereo_calibrate:
    # Initialisieren des Ergebnis-Dictionarys
    # Generieren aller möglichen Kamerapaar-Kombinationen
    combinations = list(itertools.combinations(cam_ids, 2))
    for main_cam_id in cam_ids:
        results = {
            'cam_1': [],
            'cam_2': [],
            'R': [],
            'T': [],
            'ret': [],
            'int_mtx_1': [],
            'int_mtx_2': [],
            'dist_1': [],
            'dist_2': []
        }
        for sub_cam in cam_ids:
            if main_cam_id == sub_cam:
                continue
            cam_1 = main_cam_id
            cam_2 = sub_cam
            print(f'Kombination: {cam_1} & {cam_2}')

            # Laden der Bilder für beide Kameras
            images_cam_1_paths = glob.glob(rf'\\130.75.27.111\swap\wiese\Mansoureh\leon_swpa\extrinsisch_small\{cam_1}\*.png')
            images_cam_2_paths = glob.glob(rf'\\130.75.27.111\swap\wiese\Mansoureh\leon_swpa\extrinsisch_small\{cam_2}\*.png')
            #\\130.75.27.111\alamifar\CameraCalibration\utils\camera_calibration\

            cam_image_error = []
            ret_low = 1000
            best_R = None
            best_T = None

            for o in range(len(images_cam_1_paths)):
                images_cam_1 = [cv2.imread(image) for image in images_cam_1_paths[o:o+1]]
                images_cam_2 = [cv2.imread(image) for image in images_cam_2_paths[o:o+1]]

                # Laden der intrinsischen Parameter
                int_mtx_1 = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_big\{cam_1}_camera_matrix.npy')
                dist_1 = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_big\{cam_1}_distortion_coeffs.npy')
                int_mtx_2 = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_big\{cam_2}_camera_matrix.npy')
                dist_2 = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_big\{cam_2}_distortion_coeffs.npy')

                try:
                    R, T, ret,points_cam1,points_cam2  = extrinsic_charuco_calibration(
                        images_cam_1, images_cam_2,
                        int_mtx_1, int_mtx_2,
                        dist_1, dist_2,
                        9, 6, 0.065, 0.048
                    )
                    print(f'Ret: {ret}')
                    print(f'R: {R}')
                    print(f'T: {T}')

                    if ret < ret_low:
                        ret_low = ret
                        best_R = R
                        best_T = T
                        best_int_mtx_1 = int_mtx_1
                        best_int_mtx_2 = int_mtx_2
                        best_dist_1 = dist_1
                        best_dist_2 = dist_2
                except Exception as e:
                    print(f'Kalibrierung zwischen {cam_1} und {cam_2} fehlgeschlagen: {e}')
                    continue
                cam_image_error.append(ret)

            # Speichern der besten Ergebnisse für dieses Kamerapaar
            if best_R is not None and best_T is not None:
                print(f'Beste Kalibrierung zwischen {cam_1} und {cam_2}')
                results['cam_1'].append(cam_1)
                results['cam_2'].append(cam_2)
                results['R'].append(best_R)
                results['T'].append(best_T)
                results['ret'].append(ret_low)
                results['int_mtx_1'].append(best_int_mtx_1)
                results['int_mtx_2'].append(best_int_mtx_2)
                results['dist_1'].append(best_dist_1)
                results['dist_2'].append(best_dist_2)
            else:
                print(f'Keine erfolgreiche Kalibrierung zwischen {cam_1} und {cam_2}')

            # Plotten des Reprojektionfehlers für dieses Kamerapaar
            #plt.figure()
            #plt.plot(cam_image_error)
            #plt.xlabel('Bildindex')
            #plt.ylabel('Reprojektion Fehler')
            #plt.title(f'Reprojektion Fehler für Kamerapaar {cam_1} & {cam_2}')
            #plt.show()

        # Speichern der Ergebnisse
        np.save(f'stereo_calib_results_charuco_big/stereo_calib_results_{main_cam_id}.npy', results)

############### Plotten der Kameras in 3D ###################
# usabele_cam_ids = range(23)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X /m')
ax.set_ylabel('Y /m')
ax.set_zlabel('Z /m')
ax.set_xlim(-2, 2)
ax.set_xticks([-2, 0, 2])
ax.set_ylim(-2, 2)
ax.set_yticks([-2, 0, 2])
ax.set_zlim(-2, 2)
ax.set_zticks([-2, 0, 2])
ax.view_init(elev=-160, azim=-120)
ax.set_title('Kamerapositionen')
#plotten cameras im welt koordinaten system (Ausgehend von dem Charucoboard)
calib_dict = {'cam_id':  [], 'R_charuco':[], 'T_charuco':[], 'int_mtx':[], 'dist':[]}
for cam_id in cam_ids:
    color = colors_for_cams[cam_ids.index(cam_id)]
    image = cv2.imread(rf'\\130.75.27.111\swap\wiese\tmp\charuco_calib_io_A0_8\charuco_6_9_80mm_dict_4x4_60mm_sn_{cam_id}.png')
    int_mtx = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_small\{cam_id}_camera_matrix.npy')
    dist = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_small\{cam_id}_distortion_coeffs.npy')
    R, t, camera_points = hf.charuco_solve_pnp_coord_change(image, int_mtx, dist, cam_id,
                                                            num_cols=10, num_rows=7, squar_len=0.08, marker_len=0.0486)
    R = hf.euler_to_rotation_matrix(R)
    hf.plot_camera_in_3d(ax, R, t.T, scale=0.1, camera_name=cam_id,
                         color=color)  # (ax, R, t, scale, camera_name=None
    calib_dict['cam_id'].append(cam_id)
    calib_dict['R_charuco'].append(R)
    calib_dict['T_charuco'].append(t.T)
    calib_dict['int_mtx'].append(int_mtx)
    calib_dict['dist'].append(dist)
    print(f'Kamera {cam_id} erfolgreich geplottet -> {t.T}')

plt.savefig(f'Kamerapositionen.svg')
#plt.show()
z_all = []
for i in range(1,20):
    z = []
    # überprüfen welchen einfluss die wahl der main Kamera auf die ergebnisse hat
    main_cam_id = '000365930112'
    usable_cam_ids = [1]
    point_dict = {'cam_id': [], 'points_cam': [], 'charuco_ids': [],'error':[]}

    for oo in usable_cam_ids:
        for cam_id in cam_ids:
            color = colors_for_cams[cam_ids.index(cam_id)]
            try:
                print(f'Plotten der Kamera {cam_id}')
                # images = glob.glob(rf'X:\wiese\Mansoureh\Aufnahme161024\{cam_id}\*.png')
                # image = cv2.imread(images[oo])
                # Hier werden die Extrinsischen Matrizen ermittelt! Wählen zwischen small u. big und vergleichen.
                image = cv2.imread(rf'\\130.75.27.111\swap\wiese\Mansoureh\leon_swpa\extrinsisch_small\{cam_id}\image_{cam_id}.png')
                int_mtx = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_small\{cam_id}_camera_matrix.npy')
                dist = np.load(rf'\\130.75.27.111\swap\wiese\Mansoureh\Result_Aruco_small\{cam_id}_distortion_coeffs.npy')
                R,t,camera_points = hf.charuco_solve_pnp_coord_change(image, int_mtx, dist,cam_id,
                                                                      num_cols=9,num_rows=6,squar_len=0.065,marker_len=0.048)#
                R = hf.euler_to_rotation_matrix(R)

                point_dict['cam_id'].append(cam_id)
                point_dict['points_cam'].append(camera_points[0])
                point_dict['charuco_ids'].append(camera_points[1])
                point_dict['error'].append(camera_points[2])
                # plot marker points in 3d
                # calculate marker points in 3d
            except Exception as e:
                print(f'Fehler bei der Kamera {cam_id}: {e}')
                continue
    #plt.savefig(f'Kamerapositionen.svg')
    #################### plotten der Marker in 3D ####################
    # point_dict in pandas dataframe umwandeln
    df_calib_to_charuco = pd.DataFrame(calib_dict)
    main_stereo_calib = np.load(f'stereo_calib_results_charuco_small/stereo_calib_results_{main_cam_id}.npy', allow_pickle=True).item()
    main_stereo_calib = pd.DataFrame(main_stereo_calib)
    main_cam_calib_to_charuco = df_calib_to_charuco[df_calib_to_charuco['cam_id'] == main_cam_id]
    detected_points_dict = pd.DataFrame(point_dict)
    main_cam_points = detected_points_dict[detected_points_dict['cam_id'] == main_cam_id]
    points_main_cam = main_cam_points['points_cam'].values[0]
    found_ids_main_cam = main_cam_points['charuco_ids'].values[0]
    R_main_to_charuco = main_cam_calib_to_charuco['R_charuco'].values[0]
    T_main_to_charuco = main_cam_calib_to_charuco['T_charuco'].values[0]
    for cam_sn in cam_ids:
        color = colors_for_cams[cam_ids.index(cam_sn)]
        if cam_sn == main_cam_id:
            continue
        try:
            cam_calib = df_calib_to_charuco[df_calib_to_charuco['cam_id'] == cam_sn]
            # compare the found ids with the main camera
            detected_points_dict_cam = detected_points_dict[detected_points_dict['cam_id'] == cam_sn]
            points_cam = detected_points_dict_cam['points_cam'].values[0]
            found_ids_cam = detected_points_dict_cam['charuco_ids'].values[0]
            error_cam = detected_points_dict_cam['error'].values[0]
            # find the common ids
            common_ids = np.intersect1d(found_ids_main_cam, found_ids_cam)
            # select the common points
            matching_points_main_cam = points_main_cam[np.where(np.isin(found_ids_main_cam, common_ids))]
            matching_points_cam_tmp = points_cam[np.where(np.isin(found_ids_cam, common_ids))]
            # stereo triangulation of points in 3d
            P1 = main_cam_calib_to_charuco['int_mtx'].values[0] @ np.hstack((np.eye(3), np.zeros((3, 1))))
            R_to_main = main_stereo_calib[main_stereo_calib['cam_2'] == cam_sn]['R'].values[0]
            T_to_main = main_stereo_calib[main_stereo_calib['cam_2'] == cam_sn]['T'].values[0]
            P2 = cam_calib['int_mtx'].values[0] @ np.hstack((R_to_main, T_to_main))
            matching_points_main_cam = hf.convert_to_homogeneous(matching_points_main_cam).T
            matching_points_cam_tmp = hf.convert_to_homogeneous(matching_points_cam_tmp).T

            points_3d = hf.perform_stereo_triangulation(matching_points_main_cam, matching_points_cam_tmp, P1, P2)
            for i in range(len(points_3d)):
                 #color, based on error
                if error_cam < 0.5:
                    color = 'green'
                elif error_cam < 0.99:
                   color = 'yellow'
                else:
                    color = 'red'
                points_base_3d = np.dot(R_main_to_charuco, points_3d[i].T) + T_main_to_charuco
                ax.scatter(points_base_3d[0][0],points_base_3d[0][1],points_base_3d[0][2],color=color)
                z.append(points_base_3d[0][2])
        except Exception as e:
            print('ERROR',e)
            continue
    print(f'Z: {np.mean(z)}')
    z_all.append(np.mean(z))

ax.set_title('Board Positionen')
plt.savefig(f'Board_positionen.svg')
fig_2, ax_2 = plt.subplots()
ax_2.plot(z_all)
plt.show()








