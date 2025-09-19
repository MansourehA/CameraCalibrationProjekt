import numpy as np
import os

save_dir = "X:/wiese/Mansoureh/azure_calib_parameter"
camera_dirs = ['000325420812', '000068700312', '000365930112', '000368930112', '000409930112', '000436120812']

for camera_dir in camera_dirs:
    # Lade die Datei mit allow_pickle=True, da es ein Wörterbuch-Objekt ist
    combined_data_path = os.path.join(save_dir, f'{camera_dir}_camera.npy')
    combined_data = np.load(combined_data_path, allow_pickle=True).item()  # Konvertiere das Array in ein Wörterbuch

    # Extrahiere die Kamera-Matrix und Verzerrungskoeffizienten und konvertiere sie in float64
    camera_matrix = np.array(combined_data['calibration_matrix'], dtype=np.float64)
    dist_coeffs = np.array(combined_data['distortion_coeff'], dtype=np.float64)

    # Speichere die getrennte Kamera-Matrix und Verzerrungskoeffizienten als separate Dateien
    np.save(os.path.join(save_dir, f'{camera_dir}_camera_matrix.npy'), camera_matrix)
    np.save(os.path.join(save_dir, f'{camera_dir}_distortion_coeffs.npy'), dist_coeffs)

    # Ausgabe zur Bestätigung
    print(f"Kameramatrix für {camera_dir}: \n", camera_matrix)
    print(f"Verzerrungskoeffizienten für {camera_dir}: \n", dist_coeffs)
