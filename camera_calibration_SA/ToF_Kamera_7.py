from pyk4a import PyK4A, Config, ImageFormat
import cv2
import os

# Aufnahme_Ordner
save_dir = r'W:/CameraCalibration/Camera7_Captures'
os.makedirs (save_dir, exist_ok=True)

# Azure_Config
k4a = PyK4A(
    Config(
        color_resolution=None ,

    )
)
