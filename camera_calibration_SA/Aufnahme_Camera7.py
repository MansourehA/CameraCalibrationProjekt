import os
import numpy as np
import sys
from matplotlib import pyplot as plt
from pyk4a import PyK4A, Config
import pyk4a
import cv2
import time

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.Code.helperfunctions import read_config
class KinectCamera:
    def __init__(self, device_id):
        self.config = read_config('kinect.yaml')
        self.device_id = device_id
        self.azure_config = Config(
            color_resolution=pyk4a.ColorResolution(self.config['ColorResolution']),
            color_format=pyk4a.ImageFormat(self.config['ImageFormat']),
            depth_mode=pyk4a.DepthMode(self.config['DepthMode']),
            camera_fps=pyk4a.FPS(self.config['FPS']),
        )
        self.k4a_cam = PyK4A(config=self.azure_config, device_id=self.device_id)
        self.k4a_cam.open()
        self.serialnumber = self.k4a_cam.serial
        self.k4a_cam.close()

    def get_image(self):
        self.k4a_cam.open()
        self.k4a_cam.start()
        image = self.k4a_cam.get_capture()
        color_image = image.color
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
        print(f"Image captured at {image.color_timestamp_usec} μs")
        self.k4a_cam.stop()
        return color_image


if __name__ == '__main__':
    time.sleep(5)
    target_serial = "000369930112"
    num_devices_to_scan = 7  # Adjust based on expected max devices
    save_path = r'W:/CameraCalibration/Camera7_Captures'

    found = False
    for device_id in range(num_devices_to_scan):
        try:
            cam = KinectCamera(device_id)
            if cam.serialnumber == target_serial:
                found = True
                print(f"Found camera {target_serial} at device ID {device_id}")
               # time.sleep(2)
                image = cam.get_image()

                save_folder = os.path.join(save_path, target_serial)
                os.makedirs(save_folder, exist_ok=True)
                img_count = len(os.listdir(save_folder))
                filename = f'image_{target_serial}_{img_count + 1}.png'

                plt.imshow(image)
                plt.title(f"Camera {target_serial}")
                plt.axis('off')
                plt.imsave(os.path.join(save_folder, filename), image)
                plt.show()
                break
        except Exception as e:
            continue

    if not found:
        print(f"Camera with serial number {target_serial} not found.")
