import cv2
import numpy as np


image = cv2.imread(r'X:\wiese\Mansoureh\leon_swpa\extrinsisch_small\000325420812\image_000325420812_10.png')

# detect markers

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, parameters)

corners, ids, _ = detector.detectMarkers(image)

print(f'Number of corners: {len(corners)}')

# draw markers
image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
cv2.imwrite('detected_markers_1.png', image)

