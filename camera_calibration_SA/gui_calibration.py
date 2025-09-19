import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraFeed(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Charuco Detection Camera Feed")
        self.setGeometry(100, 100, 800, 600)

        # Video feed label
        self.video_label = QLabel(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # OpenCV setup for video capture
        self.capture = cv2.VideoCapture(0)  # 0 for default camera

        # Timer to update the frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms timer to simulate ~30 FPS

        # Charuco Board Setup
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.02, self.dictionary)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

            if ids is not None:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
                if charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Convert to QImage and display
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraFeed()
    window.show()
    sys.exit(app.exec_())
