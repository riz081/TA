import sys
from ImageViewer import Ui_imageViewer
from PyQt5.QtWidgets import QMainWindow, QApplication
from function_main import Function_Main
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kamera PyQt")
        
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.start_camera)
        self.timer.start(30)  # Set delay 30ms (33.33 fps)
    
    def start_camera(self):
        capture = cv2.VideoCapture(0)  # Mengakses kamera dengan indeks 0 (default)
        ret, frame = capture.read()
        capture.release()
        
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ubah format BGR menjadi RGB
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication([])
    mainwindow = QMainWindow()
    ui = Ui_imageViewer()
    ui.setupUi(mainwindow)
    function_main = Function_Main(ui)
    mainwindow.show()
    app.exec()
