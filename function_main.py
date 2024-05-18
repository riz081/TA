import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
import os
from ImageViewer import Ui_imageViewer
from PyQt5.QtWidgets import QFileDialog

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D"]


class Function_Main():
    def __init__(self, ui: Ui_imageViewer) -> None:
        self.ui = ui
        self.init_system()

    def init_system(self):
        self.ui.pb_back.clicked.connect(self._action_pb_back)
        self.ui.pb_next.clicked.connect(self._action_pb_next)
        self.ui.tb_path.clicked.connect(self._action_tb_path)

    def _action_pb_back(self):
        print('PB Back di tekan')

    def _action_pb_next(self):
        print('PB Next di tekan')

    def _action_tb_path(self):
        self.find_folder()

    def find_folder(self):
        file_dialog = QFileDialog()
        folder_name = file_dialog.getExistingDirectory(
            None, 'Select Folder', os.getcwd())
        print(folder_name)
        self.ui.le_path.setText(folder_name)
