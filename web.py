from flask import Flask, render_template, Response
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
import os

app = Flask(__name__)

detector = HandDetector(maxHands=2)
model = os.path.join('Model/keras_model.h5')
classifier = Classifier(model, "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D"]


def generate_frames():
    # Membuka kamera
    cap = cv2.VideoCapture(0)

    while True:
        # Membaca frame dari kamera
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prdiction, index = classifier.getPrediction(
                    imgWhite, draw=False)
                print(prdiction.index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prdiction, index = classifier.getPrediction(
                    imgWhite, draw=False)

            cv2.putText(imgOutput, labels[index], (x, y-20),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 255, 0), 4)

        if not success:
            break
        else:
            # Mengubah frame menjadi format yang dapat ditampilkan di web
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            imgOutput = buffer.tobytes()

        # Menghasilkan frame untuk ditampilkan di halaman web
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
