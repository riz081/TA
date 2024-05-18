import csv
import pandas as pd
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        lmlist = hands[0]['lmlist']
        print(lmlist)
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)