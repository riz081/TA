import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import time

# Webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Find Function
# x is the raw distance, y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

# Settings
offset = 20
imgSize = 300
folder = "Data/- Coba"
counter = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]

        if 'lmList' in hand and 'bbox' in hand:
            lmList = hand['lmList']
            x, y, w, h = hand['bbox']
            
            # Extract x, y coordinates only
            x1, y1 = lmList[5][:2]
            x2, y2 = lmList[17][:2]

            # Calculate the primary distance (between landmarks 5 and 17)
            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C

            # Debugging print statement
            print(f'Primary Distance: {distanceCM} cm')

            # Calculate the secondary distance (between thumb tip and index finger tip)
            x3, y3 = lmList[4][:2]
            x4, y4 = lmList[8][:2]
            secondary_distance = int(math.sqrt((y4 - y3) ** 2 + (x4 - x3) ** 2))
            secondary_distanceCM = A * secondary_distance ** 2 + B * secondary_distance + C

            # Debugging print statement
            print(f'Secondary Distance: {secondary_distanceCM} cm')

            # Display the distances on the camera feed
            cv2.putText(img, f'Primary Distance: {int(distanceCM)} cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f'Secondary Distance: {int(secondary_distanceCM)} cm', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Ensure imgCrop is valid
            if imgCrop.size == 0:
                print("Failed to crop image correctly")
                continue

            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            print("Hand detection failed or key missing in hand dictionary")
    else:
        print("No hands detected")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
        print(lmList)

cap.release()
cv2.destroyAllWindows()
