import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

# Set the capture frame rate to 60fps
cap.set(cv2.CAP_PROP_FPS, 60)

detector = HandDetector(maxHands=1)
classifier = Classifier("models\mymodel_18_7_17h43.h5")

offset = 20
imgSize = 300

# folder = "data/A"
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J",
          "K","L","M","N","O","P","Q","R","S","T",
          "U","V","W","X","Y","Z"]

prev_time = 0
while True:
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if wCal > 0 and imgSize > 0:
                imgReSize = cv2.resize(imgCrop, (wCal, imgSize))
                imgReSizeShape = imgReSize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgReSize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if hCal > 0 and imgSize > 0:
                imgReSize = cv2.resize(imgCrop, (imgSize, hCal))
                imgReSizeShape = imgReSize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgReSize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    k = cv2.waitKey(1)
    if k == ord('q'):  # Bấm 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
