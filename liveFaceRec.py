import threading

import cv2
import numpy as np
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
match = False

oImage = cv2.imread('images/img1.jpg')

while True:
    ret, frame = cap.read()

    if ret:
        if count % 30 == 0:
            try:
                result = DeepFace.verify(frame, oImage, model_name='Facenet', distance_metric='euclidean')
                print(result)
                match = result['verified']
            except:
                print("No face detected")

    cv2.putText(frame, str(match), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
