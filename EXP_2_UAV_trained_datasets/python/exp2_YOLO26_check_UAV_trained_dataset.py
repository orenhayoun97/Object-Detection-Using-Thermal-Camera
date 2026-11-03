import cv2
import numpy as np
import os
from ultralytics import YOLO
import time

cap = cv2.VideoCapture('videos/Nightvision FPV flight Axisflying 640 thermal cam and Runcam nig.mp4')
model = YOLO('best_yolo26_2.pt')
prev_time=0.0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cTime = time.time()
    fps = 1 / (cTime - prev_time)
    prev_time = cTime
    result = model.predict(frame,conf=0.5, verbose=False)
    annotated_frame = result[0].plot()
    fps_text = f'FPS: {int(fps)}'
    cv2.putText(annotated_frame, fps_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('FPV Video', annotated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Exit on ESC
        break


cap.release()
cv2.destroyAllWindows()
