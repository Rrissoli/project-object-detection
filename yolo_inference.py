from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('models/best.pt')

results =  model.predict('Videos/corinthians-quase-gol.mp4', save=True)

print(results[0])
print("-----------------------------------------------")
for box in results[0].boxes:
    print(box)