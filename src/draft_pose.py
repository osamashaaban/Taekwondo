from ultralytics import YOLO
import cv2

model = YOLO(r"D:\pose estimation\yolov8m-pose.pt")
results = model(source=r"D:\pose estimation\Blue_Red_Players.avi", conf=0.4,show=True)
print(results)