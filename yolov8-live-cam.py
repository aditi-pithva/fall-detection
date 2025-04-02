from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Load model
image = cv2.imread("test.jpg")  # Load image
results = model(image)  # Run inference

print(results)  # Show results
