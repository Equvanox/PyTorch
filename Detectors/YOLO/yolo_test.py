# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

# Load a pretrained model (YOLOv8 nano/s/m/l/x)
model = YOLO("yolov8n.pt")   # uses weights from ultralytics hub

# Run inference
results = model("D:\VisualStudioCode\Python\AI\DeepLearning\Pytorch\Detectors\YOLO\Screenshot 2025-12-17 125515.png", imgsz=640, conf=0.25)  # returns Results object

# Draw and save
img_with_boxes = results[0].plot()  # returns numpy image with boxes
cv2.imwrite("yolo_out.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
print("Saved yolo_out.jpg")
