import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO object detection model
detection_model = YOLO("yolov8n_float32.tflite")

# Load YOLO pose estimation model
pose_model = YOLO("yolov8n-pose_float32.tflite")  # Ensure you have this model

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

# Check if webcam opens successfully
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Track start time
start_time = time.time()
max_duration = 30  # Stop after 30 seconds

# Initialize previous frame keypoints
previous_foot_y = None

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    # Run YOLO object detection model
    detection_results = detection_model(frame)

    for r in detection_results:
        im_bgr = r.plot()  # BGR-order numpy array
        
        # Loop through detected objects
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            
            # Only process if detected object is a person (Class 0 in COCO)
            if cls == 0 and conf > 0.5:  
                person_crop = frame[y1:y2, x1:x2]  # Crop person region
                
                # Run pose estimation model
                pose_results = pose_model(person_crop)

                # Extract keypoints safely
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    keypoints = pose_results[0].keypoints.xy.numpy()  # Convert keypoints to NumPy
                    
                    # Draw keypoints
                    for point in keypoints[0]:  # First person detected
                        if len(point) >= 2:
                            px, py = int(point[0]), int(point[1])
                            cv2.circle(im_bgr, (px + x1, py + y1), 5, (0, 0, 255), -1)

                    # Extract current frame keypoints
                    head_y = keypoints[0][0][1]  # Head Y coordinate
                    foot_y = keypoints[0][15][1]  # Foot Y coordinate
                    print("head_y::::::::::::::", head_y)
                    print("foot_y::::::::::::::", foot_y)
                    print(f"📊 Head_Y: {head_y} | Previous Foot_Y: {previous_foot_y}")

                    # Fall Detection: If head is close to the previous frame's foot position
                    if previous_foot_y is not None and head_y > 0 and previous_foot_y > 0 and abs(head_y - previous_foot_y) < 50:
                        cv2.putText(im_bgr, "🚨 FORWARD FALL DETECTED", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print("⚠️ Forward Fall Detected!")

                    # Update previous frame foot position
                    previous_foot_y = foot_y

    # Display output
    cv2.imshow("YOLO Forward Fall Detection", im_bgr)

    # Stop after 30 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time > max_duration:
        print("⏳ Stopping script after 30 seconds...")
        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

print("✅ Script completed.")
