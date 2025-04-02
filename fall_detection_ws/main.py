import cv2
import numpy as np
import asyncio
import websockets
import tensorflow as tf
from PIL import Image
from io import BytesIO
import json
import base64
import datetime
import os
from collections import deque
from enum import Enum

class PoseState(Enum):
    STANDING = 0
    FALLEN = 1
    UNKNOWN = 2

# Store last 15 pose states (~3 seconds at 5 fps)
pose_history = deque(maxlen=15)
fall_alert_triggered = False

# Load YOLO Pose TFLite model
print("Loading YOLO Pose TFLite model...")
interpreter = tf.lite.Interpreter(model_path="models/yolov8n-pose_float32-nms.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = 640
print("YOLO Pose model loaded successfully.")

CONFIDENCE_THRESHOLD = 0.2
KEYPOINT_CONF_THRESHOLD = 0.2

# Ensure keypoints_images directory exists
os.makedirs("keypoints_images", exist_ok=True)

def run_inference(image):
    image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    input_data = np.expand_dims(image_resized / 255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

def detect_fall(keypoints: np.ndarray, frame_shape):
    height, width = frame_shape[:2]

    # 1. Head vs Feet Y Displacement
    nose_y = keypoints[0][1]
    left_ankle_y = keypoints[15][1]
    right_ankle_y = keypoints[16][1]
    feet_y = max(left_ankle_y, right_ankle_y)
    vertical_displacement = feet_y - nose_y

    # 2. Shoulder-Hip Angle
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # 3. Bounding Box Aspect Ratio
    valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
    if len(valid_keypoints) == 0:
        return PoseState.UNKNOWN
    min_x, min_y = np.min(valid_keypoints[:, :2], axis=0)
    max_x, max_y = np.max(valid_keypoints[:, :2], axis=0)
    box_width = max_x - min_x
    box_height = max_y - min_y
    aspect_ratio = box_width / (box_height + 1e-6)

    # Updated conditions
    low_vertical_displacement = 0.05 < vertical_displacement < 0.3
    curled_torso = 45 < abs(angle) < 135
    compact_box = aspect_ratio < 0.7

    print(f"Vertical Displacement: {vertical_displacement:.3f}")
    print(f"Shoulder-Hip Angle: {angle:.2f}°")
    print(f"Bounding Box Aspect Ratio: {aspect_ratio:.2f}")
    print(f"Adjusted Criteria -> Low Disp: {low_vertical_displacement}, Curled Torso: {curled_torso}, Compact Box: {compact_box}")

    soft_criteria_match = sum([low_vertical_displacement, curled_torso, compact_box])
    print(f"Soft Criteria Match Count: {soft_criteria_match}/3")

    if soft_criteria_match >= 2 and abs(angle) <= 100:
        print("Pose classified as FALLEN (soft logic)")
        return PoseState.FALLEN
    else:
        print("Pose classified as STANDING")
        return PoseState.STANDING


def annotate_and_save(predictions, original_frame):
    num_detections, features = predictions.shape
    annotated_frame = original_frame.copy()
    height, width = annotated_frame.shape[:2]

    for i in range(num_detections):
        detection = predictions[i]
        score = detection[4]
        if score < CONFIDENCE_THRESHOLD:
            continue

        keypoints = detection[6:].reshape(17, 3)

        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > KEYPOINT_CONF_THRESHOLD:
                px, py = int(x * width), int(y * height)
                cv2.circle(annotated_frame, (px, py), 4, (0, 255, 255), -1)
                cv2.putText(annotated_frame, str(idx), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = f"keypoints_images/frame_{timestamp}.jpg"
    rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, rgb_annotated)
    print(f"Saved annotated image in RGB format: {img_path}")

async def websocket_handler(websocket, path):
    global fall_alert_triggered
    client = websocket.remote_address
    print(f"[{datetime.datetime.now()}] Client {client} connected.")
    try:
        async for message in websocket:
            frame_data = base64.b64decode(message)
            image = Image.open(BytesIO(frame_data)).convert("RGB")
            frame = np.array(image)

            predictions = run_inference(frame)
            predictions = predictions.reshape(-1, 57)

            pose_state = PoseState.UNKNOWN

            for detection in predictions:
                if detection[4] < CONFIDENCE_THRESHOLD:
                    continue
                keypoints = detection[6:].reshape(17, 3)
                pose_state = detect_fall(keypoints, frame.shape)
                break

            
            pose_history.append(pose_state)
            recent_states = list(pose_history)

            if (PoseState.STANDING in recent_states[:5] and
                all(state == PoseState.FALLEN for state in recent_states[-10:])):
                fall_alert_triggered = True
            else:
                fall_alert_triggered = False
            print("fall_alert_triggered:::", fall_alert_triggered)
            annotate_and_save(predictions, frame)

            await websocket.send(json.dumps({
                'status': 'frame_processed',
                'fall_detected': fall_alert_triggered
            }))

    except websockets.ConnectionClosed:
        print(f"[{datetime.datetime.now()}] Client {client} disconnected.")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Error: {e}")

async def main():
    print("Starting WebSocket Server...")
    async with websockets.serve(
        websocket_handler, "0.0.0.0", 8765,
        ping_interval=300,
        ping_timeout=20
    ):
        print("WebSocket Server Running on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
