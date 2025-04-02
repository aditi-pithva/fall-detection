import cv2
import numpy as np
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="yolov8n-pose_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = cv2.imread("test-2.jpg")
input_w, input_h = 640, 640  # assuming model uses 640x640 input

original_h, original_w = image.shape[:2]
resized = cv2.resize(image, (input_w, input_h))
input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

# Inference
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])  # [1, N, 57]
detections = output[0]

# Assume `image` is your original input (e.g., 640x640)
height, width = resized.shape[:2]

for person in detections:
    if person[4] < 0.4:
        continue  # low objectness score

    for i in range(17):
        x = person[6 + i * 3] * width   # scale x
        y = person[6 + i * 3 + 1] * height  # scale y
        conf = person[6 + i * 3 + 2]

        print(f"Keypoint {i}: ({x:.2f}, {y:.2f}) conf={conf:.2f}")

        if conf > 0.3:
            cv2.circle(resized, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(resized, str(i), (int(x) + 2, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Show and save
cv2.imshow("Pose Output", resized)
cv2.imwrite("pose_output.jpg", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
