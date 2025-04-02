from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import base64
import asyncio

app = FastAPI()

model = YOLO("yolov10n.pt")

ASPECT_RATIO_THRESHOLD = 1.5

def detect_fall(frame):
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 
        cls = result.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if int(cls[i]) == 0:
                x1, y1, x2, y2 = box
                height = y2 - y1
                width = x2 - x1

                if width / height > ASPECT_RATIO_THRESHOLD:
                    return 1
                
    return 0 

@app.websocket("/ws/fall-detection")
async def websocket_fall_detection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data)
            np_img = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            fall_status = detect_fall(frame)
            await websocket.send_text(str(fall_status))
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
