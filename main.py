import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import torch
from PIL import Image
import numpy as np

print("starting!!")
app = FastAPI()
#
CAR = 2 # is mapped as car
DOG = 16 # is mapped as car
PERSON = 1
recording_flag = False
CONFIDENCE_THRESHOLD = 0.65
# Replace with your camera's IP stream URL
camera_ip = os.getenv("camera_addr")

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.conf = CONFIDENCE_THRESHOLD

def get_confidence_score_and_class_name(predictions:np.array, searched_class:int)->dict:
    global recording_flag   
    pred_np = predictions[:,4:]
    class_recognition_bool:bool = False
    class_recognition_array = np.where(pred_np == searched_class)
    if len(class_recognition_array[0]) > 0:
        class_recognition_bool = True
    if class_recognition_bool:
        if not recording_flag:
            print(f"recording started flag seted as true. Recrding object {searched_class}")
            recording_flag = True
        
        
def get_video_stream():
    cap = cv2.VideoCapture(camera_ip)

    if not cap.isOpened():
        return Response("Camera stream not accessible", status_code=404)

    while True:
        success, frame = cap.read()

        if not success:
            return Response("Failed to capture image", status_code=500)

        # Convert frame to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform object detection
        results = model(img)
        predictions = results.xyxy[0].numpy()  # Convert to NumPy array for easier handling        
        pred_as_dict = get_confidence_score_and_class_name(predictions, PERSON)
        


        
        # Draw bounding boxes on the frame
        annotated_frame = np.squeeze(results.render())
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    print("uvicorn started")
    uvicorn.run(app, host="0.0.0.0", port=8000)
