from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import torch
from PIL import Image
import numpy as np

print("starting!!")
app = FastAPI()

# Replace with your camera's IP stream URL
camera_ip = "rtsp://mysupercam:lubiepasztet100@192.168.1.102/stream1"

model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# @app.get("/video_feed")
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
