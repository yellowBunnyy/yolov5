from datetime import datetime, timedelta
import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import torch
from PIL import Image
import numpy as np

print("starting!!")
app = FastAPI()

category_maper = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    10: "fire hydrant",
    11:" stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog"
}

recording_flag = False
CONFIDENCE_THRESHOLD = 0.65
# Replace with your camera's IP stream URL
camera_ip = os.getenv("camera_addr")
video_from_path = os.getenv("video_from_path")

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

model.conf = CONFIDENCE_THRESHOLD
video_writer = None

def add_text(text:str, frame:np.ndarray) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)  # Bottom-left corner of the text string in the image
    font_scale = 2
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2  # Line thickness of 2 px
    frame = cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def initialize_video_writer(frame:np.ndarray, output_path, fps=20.0) -> cv2.VideoWriter:
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return video_writer

def category_is_detected(predictions:np.array, searched_category:int)->bool | None:
    global recording_flag
    global category_maper
    pred_np = predictions[:,4:]
    category_recognition_bool:bool = False
    category_recognition_array = np.where(pred_np == searched_category)
    if len(category_recognition_array[0]) > 0:
        category_recognition_bool = True
    if category_recognition_bool:
        if not recording_flag:
            print(f"recording started flag seted as true. Recrding object {category_maper.get(searched_category)}")
            recording_flag = True
            return True
        

def get_video_stream():
    global video_from_path
    global recording_flag
    if camera_ip:
        cap = cv2.VideoCapture(camera_ip)
    elif video_from_path:
        video_from_path = os.path.join(os.getcwd(), video_from_path)
        cap = cv2.VideoCapture(video_from_path)
    else:
        return {"response": "err check venv"}

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
        is_detected:bool = category_is_detected(predictions, 0) # 0 is person

        if is_detected:
            global video_writer
            recording_start_time:datetime = datetime.now()
            stop_time:datetime = recording_start_time + timedelta(minutes=3)
            print(f"start recording video at {recording_start_time}.")
            time_as_str:str = recording_start_time.strftime("%Y_%m_%d__%H_%M")
            video_writer = initialize_video_writer(frame=frame, output_path=f"output_video_{time_as_str}.mp4")
        
        if recording_flag and stop_time >= datetime.now():
            video_writer.write(frame)
        if recording_flag:
            if stop_time < datetime.now() and recording_flag:
                video_writer.release()
                print(f"stop recording video at {datetime.now()}.")
                recording_flag= False

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
