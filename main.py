from datetime import datetime, timedelta
import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import torch
from PIL import Image
import numpy as np
from typing import Tuple
from logging_module import LogginModule

logger = LogginModule(app_name="yolov5_app").get_logger()
logger.info("Starting!!")

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

# VENV
CAMERA_IP_ADDR = os.getenv("camera_addr")
video_from_path = os.getenv("video_from_path")
VIDEO_PATH = os.path.join(os.getcwd(), video_from_path) if video_from_path else None
RECORDING_TIME = int(os.getenv("recording_time", 3))
CATEGORIES_TO_SEARCH: list[int] = list(map(int, os.getenv("category_name").split(",")))

#model
CONFIDENCE_THRESHOLD = 0.6
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.conf = CONFIDENCE_THRESHOLD


class DetectCategory():

    def __init__(self):
        self.video_writer:cv2.VideoWriter = None
        self.recording_flag:bool = False


    def initialize_video_writer(self, frame:np.ndarray, output_path, fps=20.0) -> cv2.VideoWriter:
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return video_writer

    def category_is_detected(self, predictions:np.array, searched_category:list[int]) -> Tuple[bool, str]:
        pred_np = predictions[:,4:]
        category_recognition_bool:bool = False
        category_recognition_array:list[bool] = np.isin(searched_category, pred_np)
        category_was_detected:bool = any(category_recognition_array)
        category_detected:np.array = np.array(searched_category)[category_recognition_array]
        if category_was_detected:
            category_recognition_bool = True
        if category_recognition_bool:
            if not self.recording_flag:
                detected_category_name = category_maper.get(next(iter(category_detected)))
                logger.info(f"recording started flag seted as true. Recrding object {category_maper.get(next(iter(category_detected)))}")                
                self.recording_flag = True
                return (True, detected_category_name)
        return (None, None)
        

    def get_video_stream(self):
        
        if CAMERA_IP_ADDR:
            cap = cv2.VideoCapture(CAMERA_IP_ADDR)
        elif VIDEO_PATH:            
            cap = cv2.VideoCapture(VIDEO_PATH)
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
            is_detected, category_name = self.category_is_detected(predictions, CATEGORIES_TO_SEARCH)

            if is_detected:                
                recording_start_time:datetime = datetime.now()
                stop_time:datetime = recording_start_time + timedelta(minutes=3)
                logger.info(f"start recording video at {recording_start_time}.")                
                time_as_str:str = recording_start_time.strftime("%Y_%m_%d__%H_%M")
                self.video_writer = self.initialize_video_writer(frame=frame, output_path=f"{category_name if category_name else 'output_video'}_{time_as_str}.mp4")
            
            if self.recording_flag and stop_time >= datetime.now():
                self.video_writer.write(frame)
            if self.recording_flag:
                if stop_time < datetime.now() and self.recording_flag:
                    self.video_writer.release()
                    logger.info(f"stop recording video at {datetime.now()}.")
                    self.recording_flag= False

            # Draw bounding boxes on the frame
            annotated_frame = np.squeeze(results.render())
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


detected_ob_inst = DetectCategory()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        detected_ob_inst.get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("uvicorn started.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
