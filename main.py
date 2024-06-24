from datetime import datetime, timedelta
import time
import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import torch
from PIL import Image
import numpy as np
from typing import Tuple
import logging
from logging_module import LogginModule


logger = LogginModule(app_name="yolov5_app", level=logging.DEBUG).get_logger()
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
DEBUG: bool = bool(os.getenv("debug", False))
RECORD_VIDEO: bool = bool(os.getenv("record_video", False))
logger.info(f"Record Video: {RECORD_VIDEO}")
#region DEVICE
GPU_ON = bool(os.getenv("gpu_on", False))
if GPU_ON:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        raise ValueError("Can't set GPU! Switch to CPU")    
logger.info(f"Device detected: {device}")
#endregion
#region MODEL
MODEL_TYPE = os.getenv("model_type", "yolov5s")
CONFIDENCE_THRESHOLD:float = float(os.getenv("conf_threshold", .5))
logger.info(f"Seted model: {MODEL_TYPE}.Seted confidence threshold:\t{CONFIDENCE_THRESHOLD}.")
#endregion
#region CAMERA
CAMERA_IP_ADDR = os.getenv("camera_addr")
video_from_path = os.getenv("video_from_path")
VIDEO_PATH = os.path.join(os.getcwd(), video_from_path) if video_from_path else None
RECORDING_MINUTES = int(os.getenv("recording_minutes", 0))
RECORDING_SECONDS = int(os.getenv("recording_seconds", 0))
DRAW_BOXES:bool = bool(os.getenv("draw_boxes", False))
#endregion
#region CATEGORIES
CATEGORIES_TO_SEARCH = []
categories_as_str = os.getenv("category_name", None)
if categories_as_str:
    CATEGORIES_TO_SEARCH: list[int] = list(map(int, categories_as_str.split(",")))
    category_mgs = ", ".join(category_maper.get(cat) for cat in CATEGORIES_TO_SEARCH)
    logger.info(f"Category to detect: {category_mgs}.")
#endregion
SHOW_FPS: bool = bool(os.getenv("show_fps", False))


#model
model = torch.hub.load("ultralytics/yolov5", MODEL_TYPE)
model.to(device)
model.conf = CONFIDENCE_THRESHOLD


class DetectCategory():

    def __init__(self):
        self.video_writer:cv2.VideoWriter = None
        self.recording_flag:bool = False


    def get_fps(self,):
        if self.fps_elapsed_time >= 1.0:
            fps = self.frame_counter / self.fps_elapsed_time
            print(f"FPS: {fps}")            
            self.frame_counter = 0
            self.fps_start_time = time.time()


    def draw_boxes_on_frame(self, results, frame, searched_cls):        
        result = results.xyxy[0]
        # test_only_one_class = next(iter(searched_cls))
        # result = result[result[:, -1] == test_only_one_class]
        if not len(result):
            return        
        for *box, conf, cls in result:  # x1, y1, x2, y2, confidence, class
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"        
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
    def initialize_video_writer(self, frame:np.ndarray, output_path, fps=20.0) -> cv2.VideoWriter:
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return video_writer

    def category_is_detected(self, predictions:np.array, searched_category:list[int]) -> Tuple[bool, str]:
        # if have empty prediction
        if not len(predictions):
            return (None, None)
        pred_np = predictions[:,4:]
        category_recognition_bool:bool = False
        category_recognition_array:list[bool] = np.isin(searched_category, pred_np)
        category_was_detected:bool = any(category_recognition_array)
        category_detected:np.array = np.array(searched_category)[category_recognition_array]
        if category_was_detected:
            category_recognition_bool = True
            if DEBUG:
                logger.debug(f"{category_maper.get(next(iter(category_detected))).upper()} score: {round(pred_np[pred_np[:,1] == next(iter(category_detected))][0][0], 3)} file: {VIDEO_PATH if VIDEO_PATH else CAMERA_IP_ADDR}")
        if category_recognition_bool:
            if not self.recording_flag:
                detected_category_name = category_maper.get(next(iter(category_detected)))
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
        
        #region for fps
        if SHOW_FPS:
            self.frame_counter = 0
            self.fps_start_time = time.time()
        #endregion
        while True:
            success, frame = cap.read()

            #region for fps
            if SHOW_FPS:
                self.frame_counter += 1
                self.fps_elapsed_time = time.time() - self.fps_start_time
                self.get_fps()
            #endregion

            if not success:
                return Response("Failed to capture image", status_code=500)

            # Convert frame to PIL image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform object detection
            results = model(img)

            if RECORD_VIDEO and DRAW_BOXES:
                self.draw_boxes_on_frame(frame=frame, results=results, searched_cls=CATEGORIES_TO_SEARCH)
            
            is_detected = False
            if CATEGORIES_TO_SEARCH:
                if not GPU_ON:
                    predictions = results.xyxy[0].numpy()
                else:
                    predictions = np.array(results.xyxy[0].tolist())
                is_detected, category_name = self.category_is_detected(predictions, CATEGORIES_TO_SEARCH)
                if category_name:
                    last_detected_category_name = category_name

            if is_detected:                
                recording_start_time:datetime = datetime.now()
                stop_time:datetime = recording_start_time + timedelta(minutes=RECORDING_MINUTES, seconds=RECORDING_SECONDS)
                time_as_str:str = recording_start_time.strftime("%Y_%m_%d__%H_%M")                
                if RECORD_VIDEO:
                    logger.info(f"start recording video at {recording_start_time} ==> {category_name if category_name else 'output_video'}_{time_as_str}.mp4")                    
                    self.video_writer = self.initialize_video_writer(frame=frame, output_path=f"{category_name if category_name else 'output_video'}_{time_as_str}.mp4")
            if RECORD_VIDEO:
                if self.recording_flag and stop_time >= datetime.now():
                    self.video_writer.write(frame)
                if self.recording_flag:
                    if stop_time < datetime.now() and self.recording_flag:                    
                        self.video_writer.release()
                        logger.info(f"stop recording video at {datetime.now()} ==> {last_detected_category_name if last_detected_category_name else 'output_video'}_{time_as_str}.mp4")
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
