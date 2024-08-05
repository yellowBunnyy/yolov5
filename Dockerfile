FROM ultralytics/yolov5:latest

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the requirements to model

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code to the working directory
COPY . .

ARG GPU=""
# default video
ARG VIDEO=/workspaces/yolov5/test_video.mp4
ARG CAMERA=""

# env for model
ENV MODEL_TYPE=yolov8m
ENV CAMERA_ADDR=$CAMERA
ENV VIDEO_FROM_PATH=$VIDEO
ENV RECORD_VIDEO=True
ENV DRAW_BOXES=True
ENV RECORDING_MINUTES=1
ENV RECORDING_SECONDS=0
ENV CATEGORY_NAME=0,16
ENV SHOW_FPS=True
ENV GPU_ON=$GPU
ENV CONF_THRESHOLD=0.6
ENV DEBUG=True
ENV OUTPUT_LOG_PATH=/workspace


# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI application with uvicorn
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]