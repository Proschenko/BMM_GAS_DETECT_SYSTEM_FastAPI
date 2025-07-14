import logging
import os
from types import CoroutineType
from typing import Generator
import uuid
from cv2.typing import MatLike
import cv2
from fastapi import UploadFile, WebSocket
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from models.segmentation_result import SegmentationResult

segmentation_model = YOLO()
segmentation_model_path = 'models/leak_detectorv0.1.pt'

async def upload_video(file: UploadFile):
    file_id = str(uuid.uuid4())
    filename = f"uploads/{file_id}.mp4"
    with open(filename, "wb") as f:
        f.write(await file.read())
    
    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return JSONResponse({
        "filename": file_id,
        "total_frames": total_frames
    })

def process_file(video_id: str) -> Generator[SegmentationResult]:
    segmentation_model = load_model()
    
    file_path = f"uploads/{video_id}.mp4"
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise Exception(f"Не удалось открыть видео {video_id}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # frame = prepare_frame(frame) # если загружать не "ИК" видос
            results = segmentation_model.predict(frame, imgsz=640, conf=0.5)
            yield SegmentationResult(i)
    except Exception as e:
        raise e
    finally:
        cap.release()

def load_model() -> YOLO:
    if not os.path.exists(segmentation_model_path):
        raise FileNotFoundError(f"Файл модели не найден: {segmentation_model_path}")
    try:
        return YOLO(segmentation_model_path)
    except Exception as e:
        raise Exception(f"Ошибка загрузки модели: {str(e)}")

def prepare_frame(frame: MatLike) -> MatLike:
    negative = 255 - frame
    gray = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_3ch