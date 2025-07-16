from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.service import process_video_with_annotations
import os
import shutil
import uuid

router = APIRouter()

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_filename = f"annotated_{filename}"
    output_path = os.path.join(RESULT_DIR, output_filename)

    success = process_video_with_annotations(input_path, output_path)

    if not success:
        return {"status": "error", "message": "Ошибка обработки видео"}

    # <-- Возвращаем endpoint, а не физический путь
    return {
        "status": "success",
        "output_video": f"/video/{output_filename}"
    }

@router.get("/video/{video_name}")
async def get_processed_video(video_name: str):
    path = os.path.join(RESULT_DIR, video_name)
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    return {"error": "Видео не найдено"}
