import os
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_mp4(input_path, output_path):
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]

    logger.info(f"🔄 FFmpeg: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("✅ FFmpeg: конвертация прошла успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[FFMPEG ERROR] {e.stderr.decode(errors='ignore')}")
        return False
    except FileNotFoundError as e:
        logger.error(f"[FFMPEG ERROR] ffmpeg не найден: {e}")
        return False


def process_video_with_annotations(input_path, output_path):
    model_path = "models/yolo11v0.2.pt"
    if not os.path.exists(model_path):
        logger.error("❌ Модель не найдена по пути %s", model_path)
        return False, []

    try:
        model = YOLO(model_path)
        logger.info("✅ YOLO модель загружена")
    except Exception:
        logger.exception("❌ Ошибка при загрузке модели YOLO")
        return False, []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("❌ Не удалось открыть видеофайл")
        return False, []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"📹 Видео: {total_frames} кадров, {fps:.2f} fps")

    frame_step = max(1, int(fps / 5))  # 5 кадра в секунду
    output_size = (640, 640)
    scale_x = original_width / 640
    scale_y = original_height / 640

    tmp_path = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (original_width, original_height))

    detections_info = []
    last_annotated_frame = None
    last_detections = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            original = frame.copy()
            negative = 255 - frame
            gray = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(gray_rgb, output_size)

            results = model.predict(resized, imgsz=640, conf=0.4)[0]
            frame_ann = original.copy()
            frame_detections = []

            if results.masks:
                for mask in results.masks.xy:
                    scaled_mask = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in mask], np.int32)
                    cv2.polylines(frame_ann, [scaled_mask], True, (0, 255, 0), 2)
                    cv2.fillPoly(frame_ann, [scaled_mask], color=(0, 255, 0))

            if results.boxes:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(frame_ann, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_ann, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    frame_detections.append({
                        "timestamp": round(frame_idx / fps, 2),
                        "class": model.names[cls_id],
                        "conf": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })

            last_annotated_frame = frame_ann
            last_detections = frame_detections

            if frame_detections:
                detections_info.append({
                    "frame": frame_idx,
                    "time": round(frame_idx / fps, 2),
                    "detections": frame_detections
                })

        # Пишем каждый кадр (не слайдшоу!)
        out.write(last_annotated_frame if last_annotated_frame is not None else frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    logger.info("💾 Завершено. Перекодировка в MP4...")

    if convert_to_mp4(tmp_path, output_path):
        os.remove(tmp_path)
        logger.info("🧹 Временный AVI удалён")
        return True, detections_info
    else:
        logger.error("❌ Ошибка конвертации в MP4")
        return False, []
