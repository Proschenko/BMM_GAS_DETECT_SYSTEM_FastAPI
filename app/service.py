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
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]

    logger.info(f"🔄 Запуск ffmpeg: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("✅ FFmpeg: Успешно перекодировано")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[FFMPEG ERROR] Команда завершилась с ошибкой: {e.stderr.decode(errors='ignore')}")
        return False
    except FileNotFoundError as e:
        logger.error(f"[FFMPEG ERROR] FFmpeg не найден. Убедись, что он в PATH: {e}")
        return False

def process_video_with_annotations(input_path, output_path):
    model_path = "models/leak_detectorv0.1.pt"

    if not os.path.exists(model_path):
        logger.error("❌ Model not found at %s", model_path)
        return False

    try:
        model = YOLO(model_path)
        logger.info("✅ YOLO model loaded")
    except Exception as e:
        logger.exception("❌ Ошибка загрузки модели YOLO")
        return False

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("❌ Не удалось открыть видеофайл: %s", input_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"📹 Видео: {total_frames} кадров, {fps:.2f} fps, размер: {original_width}x{original_height}")

    output_size = (640, 640)
    scale_x = original_width / 640
    scale_y = original_height / 640

    tmp_path = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (original_width, original_height))

    try:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"⚠️ Кадр {frame_idx} не удалось прочитать")
                break

            original = frame.copy()

            # Предобработка
            negative = 255 - frame
            gray = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(gray_rgb, output_size)

            # YOLO
            results = model.predict(resized, imgsz=640, conf=0.4)
            result = results[0]
            logger.info(f"🧠 Кадр {frame_idx}: {len(result.boxes)} boxes, {len(result.masks) if result.masks else 0} masks")

            annotated = original.copy()

            if result.masks:
                for mask in result.masks.xy:
                    scaled_mask = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in mask], np.int32)
                    cv2.polylines(annotated, [scaled_mask], True, (0, 255, 0), 2)
                    cv2.fillPoly(annotated, [scaled_mask], color=(0, 255, 0, 50))

            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 5)

            out.write(annotated)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    logger.info("💾 Обработка завершена. Начинаем перекодировку в MP4...")
    success = convert_to_mp4(tmp_path, output_path)

    if success:
        os.remove(tmp_path)
        logger.info("🧹 Временный файл удалён")
        return True
    else:
        logger.error("❌ Ошибка при перекодировке. Файл: %s", tmp_path)
        return False
