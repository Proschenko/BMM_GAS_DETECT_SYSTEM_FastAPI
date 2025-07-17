"""
Модуль для обработки видео с обнаружением утечек с помощью YOLO модели.
Основные функции:
- convert_to_mp4 - конвертация видео в MP4 формат
- process_video_with_annotations - обработка видео с наложением масок и bounding boxes
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import logging

# Настройка системы логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_mp4(input_path, output_path):
    """
    Конвертирует видеофайл в формат MP4 с использованием ffmpeg.
    
    Args:
        input_path (str): Путь к исходному видеофайлу
        output_path (str): Путь для сохранения результата
        
    Returns:
        bool: True если конвертация успешна, False в случае ошибки
        
    Процесс:
        1. Формирует команду ffmpeg с оптимальными параметрами
        2. Запускает процесс конвертации
        3. Обрабатывает возможные ошибки
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",  # Используем кодек H.264
        "-preset", "fast",   # Баланс между скоростью и сжатием
        "-crf", "23",        # Качество (0-51, меньше - лучше)
        "-movflags", "+faststart",  # Для потокового воспроизведения
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
    """
    Обрабатывает видеофайл, применяя модель YOLO для обнаружения утечек и добавляя аннотации.
    
    Args:
        input_path (str): Путь к исходному видеофайлу
        output_path (str): Путь для сохранения обработанного видео
        
    Returns:
        bool: True если обработка успешна, False в случае ошибки
        
    Процесс:
        1. Загрузка YOLO модели
        2. Чтение видео и получение его характеристик
        3. Обработка каждого кадра:
           - Предобработка (негатив, CLAHE)
           - Детекция с помощью YOLO
           - Визуализация результатов (маски и bounding boxes)
        4. Сохранение результата и конвертация в MP4
    """
    model_path = "models/leak_detectorv0.1.pt"  # Путь к файлу модели

    if not os.path.exists(model_path):
        logger.error("❌ Model not found at %s", model_path)
        return False

    try:
        model = YOLO(model_path)  # Загрузка YOLO модели
        logger.info("✅ YOLO model loaded")
    except Exception as e:
        logger.exception("❌ Ошибка загрузки модели YOLO")
        return False

    # Открытие видеофайла
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("❌ Не удалось открыть видеофайл: %s", input_path)
        return False

    # Получение параметров видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"📹 Видео: {total_frames} кадров, {fps:.2f} fps, размер: {original_width}x{original_height}")

    # Параметры для ресайза (YOLO работает с 640x640)
    output_size = (640, 640)
    scale_x = original_width / 640  # Коэффициенты масштабирования
    scale_y = original_height / 640

    # Создание временного файла для записи (AVI формат для быстрой записи)
    tmp_path = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (original_width, original_height))

    try:
        # Обработка каждого кадра
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"⚠️ Кадр {frame_idx} не удалось прочитать")
                break

            original = frame.copy()  # Сохраняем оригинальный кадр

            # Блок предобработки кадра
            negative = 255 - frame  # Инверсия цветов
            gray = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)  # Конвертация в grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Создание CLAHE фильтра
            gray_eq = clahe.apply(gray)  # Применение гистограммной эквализации
            gray_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)  # Обратно в RGB
            resized = cv2.resize(gray_rgb, output_size)  # Ресайз для YOLO

            # Детекция с помощью YOLO
            results = model.predict(resized, imgsz=640, conf=0.4)  # Порог уверности 0.4
            result = results[0]
            logger.info(f"🧠 Кадр {frame_idx}: {len(result.boxes)} boxes, {len(result.masks) if result.masks else 0} masks")

            annotated = original.copy()  # Копия для аннотаций

            # Отрисовка масок (если есть)
            if result.masks:
                for mask in result.masks.xy:
                    # Масштабирование координат маски к исходному размеру
                    scaled_mask = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in mask], np.int32)
                    cv2.polylines(annotated, [scaled_mask], True, (0, 255, 0), 2)  # Контур маски
                    cv2.fillPoly(annotated, [scaled_mask], color=(0, 255, 0, 50))  # Заливка маски

            # Отрисовка bounding boxes (если есть)
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Координаты bbox
                    # Масштабирование координат к исходному размеру
                    x1, y1, x2, y2 = map(int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, x2 * scale_y])
                    cls_id = int(box.cls[0].item())  # ID класса
                    conf = float(box.conf[0].item())  # Уверенность модели
                    label = f"{model.names[cls_id]} {conf:.2f}"  # Текст для отображения
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Прямоугольник
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 5)  # Текст

            out.write(annotated)  # Запись обработанного кадра

    finally:
        # Освобождение ресурсов
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    logger.info("💾 Обработка завершена. Начинаем перекодировку в MP4...")
    success = convert_to_mp4(tmp_path, output_path)

    # Удаление временного файла
    if success:
        os.remove(tmp_path)
        logger.info("🧹 Временный файл удалён")
        return True
    else:
        logger.error("❌ Ошибка при перекодировке. Файл: %s", tmp_path)
        return False