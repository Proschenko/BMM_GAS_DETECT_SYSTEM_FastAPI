import os
import cv2
import numpy
from ultralytics import YOLO
from tqdm import tqdm


def process_and_detect(input_path, output_path, model_path, show_live=True):
    """Обрабатывает видео и применяет YOLO с отображением в реальном времени"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_size = (640, 640)

    # Инициализация VideoWriter
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size, isColor=True)
        if not out.isOpened():
            raise RuntimeError(f"Не удалось инициализировать VideoWriter по пути: {output_path}")

    try:
        for _ in tqdm(range(total_frames), desc="Обработка видео"):
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Инверсия
            negative = 255 - frame

            # 2. В градации серого
            gray = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)

            # 3. CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_applied = clahe.apply(gray)

            # 4. Обратно в RGB
            gray_3ch = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2RGB)

            # 5. Масштабирование до 640x640
            resized_input = cv2.resize(gray_3ch, output_size, interpolation=cv2.INTER_AREA)

            # 6. YOLO
            results = model.predict(resized_input, imgsz=640, conf=0.4)

            # if results[0].masks:
            #     mask_points: numpy.ndarray = results[0].masks.xy[0]

            annotated_frame = results[0].plot()

            # 7. Показываем
            if show_live:
                cv2.imshow("Live Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 8. Сохраняем
            if output_path:
                out.write(annotated_frame) # type: ignore

    finally:
        cap.release()
        if output_path:
            out.release() # type: ignore
        cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    # input_video = "video_2025-07-14_21-16-19.mp4"
    # input_video = "video_2025-07-14_21-18-20.mp4"
    input_video = r"C:\Users\Artemis\source\py_repos\bmm_gas_detector\videos\video_2025-07-14_21-22-23.mp4"
    # input_video = "video_2025-07-14_21-22-23.mp4"
    output_video = "output_video_N.mp4"
    # model_path = "yolo11n-seg-b16-ep100-pat15-cm15-map50=0.885-map5095=0.498.pt" # Very good (small noise)
    # model_path = "yolo11n-seg-b16-ep99-pat22-cm22-map50=0.886-map5095=0.470.pt" # bad
    # model_path = "yolo11n-seg-b16-ep166-pat30-cm30-map50=0.875-map5095=0.488.pt" # IS WOOORK!!!! (good work (recall 100) + noise)
    model_path = r"models\best.pt" # IS WOOORK!!!! (BEST)
    # model_path = "yolo11s-seg-b16-ep162-pat30-cm30-map50=0.954-map5095=0.580.pt" # 

    
    process_and_detect(
        input_path=input_video,
        output_path=output_video,
        model_path=model_path,
        show_live=True
    )