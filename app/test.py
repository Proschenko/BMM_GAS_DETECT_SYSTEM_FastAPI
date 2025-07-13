import os
import argparse
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

def process_and_detect(input_path, output_path, model_path, show_live=True):  # show_live=True по умолчанию
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Инициализация VideoWriter (если output_path указан)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    try:
        for _ in tqdm(range(total_frames), desc="Обработка видео"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Предобработка
            negative = 255 - frame
            gray = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # 2. Детекция
            results = model.predict(gray_3ch, imgsz=640, conf=0.5)
            annotated_frame = results[0].plot()  # Автоматическая визуализация
            
            # Показываем результат
            cv2.imshow("Live Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Сохраняем (если нужно)
            if output_path:
                out.write(annotated_frame)
                
    finally:
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    return True

def main():
    parser = argparse.ArgumentParser(description='Детекция утечек с отображением в реальном времени')
    parser.add_argument('--input', required=True, help='Путь к видеофайлу')
    parser.add_argument('--output', help='Путь для сохранения (необязательно)')
    parser.add_argument('--model', default='leak_detectorv0.1.pt', help='Путь к модели YOLO')
    args = parser.parse_args()

    # Проверка и нормализация путей
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Входной файл не найден: {args.input}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("\nЗапуск обработки с отображением в реальном времени...")
    print("Нажмите 'q' для остановки")
    
    process_and_detect(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        show_live=True
    )

if __name__ == "__main__":
    main()