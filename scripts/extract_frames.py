# scripts/extract_frames.py
import cv2
import os

input_dir = 'raw_video'
output_dir = 'dataset'

os.makedirs(output_dir, exist_ok=True)

frame_rate = 1  # 1 Bild pro Sekunde

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for video_file in os.listdir(class_path):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
                filename = f"{video_file[:-4]}_frame{frame_count}.jpg"
                filepath = os.path.join(output_class_dir, filename)
                cv2.imwrite(filepath, frame)
                frame_count += 1
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
