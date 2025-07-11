# scripts/extract_frames.py
import cv2
import os
import os

# Zielordner
train_dir = os.path.join('dataset', 'train')

# Deine Klassen
classes = ['fokussiert', 'abgelenkt', 'handy', 'nicht_anwesend']

# Ordnerstruktur anlegen
for cls in classes:
    class_path = os.path.join(train_dir, cls)
    os.makedirs(class_path, exist_ok=True)
    print(f" Ordner erstellt: {class_path}")

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
import cv2
import os

# Zielordner für Trainingsdaten
train_dir = os.path.join('dataset', 'train')

# Deine Klassen (Ordnernamen in raw_video müssen genauso heißen!)
classes = ['fokussiert', 'abgelenkt', 'handy', 'nicht_anwesend']

# Ordnerstruktur anlegen
for cls in classes:
    class_path = os.path.join(train_dir, cls)
    os.makedirs(class_path, exist_ok=True)
    print(f" Ordner erstellt: {class_path}")

# Eingabeordner mit Videos
input_dir = 'raw_video'
frame_rate = 1  # 1 Bild pro Sekunde

# Alle Klassen-Videos verarbeiten
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_dir = os.path.join(train_dir, class_name)  # ✅ FIX: speichere direkt in dataset/train/...

    for video_file in os.listdir(class_path):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extrahiere 1 Frame pro Sekunde
            if int(fps) > 0 and count % int(fps) == 0:
                filename = f"{video_file[:-4]}_frame{frame_count}.jpg"
                filepath = os.path.join(output_class_dir, filename)
                cv2.imwrite(filepath, frame)
                frame_count += 1

            count += 1

        cap.release()
        print(f" Extrahiert aus: {video_file} → {frame_count} Bilder")
