import cv2
import numpy as np
import os

# Zuordnung der Labels
label_map = {
    "abgelenkt": 0,
    "fokussiert": 1,
    "handy": 2,
    "abwesend": 3
}


import cv2
import numpy as np
import os

# Zuordnung der Labels
label_map = {
    "abgelenkt": 0,
    "fokussiert": 1,
    "handy": 2,
    "abwesend": 3
}


def extract_video_sequence(video_path, num_frames=10, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        # Not enough frames → Padding (not ideal, but safe)
        while len(frames) < num_frames:
            frames.append(np.zeros_like(frames[0]))

    sequence = np.stack(frames)
    return sequence  # shape: (num_frames, 224, 224, 3)
def load_dataset_from_videos(video_dir, label_map, num_frames=10):
    X = []
    y = []

    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue
        # Label aus Dateinamen extrahieren
        for label_name in label_map:
            if label_name in filename.lower():
                label = label_map[label_name]
                break
        else:
            continue  

        video_path = os.path.join(video_dir, filename)
        sequence = extract_video_sequence(video_path, num_frames=num_frames)
        X.append(sequence)
        y.append(label)

    X = np.array(X)  
    y = np.array(y)  
    return X, y