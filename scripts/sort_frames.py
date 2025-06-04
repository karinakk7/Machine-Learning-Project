import os
import shutil
from sklearn.model_selection import train_test_split

# Pfad zu deinem Haupt-Dataset-Ordner (mit Klassen als Unterordner)
dataset_dir = 'dataset'

# Neue Ordner für train und val
train_dir = os.path.join('dataset', 'train')
val_dir = os.path.join('dataset', 'val')

# Die Klassen, die du hast
labels = ['fokussiert', 'abgelenkt', 'handy', 'nicht_anwesend']

# Ordner für train und val mit Klassen erstellen (wenn nicht vorhanden)
for split_dir in [train_dir, val_dir]:
    for label in labels:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)

# Für jede Klasse Bilder sammeln, splitten und verschieben
for label in labels:
    label_path = os.path.join(dataset_dir, label)
    images = [f for f in os.listdir(label_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Aufteilen in 80% train, 20% val
    train_files, val_files = train_test_split(images, test_size=0.2, random_state=42)

    # Bilder verschieben
    for f in train_files:
        src = os.path.join(label_path, f)
        dst = os.path.join(train_dir, label, f)
        shutil.move(src, dst)

    for f in val_files:
        src = os.path.join(label_path, f)
        dst = os.path.join(val_dir, label, f)
        shutil.move(src, dst)

print("Train/Val Split fertig!")
