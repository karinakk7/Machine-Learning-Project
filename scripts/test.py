import os
import shutil

source_root = "raw_videos"
target_root = os.path.join("raw_videos_test", "raw_videos")

# Stelle sicher, dass Zielordner existieren
classes = ["abgelenkt", "fokussiert", "handy", "nicht_anwesend"]
for cls in classes:
    os.makedirs(os.path.join(target_root, cls), exist_ok=True)

# Durchlaufe Klassen und kopiere Videos
for cls in classes:
    src_class_dir = os.path.join(source_root, cls)
    dst_class_dir = os.path.join(target_root, cls)

    for file in os.listdir(src_class_dir):
        if file.lower().endswith(".mp4"):
            src_path = os.path.join(src_class_dir, file)
            dst_path = os.path.join(dst_class_dir, file)

            # Datei kopieren
            shutil.copy2(src_path, dst_path)
            print(f"✅ {file} → {dst_class_dir}")

print("📦 Alle Videos erfolgreich kopiert.")
