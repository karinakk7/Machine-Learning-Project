import os

# Wurzelordner mit train/ und val/
base_dir = 'dataset'

# Unterordner (train und val)
splits = ['train', 'val']

for split in splits:
    print(f"\n📂 {split.upper()}:")

    split_path = os.path.join(base_dir, split)
    if not os.path.exists(split_path):
        print(f"⚠️ Ordner {split_path} nicht gefunden.")
        continue

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = len([
            f for f in os.listdir(class_path)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        print(f"  🔸 {class_name}: {count} Bilder")
