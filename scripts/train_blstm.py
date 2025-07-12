import os
import shutil
from pathlib import Path

def move_selected_files(source_dir, target_dir, keywords):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0

    for file in source_dir.glob("*.jpg"):
        fname = file.name.lower()
        if any(kw.lower() in fname for kw in keywords):
            shutil.move(str(file), target_dir / file.name)
            print(f"🔄 Verschoben: {file.name}")
            moved += 1

    print(f"\n✅ {moved} Dateien verschoben mit Keywords: {keywords}")

if __name__ == "__main__":
    move_selected_files(
        source_dir="dataset/train/abgelenkt",
        target_dir="dataset/val/abgelenkt",
        keywords=["karina"]  # nur diese Dateien
    )

