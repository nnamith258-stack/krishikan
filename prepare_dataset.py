"""Prepare a folder-based plant disease dataset for the Colab notebook.

Expected input:
    raw_data/<class_name>/*.(jpg|jpeg|png|webp)

Output:
    data/train/<class_name>/
    data/val/<class_name>/
    data/test/<class_name>/
"""

from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("raw_data")
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit("Create raw_data/<class_name>/ folders and add images first.")

    class_dirs = sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit("No class folders found inside raw_data.")

    randomizer = random.Random(SEED)
    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 3:
            print(f"Skipping {class_dir.name}: need at least 3 images, found {len(images)}")
            continue

        randomizer.shuffle(images)
        train_end = max(1, int(len(images) * TRAIN_RATIO))
        val_end = train_end + max(1, int(len(images) * VAL_RATIO))
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            target_dir = OUTPUT_DIR / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_index, image_path in enumerate(split_images):
                destination = target_dir / f"{image_index:05d}_{image_path.name}"
                shutil.copy2(image_path, destination)

        print(
            f"{class_dir.name}: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test"
        )

    print(f"Dataset prepared under {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
