"""Import selected Tomato PlantVillage classes from a ZIP archive.

The ZIP may contain the class folders directly or inside nested directories.
The imported images are copied into raw_data/ for the next preparation step.
"""

from pathlib import Path
import shutil
import tempfile
import zipfile

ZIP_PATH = Path(r"C:\Users\Lakshmi\Downloads\archive (3).zip")
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE_TO_TARGET = {
    "Tomato_healthy": "healthy",
    "Tomato_Early_blight": "leaf_blight",
    "Tomato_Late_blight": "leaf_blast",  # Placeholder mapping requested for this project.
}


def find_class_folder(extracted_dir: Path, folder_name: str) -> Path:
    matches = [
        path for path in extracted_dir.rglob("*")
        if path.is_dir() and path.name == folder_name
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{folder_name}' inside the extracted ZIP contents."
        )
    if len(matches) > 1:
        print(f"Warning: found multiple '{folder_name}' folders; using {matches[0]}")
    return matches[0]


def extract_zip_safely(zip_path: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = (destination / member.filename).resolve()
            if target_path != destination_root and destination_root not in target_path.parents:
                raise ValueError(f"Unsafe ZIP path detected: {member.filename}")
        archive.extractall(destination)


def copy_images(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for index, image_path in enumerate(images):
        # Prefixing avoids collisions when different nested folders contain the same filename.
        destination = target_dir / f"{index:05d}_{image_path.name}"
        shutil.copy2(image_path, destination)
    return len(images)


def main() -> None:
    if not ZIP_PATH.is_file():
        raise SystemExit(f"ZIP file not found: {ZIP_PATH}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plantvillage_") as temporary_dir:
        extracted_dir = Path(temporary_dir)
        print(f"Extracting {ZIP_PATH.name}...")
        extract_zip_safely(ZIP_PATH, extracted_dir)

        for source_name, target_name in SOURCE_TO_TARGET.items():
            source_dir = find_class_folder(extracted_dir, source_name)
            target_dir = RAW_DATA_DIR / target_name
            count = copy_images(source_dir, target_dir)
            print(f"Copied {count} images: {source_name} -> raw_data/{target_name}/")

    print(f"Import complete. Dataset is ready under {RAW_DATA_DIR}")
    print("The temporary extraction folder was removed automatically.")


if __name__ == "__main__":
    main()
