"""Prepare and validate a YOLO dataset from cleaned Ghost Pot JSONL annotations."""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_image(split_dir: Path, file_name: str) -> Path:
    direct = split_dir / file_name
    if direct.exists():
        return direct
    return split_dir / Path(file_name).name


def _to_yolo(box: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, box_width, box_height = [float(value) for value in box]
    values = [
        (x + box_width / 2.0) / width,
        (y + box_height / 2.0) / height,
        box_width / width,
        box_height / height,
    ]
    cx, cy, normalized_width, normalized_height = [round(value, 8) for value in values]
    # Quantization can make a box touching an image edge extend by a few
    # floating-point units. Reduce only that quantization excess.
    if cx - normalized_width / 2.0 < 0:
        normalized_width = round(max(0.0, 2.0 * cx), 8)
    if cx + normalized_width / 2.0 > 1:
        normalized_width = round(max(0.0, 2.0 * (1.0 - cx)), 8)
    if cy - normalized_height / 2.0 < 0:
        normalized_height = round(max(0.0, 2.0 * cy), 8)
    if cy + normalized_height / 2.0 > 1:
        normalized_height = round(max(0.0, 2.0 * (1.0 - cy)), 8)
    return cx, cy, normalized_width, normalized_height


def _format_yolo(values: Tuple[float, float, float, float]) -> str:
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def _draw_yolo_boxes(image, label_path: Path, class_names: List[str]):
    height, width = image.shape[:2]
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, box_width, box_height = parts
            class_id = int(class_id)
            cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
            x1 = int((cx - box_width / 2) * width)
            y1 = int((cy - box_height / 2) * height)
            x2 = int((cx + box_width / 2) * width)
            y2 = int((cy + box_height / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
            cv2.putText(image, label, (max(0, x1), max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image


def _source_group(file_name: str) -> str:
    return file_name.split(".rf.", 1)[0]


def prepare_dataset(
    cleaned_dir: Path = Path("data/processed/annotations_cleaned"),
    source_dir: Path = Path("data/raw/ghost_pot"),
    output_dir: Path = Path("data/yolo"),
    report_path: Path = Path("outputs/dataset/yolo_dataset_report.json"),
    validation_path: Path = Path("outputs/dataset/yolo_validation_report.json"),
    visualization_dir: Path = Path("outputs/visualizations/yolo_training"),
    seed: int = 42,
    visualization_count: int = 10,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Copy images and convert cleaned JSONL annotations to YOLO format."""
    cleaned_dir = Path(cleaned_dir)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    visualization_dir = Path(visualization_dir)
    report_path = Path(report_path)
    validation_path = Path(validation_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in visualization_dir.iterdir():
        if stale_file.is_file():
            stale_file.unlink()

    class_names = sorted({"Crab-Pot"})
    class_ids = {name: index for index, name in enumerate(class_names)}
    split_stats = {}
    source_split_groups = defaultdict(set)

    for split in ("train", "valid", "test"):
        yolo_split = "val" if split == "valid" else split
        image_output = output_dir / "images" / yolo_split
        label_output = output_dir / "labels" / yolo_split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        source_metadata = cleaned_dir / split / "metadata.jsonl"
        if not source_metadata.exists():
            raise FileNotFoundError(f"Cleaned metadata not found: {source_metadata}")

        stats = Counter()
        dimensions = Counter()
        source_split = source_dir / split
        with source_metadata.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                file_name = record["file_name"]
                source_image = _find_image(source_split, file_name)
                image = cv2.imread(str(source_image), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Unable to read source image: {source_image}")
                height, width = image.shape[:2]
                dimensions[f"{width}x{height}"] += 1
                source_split_groups[_source_group(file_name)].add(yolo_split)
                destination_image = image_output / Path(file_name).name
                shutil.copy2(source_image, destination_image)
                label_path = label_output / f"{Path(file_name).stem}.txt"

                objects = record.get("objects") or {}
                boxes = objects.get("bbox") or []
                categories = objects.get("category") or []
                if len(boxes) != len(categories):
                    raise ValueError(f"Mismatched cleaned annotation arrays: {source_metadata}:{line_number}")
                rows = []
                for box, category in zip(boxes, categories):
                    if category not in class_ids:
                        raise ValueError(f"Unknown class {category!r}: {source_metadata}:{line_number}")
                    rows.append(_format_yolo(_to_yolo(box, width, height)))
                label_path.write_text("\n".join(rows), encoding="utf-8")
                stats["images"] += 1
                stats["annotations"] += len(rows)
                stats["empty_label_images"] += int(not rows)

        split_stats[yolo_split] = {
            "images": stats["images"],
            "annotations": stats["annotations"],
            "empty_label_images": stats["empty_label_images"],
            "image_dimensions": dict(dimensions),
        }

    leakage_groups = {group: sorted(splits) for group, splits in source_split_groups.items() if len(splits) > 1}
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    validation = validate_yolo_dataset(output_dir, class_names)
    training_images = sorted(
        image_path
        for image_path in (output_dir / "images" / "train").glob("*")
        if (output_dir / "labels" / "train" / f"{image_path.stem}.txt").read_text(
            encoding="utf-8"
        ).strip()
    )
    rng = random.Random(seed)
    rng.shuffle(training_images)
    selected = training_images[:min(visualization_count, len(training_images))]
    for image_path in selected:
        label_path = output_dir / "labels" / "train" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = _draw_yolo_boxes(image, label_path, class_names)
        if not cv2.imwrite(str(visualization_dir / image_path.name), image):
            raise OSError(f"Unable to write visualization: {image_path.name}")

    report = {
        "source_annotations": str(cleaned_dir),
        "output_directory": str(output_dir),
        "data_yaml": str(yaml_path),
        "split_strategy": "preserved official cleaned train/valid/test split; deterministic copy order",
        "random_seed_for_visualizations": seed,
        "class_names": class_names,
        "class_ids": class_ids,
        "split_stats": split_stats,
        "data_leakage_check": {
            "source_groups_checked": len(source_split_groups),
            "groups_crossing_splits": len(leakage_groups),
            "cross_split_groups": leakage_groups,
        },
        "visualizations_generated": len(selected),
        "validation_passed": validation["passed"],
        "original_jsonl_modified": False,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    return report, validation


def validate_yolo_dataset(output_dir: Path, class_names: List[str]) -> Dict[str, object]:
    """Validate every generated YOLO label and referenced image."""
    output_dir = Path(output_dir)
    result = {
        "passed": True,
        "class_names": class_names,
        "images_checked": 0,
        "labels_checked": 0,
        "annotations_checked": 0,
        "invalid_labels": 0,
        "missing_images": 0,
        "missing_labels": 0,
        "malformed_labels": 0,
        "empty_label_images": 0,
        "errors": [],
    }
    for split in ("train", "val", "test"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            result["images_checked"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                result["missing_labels"] += 1
                continue
            result["labels_checked"] += 1
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                result["missing_images"] += 1
                continue
            height, width = image.shape[:2]
            lines = label_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                result["empty_label_images"] += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                result["annotations_checked"] += 1
                error = None
                if len(parts) != 5:
                    error = "expected five YOLO values"
                else:
                    try:
                        class_id = int(parts[0])
                        cx, cy, box_width, box_height = map(float, parts[1:])
                        valid = (
                            0 <= class_id < len(class_names)
                            and 0 <= cx <= 1 and 0 <= cy <= 1
                            and box_width > 0 and box_height > 0
                            and cx - box_width / 2 >= 0
                            and cy - box_height / 2 >= 0
                            and cx + box_width / 2 <= 1
                            and cy + box_height / 2 <= 1
                        )
                        if not valid:
                            error = "class or normalized box outside valid range"
                    except (TypeError, ValueError):
                        error = "non-numeric YOLO values"
                if error:
                    result["invalid_labels"] += 1
                    result["malformed_labels"] += 1
                    if len(result["errors"]) < 20:
                        result["errors"].append(f"{label_path}:{line_number}: {error}")
    result["passed"] = not any(
        result[key] for key in ("invalid_labels", "missing_images", "missing_labels", "malformed_labels")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SONAR-GUARD YOLO dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"))
    report, validation = prepare_dataset(seed=args.seed)
    print(json.dumps({"report": report, "validation": validation}, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
