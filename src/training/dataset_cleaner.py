"""Clean JSONL bounding-box annotations without modifying source files."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

BBox = List[float]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _numbers(values: Iterable[object]) -> BBox:
    return [float(value) for value in values]


def _box_is_inside(box: BBox, image_width: int, image_height: int) -> bool:
    x, y, width, height = box
    return (
        x >= 0 and y >= 0 and width > 0 and height > 0
        and x + width <= image_width
        and y + height <= image_height
    )


def _clip_box(box: BBox, image_width: int, image_height: int) -> Tuple[BBox, bool, bool]:
    """Clip absolute xywh coordinates and return (box, changed, valid)."""
    x, y, width, height = box
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    clipped = [x1, y1, x2 - x1, y2 - y1]
    changed = any(abs(before - after) > 1e-9 for before, after in zip(box, clipped))
    valid = clipped[2] > 0 and clipped[3] > 0
    return clipped, changed, valid


def _format_box(box: BBox) -> BBox:
    """Keep integer coordinates readable while preserving fractional values."""
    return [int(value) if float(value).is_integer() else round(value, 6) for value in box]


def _image_path(split_dir: Path, file_name: str) -> Path:
    candidate = split_dir / file_name
    if candidate.exists():
        return candidate
    return split_dir / Path(file_name).name


def _draw_box(image, box: BBox, colour, label: str):
    x, y, width, height = box
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + width)), int(round(y + height))
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(image, label, (max(0, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def _write_visualization(image_path: Path, boxes: List[Tuple[BBox, BBox]], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image for visualization: {image_path}")
    original = image.copy()
    clipped = image.copy()
    for original_box, clipped_box in boxes:
        _draw_box(original, original_box, (0, 0, 255), "original")
        _draw_box(clipped, clipped_box, (0, 200, 0), "clipped")
    comparison = cv2.hconcat([original, clipped])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Unable to write visualization: {output_path}")


def clean_dataset(
    dataset_dir: Path,
    cleaned_dir: Path,
    report_path: Path,
    log_path: Path,
    visualization_dir: Path,
    visualization_limit: int = 12,
) -> Dict[str, object]:
    """Clean all split metadata files and validate the cleaned output."""
    dataset_dir = Path(dataset_dir)
    cleaned_dir = Path(cleaned_dir)
    report_path = Path(report_path)
    log_path = Path(log_path)
    visualization_dir = Path(visualization_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    images_affected = set()
    visualization_boxes: Dict[str, List[Tuple[BBox, BBox]]] = {}
    log_rows = []
    class_names = set()
    split_reports = {}

    for source_metadata in sorted(dataset_dir.glob("*/metadata.jsonl")):
        split = source_metadata.parent.name
        split_output = cleaned_dir / split
        split_output.mkdir(parents=True, exist_ok=True)
        output_metadata = split_output / "metadata.jsonl"
        split_counts = Counter()
        with source_metadata.open("r", encoding="utf-8") as source, output_metadata.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                file_name = record["file_name"]
                image_path = _image_path(source_metadata.parent, file_name)
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Unable to read referenced image: {image_path}")
                image_height, image_width = image.shape[:2]
                objects = record.get("objects") or {}
                boxes = objects.get("bbox") or []
                categories = objects.get("category") or []
                areas = objects.get("area") or []
                if not (len(boxes) == len(categories) == len(areas)):
                    raise ValueError(f"Mismatched annotation arrays: {source_metadata}:{line_number}")

                cleaned_boxes = []
                cleaned_categories = []
                cleaned_areas = []
                for annotation_index, (raw_box, category, area) in enumerate(zip(boxes, categories, areas)):
                    original_box = _numbers(raw_box)
                    clipped_box, changed, valid = _clip_box(original_box, image_width, image_height)
                    class_names.add(str(category))
                    counts["total_annotations_inspected"] += 1
                    split_counts["total_annotations_inspected"] += 1
                    if changed:
                        counts["out_of_bounds_annotations"] += 1
                        counts["annotations_clipped"] += int(valid)
                        counts["annotations_removed_invalid"] += int(not valid)
                        images_affected.add(f"{split}/{file_name}")
                        split_counts["out_of_bounds_annotations"] += 1
                        split_counts["annotations_clipped"] += int(valid)
                        split_counts["annotations_removed_invalid"] += int(not valid)
                        visualization_boxes.setdefault(f"{split}/{file_name}", []).append((original_box, clipped_box))
                        status = "clipped" if valid else "removed_invalid"
                    else:
                        counts["annotations_unchanged"] += 1
                        split_counts["annotations_unchanged"] += 1
                        status = "unchanged"

                    log_rows.append({
                        "split": split,
                        "source_line": line_number,
                        "file_name": file_name,
                        "annotation_index": annotation_index,
                        "class": category,
                        "original_bbox_xywh": json.dumps(_format_box(original_box), separators=(",", ":")),
                        "cleaned_bbox_xywh": json.dumps(_format_box(clipped_box), separators=(",", ":")),
                        "status": status,
                        "image_width": image_width,
                        "image_height": image_height,
                        "original_area": area,
                        "cleaned_area": round(clipped_box[2] * clipped_box[3], 6),
                    })
                    if valid:
                        cleaned_boxes.append(_format_box(clipped_box))
                        cleaned_categories.append(category)
                        cleaned_areas.append(round(clipped_box[2] * clipped_box[3], 6))

                cleaned_record = dict(record)
                cleaned_objects = dict(objects)
                cleaned_objects["bbox"] = cleaned_boxes
                cleaned_objects["category"] = cleaned_categories
                cleaned_objects["area"] = cleaned_areas
                cleaned_record["objects"] = cleaned_objects
                output.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
        split_reports[split] = dict(split_counts)

    fieldnames = [
        "split", "source_line", "file_name", "annotation_index", "class",
        "original_bbox_xywh", "cleaned_bbox_xywh", "status", "image_width",
        "image_height", "original_area", "cleaned_area",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    selected_visualizations = sorted(visualization_boxes)[:visualization_limit]
    for relative_name in selected_visualizations:
        split, file_name = relative_name.split("/", 1)
        image_path = _image_path(dataset_dir / split, file_name)
        output_path = visualization_dir / f"{split}__{Path(file_name).stem}.jpg"
        _write_visualization(image_path, visualization_boxes[relative_name], output_path)

    validation = validate_cleaned_dataset(cleaned_dir, dataset_dir, class_names)
    report = {
        "source_dataset": str(dataset_dir),
        "cleaned_annotation_directory": str(cleaned_dir),
        "total_annotations_inspected": counts["total_annotations_inspected"],
        "out_of_bounds_annotations": counts["out_of_bounds_annotations"],
        "annotations_clipped": counts["annotations_clipped"],
        "annotations_removed_because_invalid": counts["annotations_removed_invalid"],
        "annotations_unchanged": counts["annotations_unchanged"],
        "images_affected": len(images_affected),
        "affected_image_names": sorted(images_affected),
        "class_names": sorted(class_names),
        "split_counts": split_reports,
        "visualizations_generated": len(selected_visualizations),
        "validation": validation,
        "original_files_modified": False,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def validate_cleaned_dataset(cleaned_dir: Path, source_dir: Path, class_names: set) -> Dict[str, object]:
    """Validate cleaned JSONL files against source images and class names."""
    result = {
        "passed": True,
        "metadata_files_checked": 0,
        "annotations_checked": 0,
        "out_of_bounds_boxes": 0,
        "non_positive_boxes": 0,
        "invalid_classes": 0,
        "missing_images": 0,
        "malformed_annotations": 0,
        "errors": [],
    }
    for metadata_file in sorted(Path(cleaned_dir).glob("*/metadata.jsonl")):
        result["metadata_files_checked"] += 1
        source_split = Path(source_dir) / metadata_file.parent.name
        with metadata_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    file_name = record["file_name"]
                    objects = record["objects"]
                    boxes = objects["bbox"]
                    categories = objects["category"]
                    areas = objects["area"]
                    if not (len(boxes) == len(categories) == len(areas)):
                        raise ValueError("annotation arrays have different lengths")
                    image_path = _image_path(source_split, file_name)
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        result["missing_images"] += 1
                        raise ValueError(f"missing image: {file_name}")
                    height, width = image.shape[:2]
                    for index, (box, category) in enumerate(zip(boxes, categories)):
                        result["annotations_checked"] += 1
                        if str(category) not in class_names:
                            result["invalid_classes"] += 1
                            raise ValueError(f"invalid class at annotation {index}: {category}")
                        values = _numbers(box)
                        if len(values) != 4:
                            raise ValueError(f"bbox at annotation {index} is not length four")
                        if not _box_is_inside(values, width, height):
                            result["out_of_bounds_boxes"] += 1
                            raise ValueError(f"box outside image at annotation {index}: {box}")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result["malformed_annotations"] += 1
                    if len(result["errors"]) < 20:
                        result["errors"].append(f"{metadata_file}:{line_number}: {exc}")
    result["passed"] = not any(
        result[key] for key in (
            "out_of_bounds_boxes", "non_positive_boxes", "invalid_classes",
            "missing_images", "malformed_annotations",
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Ghost Pot JSONL annotations")
    parser.add_argument("--dataset", default="data/raw/ghost_pot")
    parser.add_argument("--cleaned-dir", default="data/processed/annotations_cleaned")
    parser.add_argument("--report", default="outputs/dataset/annotation_cleaning_report.json")
    parser.add_argument("--log", default="outputs/dataset/annotation_cleaning_log.csv")
    parser.add_argument("--visualizations", default="outputs/visualizations/annotation_cleaning")
    parser.add_argument("--visualization-limit", type=int, default=12)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"))
    report = clean_dataset(
        Path(args.dataset), Path(args.cleaned_dir), Path(args.report),
        Path(args.log), Path(args.visualizations), args.visualization_limit,
    )
    print(json.dumps(report, indent=2))
    if not report["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
