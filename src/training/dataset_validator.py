"""
SONAR-GUARD — Dataset Validator
================================
Inspects and validates a sonar image dataset before training.

Reports:
  - Total images
  - Images with / without labels
  - Annotation count per class
  - Invalid/corrupted files
  - Bounding box violations
  - Image dimension statistics
  - Class imbalance
  - Train / val / test split

Supported annotation formats:
  - YOLO (.txt) — class_id cx cy w h (all normalised 0–1)
  - COCO JSON   — standard COCO format
  - JSON/JSONL  — flat list of {image_id, bbox: [x,y,w,h], class}

NEVER generates or invents statistics.
Every number in the report comes from actual file inspection.
"""

import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnnotationRecord:
    """Single bounding-box annotation."""
    image_id:    str
    class_id:    int
    class_name:  str
    cx:          float   # normalised x-centre (0–1)
    cy:          float   # normalised y-centre (0–1)
    width:       float   # normalised width  (0–1)
    height:      float   # normalised height (0–1)
    source_file: Path
    is_valid:    bool = True
    error:       Optional[str] = None


@dataclass
class DatasetReport:
    """
    Full dataset validation report.
    All statistics come from actual file inspection — never fabricated.
    """
    dataset_dir:           str = ""
    annotation_format:     str = "unknown"
    total_images:          int = 0
    labeled_images:        int = 0
    unlabeled_images:      int = 0
    corrupted_images:      int = 0
    total_annotations:     int = 0
    classes:               List[str]          = field(default_factory=list)
    annotations_per_class: Dict[str, int]     = field(default_factory=dict)
    images_per_class:      Dict[str, int]     = field(default_factory=dict)
    invalid_annotations:   int = 0
    annotation_errors:     List[str]          = field(default_factory=list)
    image_dims:            Dict[str, object]  = field(default_factory=dict)  # min/max/mean
    split_counts:          Dict[str, int]     = field(default_factory=dict)  # train/val/test
    class_imbalance_ratio: Optional[float]    = None   # max_class / min_class count
    warnings:              List[str]          = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "SONAR-GUARD — Dataset Validation Report",
            "=" * 60,
            f"Directory         : {self.dataset_dir}",
            f"Annotation format : {self.annotation_format}",
            "",
            f"Total images      : {self.total_images}",
            f"  Labeled         : {self.labeled_images}",
            f"  Unlabeled       : {self.unlabeled_images}",
            f"  Corrupted       : {self.corrupted_images}",
            "",
            f"Total annotations : {self.total_annotations}",
            f"Invalid annots    : {self.invalid_annotations}",
            "",
            "Classes:",
        ]
        for cls in self.classes:
            n  = self.annotations_per_class.get(cls, 0)
            ni = self.images_per_class.get(cls, 0)
            lines.append(f"  {cls:30s}: {n:5d} annotations in {ni:4d} images")

        if self.class_imbalance_ratio is not None:
            lines.append(f"\nClass imbalance ratio: {self.class_imbalance_ratio:.1f}x")

        if self.image_dims:
            dims = self.image_dims
            lines.append(
                f"\nImage dimensions  : "
                f"min={dims.get('min_w')}x{dims.get('min_h')} "
                f"max={dims.get('max_w')}x{dims.get('max_h')} "
                f"mean={dims.get('mean_w', 0):.0f}x{dims.get('mean_h', 0):.0f}"
            )

        if self.split_counts:
            lines.append("\nSplit counts:")
            for split, count in self.split_counts.items():
                lines.append(f"  {split:8s}: {count}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_dir":           self.dataset_dir,
            "annotation_format":     self.annotation_format,
            "total_images":          self.total_images,
            "labeled_images":        self.labeled_images,
            "unlabeled_images":      self.unlabeled_images,
            "corrupted_images":      self.corrupted_images,
            "total_annotations":     self.total_annotations,
            "invalid_annotations":   self.invalid_annotations,
            "classes":               self.classes,
            "annotations_per_class": self.annotations_per_class,
            "images_per_class":      self.images_per_class,
            "image_dims":            self.image_dims,
            "split_counts":          self.split_counts,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "warnings":              self.warnings,
            "annotation_errors":     self.annotation_errors[:50],  # Cap for report size
        }


# ---------------------------------------------------------------------------
# Annotation validators
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
YOLO_EXTENSIONS      = {".txt"}


def _normalise_bbox(x: float, y: float, width: float, height: float,
                    image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """Convert an absolute xywh box into YOLO-normalised cx, cy, w, h."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    return (
        (x + width / 2.0) / image_width,
        (y + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def convert_coco_to_yolo(coco_file: Path, labels_dir: Path,
                         class_names: Optional[List[str]] = None) -> List[str]:
    """Convert COCO bounding boxes to one YOLO label file per image."""
    with open(coco_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {item["id"]: item for item in coco.get("images", [])}
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    category_to_index = {item["id"]: index for index, item in enumerate(categories)}
    names = class_names or [item["name"] for item in categories]
    if len(names) != len(categories):
        raise ValueError("class_names must match the number of COCO categories")

    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for annotation in coco.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if image is None or category_id not in category_to_index or not bbox or len(bbox) != 4:
            continue
        cx, cy, box_width, box_height = _normalise_bbox(
            *map(float, bbox), int(image["width"]), int(image["height"])
        )
        grouped[image["id"]].append(
            f"{category_to_index[category_id]} {cx:.6f} {cy:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    for image_id, image in images.items():
        output = labels_dir / f"{Path(image['file_name']).stem}.txt"
        output.write_text("\n".join(grouped.get(image_id, [])), encoding="utf-8")
    return names


def convert_jsonl_to_yolo(jsonl_file: Path, labels_dir: Path,
                          class_names: List[str],
                          image_sizes: Dict[str, Tuple[int, int]]) -> None:
    """Convert flat JSONL records with absolute xywh boxes to YOLO labels."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    with open(jsonl_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = str(record["image_id"])
            bbox = record["bbox"]
            if len(bbox) != 4:
                raise ValueError(f"Line {line_number}: bbox must contain four values")
            class_id = record.get("class_id")
            if class_id is None:
                class_id = class_names.index(record["class"])
            width, height = image_sizes[image_id]
            cx, cy, box_width, box_height = _normalise_bbox(
                *map(float, bbox), width, height
            )
            grouped[image_id].append(
                f"{int(class_id)} {cx:.6f} {cy:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

    for image_id, rows in grouped.items():
        (labels_dir / f"{Path(image_id).stem}.txt").write_text(
            "\n".join(rows), encoding="utf-8"
        )


def visualize_yolo_annotations(image_path: Path, label_path: Path,
                               class_names: Optional[List[str]] = None,
                               output_path: Optional[Path] = None) -> np.ndarray:
    """Draw valid YOLO annotations over an image for manual inspection."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    names = class_names or []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != 5:
                log.warning("Skipping malformed label %s:%d", label_path, line_number)
                continue
            class_id, cx, cy, box_width, box_height = parts
            cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
            x1 = max(0, int((cx - box_width / 2) * width))
            y1 = max(0, int((cy - box_height / 2) * height))
            x2 = min(width - 1, int((cx + box_width / 2) * width))
            y2 = min(height - 1, int((cy + box_height / 2) * height))
            class_index = int(class_id)
            label = names[class_index] if 0 <= class_index < len(names) else str(class_index)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Unable to write visualization: {output_path}")
    return image


def _validate_yolo_box(cx: float, cy: float, w: float, h: float,
                        class_id: int, n_classes: int,
                        image_id: str, line_no: int) -> Optional[str]:
    """
    Validate a YOLO-format bounding box.
    Returns error string if invalid, None if valid.
    """
    errors = []
    if not (0.0 <= cx <= 1.0):
        errors.append(f"cx={cx:.4f} out of [0,1]")
    if not (0.0 <= cy <= 1.0):
        errors.append(f"cy={cy:.4f} out of [0,1]")
    if not (0.0 < w <= 1.0):
        errors.append(f"w={w:.4f} not in (0,1]")
    if not (0.0 < h <= 1.0):
        errors.append(f"h={h:.4f} not in (0,1]")
    if n_classes > 0 and (class_id < 0 or class_id >= n_classes):
        errors.append(f"class_id={class_id} out of range [0,{n_classes-1}]")
    if errors:
        return f"{image_id}:line{line_no} — " + "; ".join(errors)
    return None


# ---------------------------------------------------------------------------
# YOLO format validator
# ---------------------------------------------------------------------------

class YOLODatasetValidator:
    """
    Validates a dataset in YOLO format (images + label .txt files side-by-side
    or in a parallel labels/ directory).

    Expected structure (either):
        dataset/
            images/  *.png | *.jpg | ...
            labels/  *.txt

    Or flat:
        dataset/
            image1.png
            image1.txt
            ...
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or []

    def validate(self, dataset_dir: Path) -> DatasetReport:
        """
        Inspect the dataset directory and return a full validation report.

        Args:
            dataset_dir: Root directory of the dataset.

        Returns:
            DatasetReport — all stats from actual file inspection.
        """
        dataset_dir = Path(dataset_dir)
        report = DatasetReport(
            dataset_dir=str(dataset_dir),
            annotation_format="YOLO",
        )

        if not dataset_dir.exists():
            report.warnings.append(f"Dataset directory not found: {dataset_dir}")
            return report

        # Locate image files
        image_files = sorted([
            p for p in dataset_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        report.total_images = len(image_files)
        if report.total_images == 0:
            report.warnings.append("No image files found in dataset directory.")
            return report

        log.info("Validating %d images in %s", report.total_images, dataset_dir)

        # Dimension collection
        widths, heights = [], []

        # Annotation tracking
        all_annotations: List[AnnotationRecord] = []
        class_counter   = Counter()
        class_img_sets  = defaultdict(set)

        for img_path in image_files:
            # ── Try to open image ─────────────────────────────────────
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("cv2.imread returned None")
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
            except Exception as exc:
                log.warning("Corrupted image %s: %s", img_path.name, exc)
                report.corrupted_images += 1
                continue

            # ── Find label file ───────────────────────────────────────
            label_path = self._find_label(img_path)

            if label_path is None or not label_path.exists():
                report.unlabeled_images += 1
                continue

            # ── Parse YOLO label file ─────────────────────────────────
            report.labeled_images += 1
            img_id = img_path.stem
            n_classes = len(self.class_names)

            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as exc:
                report.annotation_errors.append(
                    f"{img_path.name}: cannot read label file — {exc}"
                )
                continue

            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    err = (f"{img_path.name}:line{line_no} — "
                           f"expected 5 values, got {len(parts)}")
                    report.annotation_errors.append(err)
                    report.invalid_annotations += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    err = f"{img_path.name}:line{line_no} — non-numeric values"
                    report.annotation_errors.append(err)
                    report.invalid_annotations += 1
                    continue

                val_err = _validate_yolo_box(cx, cy, bw, bh, cls_id,
                                              n_classes, img_path.name, line_no)
                if val_err:
                    report.annotation_errors.append(val_err)
                    report.invalid_annotations += 1
                    is_valid = False
                else:
                    is_valid = True

                cls_name = (self.class_names[cls_id]
                            if 0 <= cls_id < len(self.class_names)
                            else f"class_{cls_id}")

                rec = AnnotationRecord(
                    image_id=img_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    cx=cx, cy=cy,
                    width=bw, height=bh,
                    source_file=label_path,
                    is_valid=is_valid,
                    error=val_err,
                )
                all_annotations.append(rec)

                if is_valid:
                    class_counter[cls_name] += 1
                    class_img_sets[cls_name].add(img_id)

        # ── Finalise report ───────────────────────────────────────────
        report.total_annotations = len(all_annotations)
        report.annotations_per_class = dict(class_counter)
        report.images_per_class = {k: len(v) for k, v in class_img_sets.items()}
        report.classes = sorted(report.annotations_per_class.keys())

        if widths:
            report.image_dims = {
                "min_w":  int(min(widths)),  "min_h":  int(min(heights)),
                "max_w":  int(max(widths)),  "max_h":  int(max(heights)),
                "mean_w": float(np.mean(widths)),
                "mean_h": float(np.mean(heights)),
            }

        # Class imbalance
        counts = list(class_counter.values())
        if len(counts) >= 2:
            report.class_imbalance_ratio = max(counts) / max(1, min(counts))

        # Warnings
        if report.corrupted_images > 0:
            report.warnings.append(
                f"{report.corrupted_images} corrupted/unreadable images found."
            )
        if report.unlabeled_images > 0:
            report.warnings.append(
                f"{report.unlabeled_images} images have no label file — "
                "they will be skipped in supervised training."
            )
        if report.invalid_annotations > 0:
            report.warnings.append(
                f"{report.invalid_annotations} invalid annotation lines — "
                "they will be excluded from training."
            )
        if report.class_imbalance_ratio and report.class_imbalance_ratio > 5:
            report.warnings.append(
                f"High class imbalance ({report.class_imbalance_ratio:.1f}x). "
                "Consider augmentation or class weighting."
            )

        log.info("Validation complete. %s", report.summary())
        return report

    @staticmethod
    def _find_label(image_path: Path) -> Optional[Path]:
        """
        Locate the YOLO label file for an image.

        Tries:
          1. Same directory, same stem, .txt extension
          2. Sibling 'labels/' directory at same level as 'images/'
        """
        # Same directory
        candidate = image_path.with_suffix(".txt")
        if candidate.exists():
            return candidate

        # images/ → labels/ swap
        parts = image_path.parts
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            label_path = Path(*parts[:idx]) / "labels" / Path(*parts[idx + 1:])
            label_path = label_path.with_suffix(".txt")
            if label_path.exists():
                return label_path

        return None


# ---------------------------------------------------------------------------
# Train / Val / Test splitter
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """
    Creates a reproducible train/val/test split of annotated images.

    Avoids leakage by grouping images from the same sequence/survey together
    when sequence information is available (images with common prefix up to _NNN).
    """

    def __init__(self,
                 train_ratio: float = 0.70,
                 val_ratio:   float = 0.15,
                 test_ratio:  float = 0.15,
                 seed:        int   = 42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed

    def split(self,
              image_paths:     List[Path],
              label_paths:     Optional[List[Optional[Path]]] = None,
              output_dir:      Optional[Path] = None,
              copy_files:      bool = False) -> Dict[str, List[Path]]:
        """
        Split image_paths into train/val/test sets.

        Args:
            image_paths:   List of image file paths to split.
            label_paths:   Corresponding label paths (same order). None = skip copying labels.
            output_dir:    If provided and copy_files=True, copy files into
                           output_dir/{train,val,test}/{images,labels}/
            copy_files:    If True, copy files to output_dir splits.

        Returns:
            Dict with keys 'train', 'val', 'test' mapping to lists of image paths.
        """
        if not image_paths:
            log.warning("No images to split.")
            return {"train": [], "val": [], "test": []}

        rng = random.Random(self.seed)
        groups = defaultdict(list)
        for image_path in image_paths:
            groups[self._group_key(Path(image_path))].append(Path(image_path))
        group_list = list(groups.values())
        rng.shuffle(group_list)

        n = len(image_paths)
        target_counts = {
            "train": int(n * self.train_ratio),
            "val": int(n * self.val_ratio),
        }
        target_counts["test"] = n - target_counts["train"] - target_counts["val"]
        splits = {"train": [], "val": [], "test": []}
        for group in group_list:
            split_name = max(
                splits,
                key=lambda name: target_counts[name] - len(splits[name]),
            )
            splits[split_name].extend(group)

        n_train = len(splits["train"])
        n_val = len(splits["val"])
        n_test = len(splits["test"])

        log.info(
            "Split (seed=%d): %d train / %d val / %d test",
            self.seed, n_train, n_val, n_test,
        )

        if copy_files and output_dir is not None:
            self._copy_splits(splits, label_paths, image_paths, output_dir)

        return splits

    @staticmethod
    def _group_key(image_path: Path) -> str:
        """Group numbered frames by the survey prefix before the frame index."""
        match = re.match(r"^(.*?)(?:[_-]\d{3,})$", image_path.stem)
        return match.group(1) if match else image_path.stem

    def _copy_splits(self,
                     splits:       Dict[str, List[Path]],
                     label_paths:  Optional[List[Optional[Path]]],
                     all_images:   List[Path],
                     output_dir:   Path) -> None:
        """Copy split files into output directory structure."""
        # Build lookup: image_path → label_path
        label_map: Dict[Path, Optional[Path]] = {}
        if label_paths:
            label_map = dict(zip(all_images, label_paths))

        for split_name, img_list in splits.items():
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_list:
                shutil.copy2(img_path, img_dir / img_path.name)
                lbl = label_map.get(img_path)
                if lbl and lbl.exists():
                    shutil.copy2(lbl, lbl_dir / lbl.name)

        log.info("Split files copied to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# Backwards-compatible facade name used by older project documentation.
DatasetValidator = YOLODatasetValidator

def main():
    """Quick CLI: python -m src.training.dataset_validator <dataset_dir> [class1 class2 ...]"""
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.training.dataset_validator <dataset_dir> [class1 class2 ...]")
        sys.exit(1)

    dataset_dir  = Path(sys.argv[1])
    class_names  = sys.argv[2:] if len(sys.argv) > 2 else []

    validator = YOLODatasetValidator(class_names=class_names)
    report    = validator.validate(dataset_dir)
    print(report.summary())

    out_file = dataset_dir / "dataset_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved: {out_file}")


if __name__ == "__main__":
    main()

