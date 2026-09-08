"""Focused tests for dataset inspection and preparation."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.training.dataset_validator import (
    DatasetSplitter,
    YOLODatasetValidator,
    convert_coco_to_yolo,
    convert_jsonl_to_yolo,
    visualize_yolo_annotations,
)


def _write_image(path: Path, width: int = 200, height: int = 100) -> None:
    assert cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def test_yolo_validator_reports_real_statistics(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "survey_001.png")
    _write_image(images / "unlabeled.png")
    (labels / "survey_001.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    report = YOLODatasetValidator(["debris"]).validate(tmp_path)

    assert report.total_images == 2
    assert report.labeled_images == 1
    assert report.unlabeled_images == 1
    assert report.total_annotations == 1
    assert report.annotations_per_class == {"debris": 1}
    assert report.image_dims["min_w"] == 200


def test_invalid_yolo_box_is_reported(tmp_path):
    image = tmp_path / "image.png"
    _write_image(image)
    (tmp_path / "image.txt").write_text("0 1.2 0.5 0.2 0.4\n", encoding="utf-8")

    report = YOLODatasetValidator(["debris"]).validate(tmp_path)

    assert report.invalid_annotations == 1
    assert report.total_annotations == 1
    assert report.annotations_per_class == {}


def test_coco_and_jsonl_conversion_write_normalized_labels(tmp_path):
    image = tmp_path / "survey_001.png"
    _write_image(image)
    coco_file = tmp_path / "annotations.json"
    coco_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image.name, "width": 200, "height": 100}],
        "categories": [{"id": 9, "name": "debris"}],
        "annotations": [{"image_id": 1, "category_id": 9, "bbox": [20, 10, 40, 20]}],
    }), encoding="utf-8")

    labels = tmp_path / "coco_labels"
    assert convert_coco_to_yolo(coco_file, labels) == ["debris"]
    expected = "0 0.200000 0.200000 0.200000 0.200000"
    assert (labels / "survey_001.txt").read_text(encoding="utf-8") == expected

    jsonl_file = tmp_path / "annotations.jsonl"
    jsonl_file.write_text(
        json.dumps({"image_id": image.name, "class": "debris", "bbox": [20, 10, 40, 20]}) + "\n",
        encoding="utf-8",
    )
    jsonl_labels = tmp_path / "jsonl_labels"
    convert_jsonl_to_yolo(jsonl_file, jsonl_labels, ["debris"], {image.name: (200, 100)})
    assert (jsonl_labels / "survey_001.txt").read_text(encoding="utf-8") == expected


def test_visualization_writes_overlay(tmp_path):
    image = tmp_path / "image.png"
    label = tmp_path / "image.txt"
    output = tmp_path / "visualizations" / "image.png"
    _write_image(image)
    label.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    rendered = visualize_yolo_annotations(image, label, ["debris"], output)

    assert rendered.shape == (100, 200, 3)
    assert output.exists()
    assert int(rendered.sum()) > 0


def test_splitter_keeps_numbered_survey_frames_together(tmp_path):
    paths = []
    for survey in ("survey_a", "survey_b", "survey_c"):
        for frame in range(3):
            path = tmp_path / f"{survey}_{frame:03d}.png"
            _write_image(path)
            paths.append(path)

    first = DatasetSplitter(seed=42).split(paths)
    second = DatasetSplitter(seed=42).split(paths)

    assert first == second
    survey_splits = {}
    for split_name, split_paths in first.items():
        for path in split_paths:
            survey_name = path.stem.rsplit("_", 1)[0]
            assert survey_name not in survey_splits or survey_splits[survey_name] == split_name
            survey_splits[survey_name] = split_name
    assert set(survey_splits) == {"survey_a", "survey_b", "survey_c"}
    assert sorted(path for values in first.values() for path in values) == sorted(paths)
