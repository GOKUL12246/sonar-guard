"""Tests for portable runtime dataset YAML resolution."""

from pathlib import Path

import yaml

from src.training.dataset_paths import resolve_dataset_yaml


def test_resolve_dataset_yaml_rebases_stale_absolute_path(tmp_path):
    dataset_root = tmp_path / "data" / "yolo"
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True)
    source_yaml = dataset_root / "data.yaml"
    source_yaml.write_text(
        yaml.safe_dump({
            "path": "C:/old-machine/project/data/yolo",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["Crab-Pot"],
        }, sort_keys=False),
        encoding="utf-8",
    )

    runtime_yaml = resolve_dataset_yaml(source_yaml, tmp_path / "runtime")

    assert runtime_yaml != source_yaml
    resolved = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    assert Path(resolved["path"]).resolve() == dataset_root.resolve()
    assert resolved["train"] == "images/train"
    assert source_yaml.read_text(encoding="utf-8").find("old-machine") >= 0
