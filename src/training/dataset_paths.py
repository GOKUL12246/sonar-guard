"""Portable runtime resolution for YOLO dataset YAML files."""

from pathlib import Path
from typing import Dict

import yaml


def resolve_dataset_yaml(dataset_yaml: Path, runtime_dir: Path = Path("outputs/runtime")) -> Path:
    """Return a usable YAML without modifying the source dataset YAML.

    The prepared YAML may contain an absolute path from the machine where it
    was generated. When that path is unavailable, resolve the same relative
    train/val/test directories from the YAML file's local project location and
    write a runtime-only YAML copy.
    """
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data: Dict[str, object] = yaml.safe_load(handle) or {}

    source_root = Path(str(data.get("path", ".")))
    if not source_root.is_absolute():
        source_root = dataset_yaml.parent / source_root
    local_root = dataset_yaml.parent.resolve()

    def resolve_split(value: object) -> Path:
        path = Path(str(value))
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(source_root / path)
            candidates.append(local_root / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Dataset split path not found for {value!r}; checked {candidates}"
        )

    split_paths = {
        key: resolve_split(data[key])
        for key in ("train", "val", "test")
        if key in data
    }
    configured_root = source_root.resolve() if source_root.exists() else None
    if configured_root == local_root and all(
        split_paths[key].exists() for key in split_paths
    ):
        return dataset_yaml

    runtime_data = dict(data)
    runtime_data["path"] = str(local_root)
    runtime_data.update({
        key: str(path.relative_to(local_root)).replace("\\", "/")
        for key, path in split_paths.items()
    })
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"{dataset_yaml.stem}_runtime.yaml"
    with runtime_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_data, handle, sort_keys=False)
    return runtime_yaml
