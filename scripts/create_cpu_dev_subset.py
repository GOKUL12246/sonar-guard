"""
SONAR-GUARD — CPU Development Dataset Subset Creator
======================================================
Creates a small reproducible subset of the full Ghost Pot YOLO dataset
for CPU development / smoke training runs.

Usage:
    python scripts/create_cpu_dev_subset.py

Output:
    data/yolo_cpu_dev/
        images/train/   (400 images)
        images/val/     (80 images)
        labels/train/   (400 label files)
        labels/val/     (80 label files)
        data.yaml

Design:
  - Copies (does NOT move or modify) original dataset files.
  - Selects files with a fixed random seed for reproducibility.
  - Only includes images that have a matching label file.
  - Skips images with zero-byte label files (unannotated negatives).
  - Writes a valid YOLO data.yaml pointing to absolute paths.
  - Safe to re-run — skips copy if destination already exists.
"""

import random
import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────
SEED        = 42
N_TRAIN     = 400     # images to include in dev train split
N_VAL       = 80      # images to include in dev val split

SRC_ROOT    = Path("data/yolo")
DST_ROOT    = Path("data/yolo_cpu_dev")

CLASS_NAMES = ["Crab-Pot"]

# ── Helpers ───────────────────────────────────────────────────────────────

def select_paired_images(img_dir: Path, lbl_dir: Path, n: int, seed: int) -> list:
    """
    Return n (image_path, label_path) pairs.
    Only includes images that have a matching, non-empty label file.
    """
    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(seed)
    random.shuffle(all_images)

    paired = []
    for img in all_images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            paired.append((img, lbl))
        if len(paired) >= n:
            break

    return paired


def copy_pairs(pairs: list, dst_img_dir: Path, dst_lbl_dir: Path):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_src, lbl_src in pairs:
        img_dst = dst_img_dir / img_src.name
        lbl_dst = dst_lbl_dir / lbl_src.name
        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)
        if not lbl_dst.exists():
            shutil.copy2(lbl_src, lbl_dst)


def write_yaml(dst_root: Path, class_names: list):
    import yaml
    yaml_data = {
        "path":  str(dst_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"[yaml] Written: {yaml_path}")
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SONAR-GUARD — Creating CPU Dev Subset")
    print("=" * 60)
    print(f"  Source root  : {SRC_ROOT.resolve()}")
    print(f"  Target root  : {DST_ROOT.resolve()}")
    print(f"  Train images : {N_TRAIN}")
    print(f"  Val   images : {N_VAL}")
    print(f"  Seed         : {SEED}")
    print()

    # ── Train split ───────────────────────────────────────────────────
    train_img_src = SRC_ROOT / "images" / "train"
    train_lbl_src = SRC_ROOT / "labels" / "train"

    if not train_img_src.exists():
        print(f"ERROR: Train images directory not found: {train_img_src}")
        raise SystemExit(1)

    print(f"[train] Scanning {train_img_src} …")
    train_pairs = select_paired_images(train_img_src, train_lbl_src, N_TRAIN, SEED)
    print(f"[train] Selected {len(train_pairs)} paired image/label files.")

    if len(train_pairs) < N_TRAIN:
        print(f"[warn] Only {len(train_pairs)} valid pairs found (wanted {N_TRAIN}).")

    dst_train_imgs = DST_ROOT / "images" / "train"
    dst_train_lbls = DST_ROOT / "labels" / "train"
    copy_pairs(train_pairs, dst_train_imgs, dst_train_lbls)
    print(f"[train] Copied to {dst_train_imgs}")

    # ── Val split ─────────────────────────────────────────────────────
    val_img_src = SRC_ROOT / "images" / "val"
    val_lbl_src = SRC_ROOT / "labels" / "val"

    if not val_img_src.exists():
        print(f"ERROR: Val images directory not found: {val_img_src}")
        raise SystemExit(1)

    print(f"[val]   Scanning {val_img_src} …")
    val_pairs = select_paired_images(val_img_src, val_lbl_src, N_VAL, SEED + 1)
    print(f"[val]   Selected {len(val_pairs)} paired image/label files.")

    dst_val_imgs = DST_ROOT / "images" / "val"
    dst_val_lbls = DST_ROOT / "labels" / "val"
    copy_pairs(val_pairs, dst_val_imgs, dst_val_lbls)
    print(f"[val]   Copied to {dst_val_imgs}")

    # ── YAML ──────────────────────────────────────────────────────────
    yaml_path = write_yaml(DST_ROOT, CLASS_NAMES)

    # ── Verify ────────────────────────────────────────────────────────
    actual_train = len(list((DST_ROOT / "images" / "train").glob("*.jpg")) +
                       list((DST_ROOT / "images" / "train").glob("*.png")))
    actual_val   = len(list((DST_ROOT / "images" / "val").glob("*.jpg")) +
                       list((DST_ROOT / "images" / "val").glob("*.png")))
    actual_tlbl  = len(list((DST_ROOT / "labels" / "train").glob("*.txt")))
    actual_vlbl  = len(list((DST_ROOT / "labels" / "val").glob("*.txt")))

    print()
    print("=" * 60)
    print("Subset Creation Complete")
    print("=" * 60)
    print(f"  Train images : {actual_train}")
    print(f"  Train labels : {actual_tlbl}")
    print(f"  Val   images : {actual_val}")
    print(f"  Val   labels : {actual_vlbl}")
    print(f"  YAML         : {yaml_path}")
    print()
    print("All values match :", actual_train == actual_tlbl and actual_val == actual_vlbl)
    print()
    print("IMPORTANT: This is a DEVELOPMENT SUBSET — NOT the full dataset.")
    print("  Full dataset: 5721 train / 555 val images (untouched).")
    print()
    print("Next: run training with:")
    print(f"  python src/training/train_yolo.py \\")
    print(f"    --dataset {yaml_path} \\")
    print(f"    --model-size n \\")
    print(f"    --epochs 5 \\")
    print(f"    --img-size 320 \\")
    print(f"    --batch 4 \\")
    print(f"    --device cpu \\")
    print(f"    --seed 42 \\")
    print(f"    --output-dir outputs/yolo_cpu_dev")


if __name__ == "__main__":
    main()
