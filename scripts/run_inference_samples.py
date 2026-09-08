import cv2
import json
import random
from pathlib import Path
from ultralytics import YOLO

def main():
    random.seed(42)
    model_path = Path("outputs/yolo_cpu_dev/weights/yolov8n_cpu_dev_320_best.pt")
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        return

    model = YOLO(str(model_path))

    val_imgs = list(Path("data/yolo_cpu_dev/images/val").glob("*.jpg")) + \
               list(Path("data/yolo_cpu_dev/images/val").glob("*.png"))
    random.shuffle(val_imgs)
    samples = val_imgs[:6]

    out_dir = Path("outputs/yolo_cpu_dev/inference_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_log = []
    print("=== Running Inference on 6 Sample Val Images ===")
    print(f"Model : {model_path}")
    print("Conf  : 0.15 | IoU: 0.45 | Device: cpu\n")

    for i, img_path in enumerate(samples):
        results = model.predict(
            source  = str(img_path),
            imgsz   = 320,
            conf    = 0.15,
            iou     = 0.45,
            device  = "cpu",
            verbose = False,
            save    = False,
        )
        r = results[0]
        boxes = r.boxes

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        n_dets = len(boxes) if boxes is not None else 0
        det_list = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 80), 2)
                lbl = f"{cls_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 200, 80), -1)
                cv2.putText(img, lbl, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                det_list.append({
                    "class": cls_name,
                    "conf": round(conf_val, 4),
                    "bbox": [x1, y1, x2, y2]
                })

        stamp = f"CPU DEV INFERENCE | {img_path.name[:30]} | dets={n_dets}"
        cv2.putText(img, stamp, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 230, 50), 1)

        out_path = out_dir / f"sample_{i+1:02d}_{img_path.stem[:30]}.jpg"
        cv2.imwrite(str(out_path), img)

        inference_log.append({
            "image": img_path.name,
            "detections": n_dets,
            "boxes": det_list,
            "output": str(out_path)
        })
        print(f"[{i+1}/6] {img_path.name[:45]} -> {n_dets} detection(s)")
        for d in det_list:
            print(f"       {d['class']} conf={d['conf']} bbox={d['bbox']}")

    log_path = out_dir / "inference_log.json"
    with open(log_path, "w") as f:
        json.dump(inference_log, f, indent=2)
    print(f"\nInference log saved to: {log_path}")

if __name__ == "__main__":
    main()
