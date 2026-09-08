"""
SONAR-GUARD — Active Learning Feedback Pipeline
=================================================
Stores human-verified detections for future model retraining.

Design principles:
  - Human feedback is STORED immediately when buttons are clicked.
  - The model is NOT automatically retrained after every click.
  - Retraining is a controlled, deliberate action.
  - The system tracks dataset growth and provides a "Ready for retraining" signal
    when enough new verified samples have accumulated.

Feedback commands:
    python -m src.active_learning.feedback --status
    python -m src.active_learning.feedback --export
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

FEEDBACK_FILE_DEFAULT = Path("data/verified/feedback.jsonl")


class FeedbackStore:
    """
    Persistent store for human-verified detection feedback.

    Storage format: JSONL (one JSON object per line).
    Each record is self-contained and includes all pipeline outputs.
    """

    def __init__(
        self,
        feedback_file:        Path = FEEDBACK_FILE_DEFAULT,
        retraining_threshold: int  = 50,
    ):
        self.feedback_file        = Path(feedback_file)
        self.retraining_threshold = retraining_threshold
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        log.info("FeedbackStore initialised: %s", self.feedback_file)

    def store(
        self,
        image_id:          str,
        image_path:        Optional[str],
        detection_dict:    dict,
        scores_dict:       dict,
        user_decision:     str,        # "confirmed" | "rejected" | "rov_inspection"
        user_notes:        str = "",
        session_id:        Optional[str] = None,
    ) -> str:
        """
        Store a human feedback record.

        Args:
            image_id:       Source image identifier.
            image_path:     Path to source image file (for future reference).
            detection_dict: Serialised Detection.to_dict() output.
            scores_dict:    Dict of all computed scores for this detection.
            user_decision:  "confirmed" | "rejected" | "rov_inspection"
            user_notes:     Optional free-text notes from the operator.
            session_id:     Optional dashboard session identifier.

        Returns:
            feedback_id — unique identifier for this record.
        """
        valid_decisions = {"confirmed", "rejected", "rov_inspection"}
        if user_decision not in valid_decisions:
            raise ValueError(
                f"Invalid user_decision '{user_decision}'. "
                f"Must be one of: {valid_decisions}"
            )

        feedback_id = str(uuid.uuid4())
        timestamp   = datetime.now(tz=timezone.utc).isoformat()

        record = {
            "feedback_id":    feedback_id,
            "timestamp_utc":  timestamp,
            "session_id":     session_id,
            "image_id":       image_id,
            "image_path":     image_path,
            "user_decision":  user_decision,
            "user_notes":     user_notes,
            "detection":      detection_dict,
            "scores":         scores_dict,
        }

        # Append to JSONL file (one record per line)
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info(
            "Feedback stored | id=%s | image=%s | decision=%s",
            feedback_id[:8], image_id, user_decision,
        )
        return feedback_id

    def load_all(self) -> list:
        """Load all feedback records from the JSONL file."""
        if not self.feedback_file.exists():
            return []
        records = []
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        log.warning("Skipping malformed feedback line: %s", e)
        return records

    def summary(self) -> dict:
        """Return summary statistics of the feedback dataset."""
        records = self.load_all()
        total = len(records)
        confirmed  = sum(1 for r in records if r.get("user_decision") == "confirmed")
        rejected   = sum(1 for r in records if r.get("user_decision") == "rejected")
        rov        = sum(1 for r in records if r.get("user_decision") == "rov_inspection")

        ready = total >= self.retraining_threshold
        return {
            "total_feedback_records": total,
            "confirmed":              confirmed,
            "rejected":               rejected,
            "rov_inspection":         rov,
            "retraining_threshold":   self.retraining_threshold,
            "ready_for_retraining":   ready,
            "retraining_message": (
                f"✓ Ready: {total} verified samples — trigger retraining when convenient."
                if ready else
                f"Not ready: {total}/{self.retraining_threshold} samples collected."
            ),
            "feedback_file":          str(self.feedback_file),
        }

    def export_verified_dataset(self, output_dir: Path) -> dict:
        """
        Export confirmed detections as a YOLO-format dataset stub.

        The exported images and labels can be added to the training dataset
        and used to retrain the model in a controlled fashion.

        Returns: dict with export statistics.
        """
        import shutil

        records  = self.load_all()
        output_dir = Path(output_dir)
        img_out  = output_dir / "images"
        lbl_out  = output_dir / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped  = 0

        for rec in records:
            if rec.get("user_decision") != "confirmed":
                continue

            img_path = rec.get("image_path")
            detection = rec.get("detection", {})

            if not img_path or not Path(img_path).exists():
                skipped += 1
                continue

            # Copy image
            src = Path(img_path)
            dst_img = img_out / src.name
            shutil.copy2(src, dst_img)

            # Write YOLO label
            bbox  = detection.get("bbox", [])
            cls_id = detection.get("class_id", 0)
            iw    = detection.get("image_width", 1)
            ih    = detection.get("image_height", 1)

            if len(bbox) == 4 and iw > 0 and ih > 0:
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                lbl_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

                lbl_file = lbl_out / (src.stem + ".txt")
                with open(lbl_file, "a", encoding="utf-8") as f:
                    f.write(lbl_line)

            exported += 1

        log.info("Exported %d verified samples to %s", exported, output_dir)
        return {
            "exported":  exported,
            "skipped":   skipped,
            "output_dir": str(output_dir),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description="SONAR-GUARD Active Learning Feedback")
    parser.add_argument("--status",  action="store_true", help="Show feedback dataset status")
    parser.add_argument("--export",  action="store_true", help="Export verified dataset")
    parser.add_argument("--out",     default="data/active_learning_export",
                        help="Export output directory")
    parser.add_argument("--file",    default=str(FEEDBACK_FILE_DEFAULT))
    args = parser.parse_args()

    store = FeedbackStore(feedback_file=Path(args.file))

    if args.status:
        summary = store.summary()
        print(json.dumps(summary, indent=2))

    if args.export:
        result = store.export_verified_dataset(Path(args.out))
        print(json.dumps(result, indent=2))

    if not args.status and not args.export:
        parser.print_help()


if __name__ == "__main__":
    main()

