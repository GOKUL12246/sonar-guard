# SONAR-GUARD
## AI-Powered Automated Underwater Marine Debris and Anomaly Detection System
### Using Side-Scan Sonar Imagery

**Team:** ZENVIS | M. Kumarasamy College of Engineering
**Event:** Smart India Hackathon 2026 — Problem Statement 26057

---

## 1. Project Overview

SONAR-GUARD is a real, executable AI system that processes Side-Scan Sonar (SSS) imagery to detect, analyse, prioritise, and report underwater marine debris and anomalies.

The system pipeline:

```
RAW SONAR → PREPROCESSING → YOLO DETECTION → SEGMENTATION
→ SONAR EVIDENCE ANALYSIS → UNCERTAINTY → EVIDENCE FUSION
→ ARTIFICIALITY SCORE → MARINE RISK SCORE → HUMAN VERIFICATION
→ GEOLOCATION → ACTIONABLE REPORT → ACTIVE LEARNING
```

---

## 2. Problem Statement

Side-scan sonar surveys produce large volumes of imagery that human analysts must manually review. SONAR-GUARD automates:
- Object detection and localisation
- Natural vs. artificial object classification using sonar-specific evidence
- Acoustic shadow analysis
- Risk prioritisation for field response
- Human-in-the-loop verification with active learning

---

## 3. System Architecture

| Phase | Module | Status |
|-------|--------|--------|
| Data Ingestion | `src/ingestion/loader.py` | ✅ Implemented |
| Preprocessing | `src/preprocessing/sonar_preprocessor.py` | ✅ Implemented |
| Self-Supervised Pretraining | `src/self_supervised/pretraining.py` | ✅ Experimental (SimCLR) |
| YOLOv8 Detection | `src/detection/yolo_detector.py` | ✅ Implemented |
| Segmentation | `src/segmentation/segmenter.py` | ✅ Contour fallback; SAM optional |
| Natural/Artificial Analysis | `src/sonar_analysis/natural_artificial.py` | ✅ Implemented |
| Acoustic Shadow Analysis | `src/sonar_analysis/acoustic_shadow.py` | ✅ Implemented |
| Uncertainty | `src/uncertainty/estimator.py` | ✅ Implemented |
| Evidence Fusion | `src/fusion/evidence_fusion.py` | ✅ Weighted linear |
| Artificiality Score | `src/scoring/artificiality.py` | ✅ Implemented |
| Marine Risk Score | `src/scoring/marine_risk.py` | ✅ Implemented |
| Multi-Pass Verification | `src/verification/multi_pass.py` | ✅ Implemented |
| Geolocation | `src/geolocation/geotag.py` | ✅ Implemented (metadata-based) |
| Active Learning | `src/active_learning/feedback.py` | ✅ Implemented |
| Reports | `src/reporting/report_generator.py` | ✅ JSON + CSV |
| API backend | `backend/main.py` + `backend/pipeline.py` | ✅ FastAPI (contacts, geocoding proxy, analysis, verification) |
| Frontend | `frontend/` (React + React-Leaflet + OSM + Nominatim) | ✅ Vite dev server / static build |
| Training | `src/training/train_yolo.py` | ✅ Implemented |
| Evaluation | `src/training/evaluate.py` | ✅ Implemented |
| Dataset Validator | `src/training/dataset_validator.py` | ✅ Implemented |

---

## 4. Installation

### Requirements
- Python 3.11.9 (tested)
- Windows 10/11 (Linux/macOS compatible)
- No GPU required (CPU mode supported)

### Setup

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

### 5.1 Obtain a Side-Scan Sonar dataset

Legitimate SSS datasets include:
- **Ghost Pot Detection** — https://github.com/pbeauclair/ghost-pot-detection
- **SubPipe** — https://subpipe-dataset.github.io
- **AI4Shipwrecks** — contact dataset authors
- Any YOLO-format annotated SSS dataset

Place raw images in `data/raw/`.

### 5.2 Validate your dataset

```powershell
python -m src.training.dataset_validator data\raw debris shipwreck net
```

This produces `data/raw/dataset_report.json` with real statistics.

### 5.3 Visualise annotations (verify labels look correct)

Use `visualize_yolo_annotations` to render real labels over an image before training:

```powershell
python -c "
from pathlib import Path
from src.training.dataset_validator import visualize_yolo_annotations
visualize_yolo_annotations(
    Path('data/raw/images/example.png'),
    Path('data/raw/labels/example.txt'),
    ['marine_debris'],
    Path('outputs/visualizations/example.png'),
)
"
```

Inspect the generated image. If boxes look wrong, fix labels before training.

### 5.4 Convert annotations

COCO bounding boxes can be converted to YOLO labels with
`convert_coco_to_yolo`. Flat JSONL records must contain `image_id`, `bbox`
(`x, y, width, height` in pixels), and either `class_id` or `class`; provide
image dimensions to `convert_jsonl_to_yolo`. Both converters write normalized
YOLO labels and do not invent missing annotations.

### 5.5 Split dataset

```powershell
python -c "
from pathlib import Path
from src.training.dataset_validator import DatasetSplitter
splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
images = sorted(Path('data/raw/images').glob('*.png'))
labels = [Path('data/raw/labels') / image.with_suffix('.txt').name for image in images]
splits = splitter.split(images, labels, output_dir=Path('data'), copy_files=True)
print('Split:', {k: len(v) for k, v in splits.items()})
"
```

Splitting is reproducible with the configured seed. Numbered frames sharing a
survey prefix, such as `survey_001` and `survey_002`, remain in the same split
to reduce sequence leakage between training and evaluation.

---

## 6. Training

### 6.1 Generate dataset YAML

```powershell
python -c "
from pathlib import Path
from src.training.train_yolo import generate_dataset_yaml
generate_dataset_yaml(
    train_dir   = Path('data/train/images'),
    val_dir     = Path('data/val/images'),
    test_dir    = Path('data/test/images'),
    class_names = ['marine_debris', 'shipwreck', 'ghost_gear'],
    output_path = Path('data/sonar.yaml'),
)
"
```

### 6.2 Train YOLOv8

```powershell
# CPU training (nano model — fastest on CPU)
python -m src.training.train_yolo --dataset data\sonar.yaml --model-size n --epochs 50 --batch 4

# With GPU (if available)
python -m src.training.train_yolo --dataset data\sonar.yaml --model-size s --epochs 100 --batch 16 --device cuda
```

Trained model saved to: `outputs/yolo_runs/sonarguard_yolo/weights/best.pt`

### 6.3 Portable CUDA GPU Training

The current CPU-only machine must not be used for long multi-epoch training.
Run the following setup on a CUDA-enabled NVIDIA Windows or Linux machine.

#### Required environment

- Python 3.10 or 3.11
- NVIDIA driver compatible with the selected CUDA-enabled PyTorch wheel
- A CUDA-capable NVIDIA GPU with enough VRAM for the selected batch size
- The repository copied with `data/yolo/` present

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

On Linux, activate with `source .venv/bin/activate` instead. Install the
CUDA-enabled PyTorch build using the command from the official PyTorch
selector for the target driver. For example, a CUDA 12.4 wheel is:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install ultralytics==8.4.11
python -m pip install -r requirements.txt --no-deps
```

Do not replace the CUDA-enabled PyTorch installation with the CPU-only torch
pin from `requirements.txt`.

#### Dataset placement and validation

Keep the validated dataset at `data/yolo/` and do not modify the original
JSONL annotations. The training and evaluation scripts rebase the prepared
YAML at runtime if its stored absolute path came from another machine.

```powershell
python -m src.training.yolo_dataset_preparer --seed 42
```

This must report 5,721 train images, 555 validation images, 398 test images,
class `Crab-Pot`, and zero invalid labels before training.

#### CUDA preflight

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'UNAVAILABLE'); print('vram_gb:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2) if torch.cuda.is_available() else 'UNAVAILABLE')"
```

Proceed only when CUDA is `True`, the expected GPU is shown, and the dataset
validation passes.

#### GPU training command

The reproducible configuration is stored in
`configs/yolo_gpu_baseline.yaml`:

```powershell
python -m src.training.train_yolo --dataset data\yolo\data.yaml --model-size n --epochs 50 --img-size 640 --batch 16 --device cuda --workers 4 --seed 42 --output-dir outputs\yolo_runs_gpu_baseline --log-level INFO
```

If GPU VRAM is insufficient, reduce only `--batch` to `8`, `4`, or `2`.
Keep the model, dataset, seed, and test set unchanged.

#### GPU validation and test evaluation

```powershell
python -m src.training.evaluate --model runs\detect\outputs\yolo_runs_gpu_baseline\sonarguard_yolo\weights\best.pt --dataset data\yolo\data.yaml --split val --img-size 640 --batch 16 --device cuda --output outputs\evaluation\gpu_baseline

python -m src.training.evaluate --model runs\detect\outputs\yolo_runs_gpu_baseline\sonarguard_yolo\weights\best.pt --dataset data\yolo\data.yaml --split test --img-size 640 --batch 16 --device cuda --output outputs\evaluation\gpu_baseline
```

Expected artifacts include `best.pt`, `last.pt`, `results.csv`, `results.png`,
confusion matrices, precision-recall curves, validation plots, and evaluation
reports under the run and evaluation output directories. Copy the verified
best checkpoint to `models/yolo/` only after the run completes.

---

## 7. Evaluation

```powershell
python -m src.training.evaluate --model outputs\yolo_runs\sonarguard_yolo\weights\best.pt --dataset data\sonar.yaml --split test
```

Report saved to: `outputs/evaluation/eval_test/evaluation_report.json`

**NOTE:** No metrics are reported until the model has actually been evaluated.

---

## 8. Running the App (React frontend + FastAPI backend)

```powershell
# 1. Backend (detection pipeline, contacts, Nominatim place names, verification)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Frontend (React + React-Leaflet + OpenStreetMap, no API key needed)
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

Tabs: 📊 Dashboard · 🔬 Analysis Studio (image upload → live YOLO detection) ·
🗺️ Marine GIS Map (real places via Nominatim) · 👤 Review Queue · 📄 Mission Reports.

**To use a trained model:** Place `best.pt` in `models/yolo/` — the backend auto-loads
the first `*.pt` found there.

---

## 9. Self-Supervised Pretraining (Experimental)

```powershell
python -m src.self_supervised.pretraining --data data\raw --epochs 30 --batch 16
```

**IMPORTANT:** This is experimental. The SimCLR encoder is NOT inserted into the YOLOv8 backbone. It produces standalone embeddings for clustering/anomaly detection research.

---

## 10. Active Learning

```powershell
# Check feedback dataset status
python -m src.active_learning.feedback --status

# Export verified samples for retraining
python -m src.active_learning.feedback --export --out data\active_learning_export
```

**NOTE:** The model is NOT automatically retrained. Retraining is a deliberate, controlled action.

---

## 11. Running Tests

```powershell
python -m pytest tests/ -v
```

---

## 12. Folder Structure

```
SONAR-GUARD/
├── backend/main.py                 # FastAPI backend (contacts, geocoding, analysis, verify)
├── backend/pipeline.py             # Streamlit-free detection pipeline
├── frontend/                       # React + React-Leaflet + OSM + Nominatim app
├── config.py                       # Central configuration
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── ingestion/loader.py         # Image + metadata loading
│   ├── preprocessing/              # Sonar preprocessing pipeline
│   ├── training/                   # Dataset validation, YOLO training, evaluation
│   ├── self_supervised/            # SimCLR pretraining [EXPERIMENTAL]
│   ├── detection/                  # YOLOv8 inference wrapper
│   ├── segmentation/               # Pluggable segmenter (contours / SAM)
│   ├── sonar_analysis/             # Shape/texture/shadow analysis
│   ├── uncertainty/                # Multi-signal uncertainty estimation
│   ├── fusion/                     # Weighted evidence fusion
│   ├── scoring/                    # Artificiality + Marine Risk scores
│   ├── verification/               # Multi-pass comparison
│   ├── geolocation/                # Navigation metadata → GPS projection
│   ├── active_learning/            # Human feedback storage
│   └── reporting/                  # JSON + CSV report generation
├── models/
│   ├── yolo/                       # Trained YOLO weights (.pt)
│   └── sam/                        # SAM checkpoint (optional)
├── data/
│   ├── raw/                        # Raw sonar images (unprocessed)
│   ├── processed/
│   ├── train/ val/ test/           # Split dataset (images + labels)
│   └── verified/                   # Active learning feedback (feedback.jsonl)
├── outputs/
│   ├── detections/
│   ├── masks/
│   ├── reports/                    # Generated JSON/CSV reports
│   └── visualizations/
└── tests/
    ├── test_preprocessing.py
    ├── test_detection.py
    ├── test_scoring.py
    └── test_reporting.py
```

---

## 13. Limitations (Technical Honesty)

| Limitation | Status |
|------------|--------|
| No CUDA GPU on development machine | CPU-only training/inference. Slower but functional. |
| SAM not installed | Using contour-based segmentation as sonar-appropriate fallback. SAM installable separately. |
| Self-supervised encoder not in YOLO backbone | SimCLR encoder is a standalone module. Backbone integration requires custom architecture. |
| No calibrated Bayesian fusion | Evidence fusion is deterministic weighted linear combination. Bayesian fusion requires calibrated data. |
| Geolocation accuracy | Depends entirely on navigation metadata quality. No GPS is extracted from the image itself. |
| Dataset not included | Raw SSS datasets are large binary files. Obtain from legitimate sources listed above. |
| Accuracy metrics not reported yet | Will be reported after training on real SSS dataset and running evaluation. |

---

## 14. Future Work

- [ ] Integrate SSL encoder features into YOLO backbone (requires architecture customisation)
- [ ] Calibrated Bayesian evidence fusion (requires labelled calibration dataset)
- [ ] SAM integration with GPU
- [ ] ONNX export for deployment
- [ ] Support for SDF/JSF sonar file formats
- [ ] Sequence-aware multi-pass matching using GPS track
- [ ] Web API (FastAPI) for integration with AUV mission systems

---

## 15. Technical Honesty Statement

SONAR-GUARD does NOT:
- Report accuracy/mAP until actually measured on real test data
- Generate fake GPS coordinates (returns null if unavailable)
- Claim Bayesian inference without actual Bayesian implementation
- Pretend SAM works on sonar without validation
- Fabricate detection results, confidence scores, or risk probabilities
- Claim real-time performance without actual benchmarking

All displayed results come from actual computation or are explicitly labelled **"DEMO DATA — NOT REAL INFERENCE"**.

---

*SONAR-GUARD v0.1.0 — Phase 1 (Skeleton + Core Pipeline) complete.*
*Built for SIH 2026 — PS 26057*

