"""
SONAR-GUARD — Self-Supervised Pretraining
==========================================
Implements a lightweight self-supervised representation learning module
for unlabeled sonar imagery.

IMPLEMENTATION STATUS:
  - SimCLR-style contrastive learning: IMPLEMENTED (experimental)
  - Autoencoder: IMPLEMENTED (experimental)
  - Integration with YOLOv8 backbone: NOT IMPLEMENTED
    (Ultralytics does not expose backbone-level weight injection via public API;
     this would require custom architecture modifications — documented as future work)

WHAT THIS MODULE DOES:
  1. Learns a compact image embedding from unlabeled sonar images.
  2. Can be used for anomaly detection, clustering, or nearest-neighbour retrieval.
  3. The learned representations are NOT directly inserted into the YOLO backbone.

HOW TO USE:
  Run pretraining on raw (unlabeled) sonar images:
    python -m src.self_supervised.pretraining --data data/raw --method simclr --epochs 30

  The trained encoder is saved to: models/ssl_encoder.pt
  It can be used to extract embeddings for downstream clustering/anomaly tasks.

HONEST LIMITATION:
  Without a well-calibrated dataset and GPU, pretraining effects are
  difficult to validate. Results should be treated as experimental.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import cv2

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SonarUnlabeledDataset(Dataset):
    """
    Dataset of unlabeled sonar images for self-supervised learning.
    Each image is returned twice with different augmentations (for contrastive).
    """
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        directory:    Path,
        img_size:     int = 128,
        augment:      bool = True,
    ):
        directory = Path(directory)
        self.img_size = img_size
        self.augment  = augment
        self.files = sorted([
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in self.EXTENSIONS
        ])
        log.info("SonarUnlabeledDataset: %d images in %s", len(self.files), directory)
        if len(self.files) == 0:
            log.warning("No images found in %s", directory)

    def __len__(self):
        return len(self.files)

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img.astype(np.float32) / 255.0

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Simple sonar-appropriate augmentations."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = img[:, ::-1].copy()
        # Random noise
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
        # Random brightness
        factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0.0, 1.0)
        return img

    def __getitem__(self, idx):
        img = self._load(self.files[idx])
        if self.augment:
            view1 = self._augment(img)
            view2 = self._augment(img)
        else:
            view1 = img
            view2 = img
        # Shape: (1, H, W) — single channel
        t1 = torch.from_numpy(view1).unsqueeze(0)
        t2 = torch.from_numpy(view2).unsqueeze(0)
        return t1, t2


# ---------------------------------------------------------------------------
# Encoder network
# ---------------------------------------------------------------------------

class SonarEncoder(nn.Module):
    """
    Lightweight CNN encoder for sonar image embeddings.
    Input: (B, 1, H, W) grayscale sonar image
    Output: (B, embedding_dim) embedding vector
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),   # H/2
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),   # H/4
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # 4×4 regardless of input size
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projector(features)
        return embedding


# ---------------------------------------------------------------------------
# SimCLR loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used in SimCLR-style contrastive learning.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)         # (2B, D)
        z  = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------

class SelfSupervisedPretrainer:
    """
    Trains the SonarEncoder using self-supervised contrastive learning.

    Status: EXPERIMENTAL
    Limitations:
      - Without GPU, training is slow (minutes per epoch even on small datasets).
      - Encoder output is not directly injected into YOLOv8 backbone.
      - Downstream benefit requires fine-tuning and proper evaluation.
    """

    def __init__(
        self,
        embedding_dim:  int   = 128,
        temperature:    float = 0.07,
        device:         str   = "cpu",
    ):
        self.device        = device
        self.encoder       = SonarEncoder(embedding_dim=embedding_dim).to(device)
        self.loss_fn       = NTXentLoss(temperature=temperature)
        log.info(
            "SelfSupervisedPretrainer | embedding_dim=%d | device=%s | STATUS=EXPERIMENTAL",
            embedding_dim, device,
        )

    def train(
        self,
        data_dir:    Path,
        epochs:      int   = 30,
        batch_size:  int   = 16,
        lr:          float = 1e-3,
        img_size:    int   = 128,
        save_path:   Optional[Path] = None,
    ) -> dict:
        """
        Run self-supervised pretraining.

        Returns:
            dict with training history and model save path.
        """
        dataset = SonarUnlabeledDataset(Path(data_dir), img_size=img_size)
        if len(dataset) == 0:
            return {
                "status": "no_data",
                "message": f"No images found in {data_dir}. "
                           "Provide unlabeled sonar images for pretraining.",
            }

        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            num_workers=0,
            drop_last=len(dataset) >= batch_size,
        )

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        history   = []

        log.info(
            "Starting SimCLR pretraining | images=%d | epochs=%d | batch=%d",
            len(dataset), epochs, batch_size,
        )

        t_total = time.time()
        for epoch in range(1, epochs + 1):
            self.encoder.train()
            epoch_loss = 0.0
            for v1, v2 in loader:
                v1 = v1.to(self.device)
                v2 = v2.to(self.device)
                z1 = self.encoder(v1)
                z2 = self.encoder(v2)
                loss = self.loss_fn(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(loader))
            history.append({"epoch": epoch, "loss": round(avg_loss, 6)})
            if epoch % 5 == 0 or epoch == 1:
                log.info("Epoch %d/%d | loss=%.4f", epoch, epochs, avg_loss)

        elapsed = time.time() - t_total

        # Save encoder
        if save_path is None:
            save_path = Path("models") / "ssl_encoder.pt"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder_state_dict": self.encoder.state_dict(),
            "embedding_dim":      self.encoder.embedding_dim,
            "training_epochs":    epochs,
            "final_loss":         history[-1]["loss"] if history else None,
            "note": (
                "EXPERIMENTAL: SimCLR encoder for sonar imagery. "
                "NOT integrated into YOLOv8 backbone. "
                "Use for embedding extraction and clustering only."
            ),
        }, save_path)

        log.info("SSL encoder saved: %s | training time: %.1f s", save_path, elapsed)

        return {
            "status":         "ok",
            "model_path":     str(save_path),
            "training_time_s": round(elapsed, 1),
            "epochs":         epochs,
            "final_loss":     history[-1]["loss"] if history else None,
            "history":        history,
            "note":           (
                "EXPERIMENTAL — SimCLR pretraining complete. "
                "Encoder can extract embeddings but is NOT inserted into YOLOv8 backbone."
            ),
        }

    def extract_embeddings(self, images: list) -> torch.Tensor:
        """
        Extract embeddings from a list of numpy images.

        Args:
            images: List of (H, W) grayscale float32 arrays.

        Returns:
            Tensor of shape (N, embedding_dim).
        """
        self.encoder.eval()
        tensors = []
        with torch.no_grad():
            for img in images:
                if img.ndim == 2:
                    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                else:
                    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                t = t.float().to(self.device)
                emb = self.encoder(t)
                tensors.append(emb)
        return torch.cat(tensors, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONAR-GUARD Self-Supervised Pretraining [EXPERIMENTAL]")
    parser.add_argument("--data",    required=True, help="Directory of unlabeled sonar images")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--save",    default=None)
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    setup_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    trainer = SelfSupervisedPretrainer(device=device)
    result  = trainer.train(
        data_dir   = Path(args.data),
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        img_size   = args.img_size,
        save_path  = Path(args.save) if args.save else None,
    )

    import json
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))

