"""
Step 1: Pre-compute and cache teacher features to disk.
Run this ONCE before training. Teacher model is loaded on GPU briefly,
features are computed per image, saved to disk, then teacher is freed.
"""
from __future__ import annotations

import os
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from reviewkd_modules import FeatureTap

# --------------------------------------------------
# CONFIG — EDIT THESE
# --------------------------------------------------
TEACHER_WEIGHTS = "/work/biraj/neuromorphic/runs/detect/26Xtrain/weights/best.pt"
DATA_YAML = "/work/biraj/neuromorphic/trainval_dataset.yaml"
FEATURES_DIR = "/work/biraj/neuromorphic/teacher_features"
IMG_SIZE = 1024
HOOK_INDICES = [16, 19, 22]
DEVICE = "cuda:0"

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print(f"Loading teacher from {TEACHER_WEIGHTS}...")
    teacher_yolo = YOLO(TEACHER_WEIGHTS)
    teacher = teacher_yolo.model.to(DEVICE).half().eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Register hooks to capture teacher features
    tap = FeatureTap(teacher.model, HOOK_INDICES)

    # Build dataloader using Ultralytics' internal data builder
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = DATA_YAML
    cfg.imgsz = IMG_SIZE
    cfg.task = "detect"

    # Load dataset info
    from ultralytics.data.utils import check_det_dataset
    data_info = check_det_dataset(DATA_YAML)
    train_path = data_info["train"]

    dataset = build_yolo_dataset(
        cfg, img_path=train_path, batch=1, data=data_info, mode="train", rect=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )

    print(f"Processing {len(dataset)} images...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            tap.clear()

            # Get image path for unique filename
            img_file = Path(batch["im_file"][0]).stem

            # Forward pass
            imgs = batch["img"].to(DEVICE).half() / 255.0
            _ = teacher(imgs)

            # Save features to disk as FP16 (small file size)
            features = [f.cpu().half() for f in tap.features]
            torch.save(features, os.path.join(FEATURES_DIR, f"{img_file}.pt"))

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(dataset)}] processed")

    tap.close()
    print(f"\nDone. Teacher features saved to {FEATURES_DIR}")
    print(f"Total files: {len(os.listdir(FEATURES_DIR))}")


if __name__ == "__main__":
    main()
