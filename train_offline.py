"""
Step 2: Train student N with pre-computed teacher features.
Run this AFTER precompute_teacher_features.py has finished.
"""
from __future__ import annotations

from offline_reviewkd_trainer import OfflineReviewKDTrainer

# --------------------------------------------------
# EDIT THESE PATHS
# --------------------------------------------------
STUDENT_INIT = "/work/biraj/neuromorphic/runs/detect/26Ntrain/weights/best.pt"
#STUDENT_INIT = "/work/biraj/neuromorphic/runs/detect/26N_offline_reviewkd/weights/last.pt"
DATA_YAML = "/work/biraj/neuromorphic/trainval_dataset.yaml"
FEATURES_DIR = "/work/biraj/neuromorphic/teacher_features"

# Teacher channel sizes for YOLO26-X at layers [16, 19, 22]
# You need to check these once — run precompute script and it prints feature shapes
# Typical values for YOLO26-X: [320, 640, 640] — verify and update
TEACHER_CHANNELS = [384, 768, 768]

# --------------------------------------------------
# TRAINING CONFIG
# --------------------------------------------------
overrides = dict(
    model=STUDENT_INIT,
    data=DATA_YAML,
    epochs=200,
    imgsz=1024,
    batch=4,
    device=0,
    project="/work/biraj/neuromorphic/runs/detect",
    name="26N_offline_reviewkd",
    optimizer="AdamW",
    lr0=2e-4,
    plots=True,
    
    fliplr=0.0,
    flipud=0.0,

    # ---- custom offline KD args ----
    features_dir=FEATURES_DIR,
    teacher_channels=TEACHER_CHANNELS,
    kd_weight=5.0,
    kd_warmup_epochs=10,
    kd_hook_indices=[16, 19, 22],
)

if __name__ == "__main__":
    trainer = OfflineReviewKDTrainer(overrides=overrides)
    trainer.train()
