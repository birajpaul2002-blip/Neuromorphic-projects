"""
Offline Review KD trainer. uses pre-computed teacher features from disk.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer

from reviewkd_modules import FeatureTap, HCLLoss, ReviewKDAdapter, infer_hook_channels


class OfflineReviewKDTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = deepcopy(overrides) if overrides is not None else {}

        # Pop custom KD args
        self.features_dir = overrides.pop("features_dir", None)
        self.teacher_channels = overrides.pop("teacher_channels", None)
        self.kd_weight = float(overrides.pop("kd_weight", 1.0))
        self.kd_warmup_epochs = int(overrides.pop("kd_warmup_epochs", 10))
        self.kd_hook_indices = overrides.pop("kd_hook_indices", None)

        super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)

        self.student_tap: Optional[FeatureTap] = None
        self.review_adapter: Optional[ReviewKDAdapter] = None
        self.hcl_loss: Optional[HCLLoss] = None
        self._orig_loss_fn = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        if self.features_dir is None:
            raise ValueError("features_dir is required — run precompute_teacher_features.py first")

        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Teacher features directory not found: {self.features_dir}")

        # Auto-infer student hook indices
        if self.kd_hook_indices is None:
            if hasattr(model.model[-1], "f"):
                self.kd_hook_indices = list(model.model[-1].f)
            else:
                raise ValueError("Set kd_hook_indices manually.")

        # Register student taps only
        self.student_tap = FeatureTap(model.model, self.kd_hook_indices)

        # Infer student channels from a dummy forward
        student_channels = infer_hook_channels(model, self.kd_hook_indices, img_size=self.args.imgsz)

        # Teacher channels must be provided
        if self.teacher_channels is None:
            raise ValueError("teacher_channels must be provided, e.g. [256, 512, 1024]")

        self.review_adapter = ReviewKDAdapter(
            student_channels=student_channels,
            teacher_channels=self.teacher_channels,
            mid_channel=256,
        ).to(self.device).float()

        self.hcl_loss = HCLLoss().to(self.device).float()

        self._patch_model_loss(model)
        return model

    def _patch_model_loss(self, model):
        self._orig_loss_fn = model.loss
        trainer = self

        def reviewkd_loss(batch, preds=None):
            trainer.student_tap.clear()
            base_loss, loss_items = trainer._orig_loss_fn(batch, preds)

            if preds is not None:
                return base_loss, loss_items

            if trainer.kd_warmup_epochs > 0:
                warmup_scale = min(1.0, max(0.0, float(trainer.epoch + 1) / float(trainer.kd_warmup_epochs)))
            else:
                warmup_scale=1.0

            kd_scale = trainer.kd_weight * warmup_scale

            kd_loss = base_loss.new_tensor(0.0)

            if kd_scale > 0:
                teacher_feats = []
                try:
                    for im_file in batch["im_file"]:
                        stem = Path(im_file).stem
                        feat_path = os.path.join(trainer.features_dir, f"{stem}.pt")
                        feats = torch.load(feat_path, map_location=trainer.device, weights_only=False)
                        
                        teacher_feats.append([f.float().detach() for f in feats])
                except FileNotFoundError:
                    return base_loss, loss_items

                stacked_teacher = []
                for scale_idx in range(len(teacher_feats[0])):
                    stacked = torch.cat([tf[scale_idx] for tf in teacher_feats], dim=0)
                    stacked_teacher.append(stacked)

                student_feats = [f.float() for f in trainer.student_tap.features]

                if len(student_feats) == 0 or len(stacked_teacher) == 0:
                    return base_loss, loss_items

                reviewed_student_feats = trainer.review_adapter(student_feats)
                kd_loss = trainer.hcl_loss(reviewed_student_feats, stacked_teacher)

            total_loss = base_loss + kd_scale * kd_loss
            return total_loss, loss_items

        model.loss = reviewkd_loss

        try:
            if hasattr(model, "loss_names"):
                model.loss_names = tuple(list(model.loss_names) + ["kd_loss"])
        except Exception:
            pass

    def save_model(self):
        """
        """
        # 1. Store the current customized loss functions
        model_loss = getattr(self.model, "loss", None)
        ema_loss = getattr(self.ema.ema, "loss", None) if hasattr(self, "ema") and self.ema else None

        # 2. Revert back to Ultralytics native loss function
        self.model.loss = self._orig_loss_fn
        if ema_loss:
            self.ema.ema.loss = self._orig_loss_fn

        # 3. STRIP HOOKS: Remove them from the model so pickle doesn't see them
        if self.student_tap is not None:
            self.student_tap.close()

        # 4. Save safely 
        try:
            return super().save_model()
        finally:
            # 5. Restore the ReviewKD loss hooks for the next epoch
            self.model.loss = model_loss
            if ema_loss:
                self.ema.ema.loss = ema_loss
            # 6. REATTACH HOOKS for the next epoch's forward passes
            if self.student_tap is not None:
                self.student_tap = FeatureTap(self.model.model, self.kd_hook_indices)
