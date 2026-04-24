from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected Tensor or tuple/list of Tensors, got {type(x)}")
    return x

class DictHook:
    def __init__(self, target_dict: dict, idx: int):
        self.target_dict = target_dict
        self.idx = idx

    def __call__(self, module, inputs, output):
        self.target_dict[self.idx] = _as_tensor(output)


class FeatureTap:
    """Capture intermediate feature maps with forward hooks."""

    def __init__(self, module_list: Union[nn.ModuleList, Sequence[nn.Module]], indices: Sequence[int]):
        self.indices = list(indices)
        self.features_dict = {idx: None for idx in self.indices}
        self.handles = []
        
        for idx in self.indices:
            # Use the globally defined callable class so pickle doesn't crash
            hook = DictHook(self.features_dict, idx)
            self.handles.append(module_list[idx].register_forward_hook(hook))

    @property
    def features(self) -> List[torch.Tensor]:
        """Return captured features in the correct order."""
        return [self.features_dict[idx] for idx in self.indices if self.features_dict[idx] is not None]

    def clear(self):
        """Reset captured features."""
        for idx in self.indices:
            self.features_dict[idx] = None

    def close(self):
        """Remove hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


class ABF(nn.Module):
    """
    Attention-Based Fusion block from ReviewKD.
    """

    def __init__(self, in_channel: int, mid_channel: int, out_channel: int, fuse: bool):
        super().__init__()
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU(inplace=True)
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        out_hw: Optional[Tuple[int, int]] = None,
    ):
        x = self.conv1(x)
        if self.fuse:
            if residual is None:
                raise ValueError("residual cannot be None when fuse=True")
            if out_hw is None:
                out_hw = x.shape[-2:]
            residual = F.interpolate(residual, size=out_hw, mode="nearest")
            z = torch.cat([x, residual], dim=1)
            att = self.att_conv(z)
            x = x * att[:, :1] + residual * att[:, 1:]
        out = self.conv2(self.relu(x))
        return out, x


class ReviewKDAdapter(nn.Module):
    """
    ReviewKD adapter over a list of multi-scale student features.
    """

    def __init__(self, student_channels: Sequence[int], teacher_channels: Sequence[int], mid_channel: int = 256):
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")
        self.num_stages = len(student_channels)
        abfs: List[ABF] = []
        for i, (sc, tc) in enumerate(zip(reversed(student_channels), reversed(teacher_channels))):
            abfs.append(ABF(sc, mid_channel, tc, fuse=i > 0))
        self.abfs = nn.ModuleList(abfs)

    def forward(self, student_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feats = list(student_feats)
        feats = [_as_tensor(x) for x in feats]
        results: List[torch.Tensor] = []
        residual = None
        for i, (abf, feat) in enumerate(zip(self.abfs, reversed(feats))):
            out_hw = feat.shape[-2:]
            if i == 0:
                out, residual = abf(feat, residual=None, out_hw=out_hw)
            else:
                out, residual = abf(feat, residual=residual, out_hw=out_hw)
            results.append(out)
        return list(reversed(results))


class HCLLoss(nn.Module):
    """
    Hierarchical Context Loss from ReviewKD.
    """

    def __init__(self, pool_sizes: Sequence[int] = (8,4, 2, 1)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

    def forward(self, reviewed_feats: Sequence[torch.Tensor], teacher_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(reviewed_feats) != len(teacher_feats):
            raise ValueError("reviewed_feats and teacher_feats must have the same length")
        loss = reviewed_feats[0].new_tensor(0.0)
        for fs, ft in zip(reviewed_feats, teacher_feats):
            fs = _as_tensor(fs)
            ft = _as_tensor(ft)
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="nearest")
            stage_loss = F.mse_loss(fs, ft, reduction="mean")
            norm = 1.0
            weight =1.0
            h, w = ft.shape[-2:]
            for size in self.pool_sizes:
                if size >= h or size >= w:
                    continue
                weight *= 0.8
                pooled_s = F.adaptive_avg_pool2d(fs, (size, size))
                pooled_t = F.adaptive_avg_pool2d(ft, (size, size))
                stage_loss = stage_loss + weight * F.mse_loss(pooled_s, pooled_t, reduction="mean")
                norm += weight
            loss = loss + stage_loss / norm
        return loss


@torch.no_grad()
def infer_hook_channels(model: nn.Module, indices: Sequence[int], img_size: int = 1024) -> List[int]:
    taps = FeatureTap(model.model, indices)
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype  # auto-match model dtype (FP16 or FP32)
        x = torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype)
        _ = model(x)
        channels = [feat.shape[1] for feat in taps.features]
    finally:
        if was_training:
            model.train()
        taps.close()
    return channels
