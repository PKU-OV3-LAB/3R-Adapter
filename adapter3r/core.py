"""3R-Adapter core implementation with checkpoint-compatible parameters."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Adapter3R(nn.Module):
    """Low-rank deformable cross-view adapter."""

    def __init__(
        self,
        embed_dim,
        reduction_dim,
        drop_rate_adapter=0,
        K_sample=4,
        region_scale=0.2,
        gating=False,
        fusion_factor=0.1,
        grid_layout="legacy",
        sampling_layout="legacy",
        **kwargs,
    ):
        super().__init__()
        if grid_layout not in {"legacy", "pixel_center"}:
            raise ValueError(f"unknown grid_layout: {grid_layout}")
        if sampling_layout not in {"legacy", "reference_major"}:
            raise ValueError(f"unknown sampling_layout: {sampling_layout}")
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.K_sample = K_sample
        self.region_scale = region_scale
        self.grid_layout = grid_layout
        self.sampling_layout = sampling_layout
        self.gating = gating
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_projection = nn.Linear(embed_dim, reduction_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(drop_rate_adapter)
        self.up_projection = nn.Linear(reduction_dim, embed_dim)
        self.gate = nn.Parameter(torch.zeros(embed_dim)) if gating else 1
        self.offset_head = nn.Linear(reduction_dim, K_sample * 2)
        self.weight_head = nn.Linear(reduction_dim, K_sample)
        self.reference_norm = nn.LayerNorm(embed_dim)
        self.reference_down_projection = nn.Linear(embed_dim, reduction_dim)
        self.fusion_factor = fusion_factor
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.xavier_uniform_(self.reference_down_projection.weight)
        nn.init.zeros_(self.reference_down_projection.bias)
        nn.init.xavier_uniform_(self.up_projection.weight, gain=0.3)
        nn.init.zeros_(self.up_projection.bias)
        if self.gating:
            nn.init.zeros_(self.gate)
        nn.init.xavier_uniform_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.xavier_uniform_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def create_grid(self, batch_size, height, width, device, dtype=None):
        """Create an ``(x, y)`` grid for the configured coordinate layout."""
        dtype = dtype or torch.float32
        if self.grid_layout == "legacy":
            ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
            xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2 / height - 1
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2 / width - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), -1).unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )

    def normalize_to_grid(self, positions, height, width):
        """Convert ``(y, x)`` indices to normalized ``(x, y)`` coordinates."""
        if min(height, width) <= 0:
            raise ValueError("position bounds must be positive")
        y, x = positions[..., 0], positions[..., 1]
        if self.grid_layout == "legacy":
            if min(height, width) <= 1:
                raise ValueError("legacy sparse coordinates require height/width > 1")
            grid_x, grid_y = 2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1
        else:
            grid_x, grid_y = 2 * (x + 0.5) / width - 1, 2 * (y + 0.5) / height - 1
        return torch.stack((grid_x, grid_y), -1)

    def forward(
        self,
        x,
        x_refs,
        img_shape,
        patch_size,
        scale=0.5,
        **kwargs,
    ):
        if x.ndim != 3 or x_refs.ndim != 4:
            raise ValueError("inputs must have shapes (B,Q,C) and (B,M,HW,C)")
        batch_size, references, token_count, channels = x_refs.shape
        height = img_shape[0] // patch_size
        width = img_shape[1] // patch_size
        if (
            height * width != token_count
            or x.shape[0] != batch_size
            or x.shape[2] != channels
        ):
            raise ValueError("input dimensions do not match the patch grid")
        queries = x.shape[1]
        self_out = self.dropout(
            self.activate(self.down_projection(self.layer_norm(x)))
        )
        offset = (
            torch.tanh(
                self.offset_head(self_out).view(
                    batch_size, queries, 1, self.K_sample, 2
                )
            )
            * self.region_scale
        )
        raw_weight = F.softmax(
            self.weight_head(self_out).view(
                batch_size, queries, 1, self.K_sample
            ),
            -1,
        )
        sparse = kwargs.get("sparse_pos")
        bound = kwargs.get("pos_bound")
        if sparse is not None or bound is not None:
            if sparse is None or bound is None:
                raise ValueError("sparse_pos and pos_bound must be provided together")
            if sparse.shape != (batch_size, queries, 2) or len(bound) != 2:
                raise ValueError("sparse_pos must have shape (B,Q,2), with two bounds")
            base = self.normalize_to_grid(sparse.to(x.device), *bound)
            base = base.unsqueeze(2).unsqueeze(3)
        elif queries == token_count:
            # Match the FP32 grid used by the completed training runs, including
            # under BF16 autocast. Rounding the base to token dtype changes sampling.
            base = self.create_grid(
                batch_size, height, width, x.device
            ).reshape(batch_size, queries, 1, 1, 2)
        else:
            raise ValueError("sparse queries require sparse_pos and pos_bound")
        locations = (base + offset).clamp(-1, 1)
        expanded_grid = locations.unsqueeze(1).expand(
            -1, references, -1, -1, -1, -1
        )
        if self.sampling_layout == "reference_major":
            expanded_grid = expanded_grid.permute(0, 1, 4, 2, 3, 5)
        grid = expanded_grid.reshape(
            batch_size * references * self.K_sample, queries, 1, 2
        )
        refs = self.reference_down_projection(self.reference_norm(x_refs))
        refs = refs.view(
            batch_size, references, height, width, self.reduction_dim
        ).permute(0, 1, 4, 2, 3).unsqueeze(2)
        refs = refs.expand(-1, -1, self.K_sample, -1, -1, -1).reshape(
            batch_size * references * self.K_sample,
            self.reduction_dim,
            height,
            width,
        )
        sampled = F.grid_sample(
            refs,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch_size,
            references,
            self.K_sample,
            self.reduction_dim,
            queries,
            1,
        )
        weight = raw_weight.permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(3)
        cross = (
            (sampled * weight)
            .sum(2)
            .sum(1)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, queries, self.reduction_dim)
        )
        fused = (1 - self.fusion_factor) * self_out + self.fusion_factor * cross
        out = self.up_projection(fused * scale) * self.gate
        return out
