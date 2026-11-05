import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Adapter(nn.Module):

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0):
        super(Adapter, self).__init__()
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.dropout = nn.Dropout(p=drop_rate_adapter)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.down_projection = nn.Linear(self.embed_dim, self.reduction_dim)
        self.activate = nn.GELU()
        self.up_projection = nn.Linear(self.reduction_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward(self, x, scale=0.5):
        x = self.layer_norm(x)
        x = self.down_projection(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.up_projection(x) * scale
        return x

class ModLN(nn.Module):

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(mod_dim, inner_dim * 2))

    def modulate(x, shift, scale):
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(mod).chunk(2, dim=-1)
        x = self.modulate(self.norm(x), shift, scale)
        return x

class DeformableAdapterHighRank(nn.Module):

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, K_sample=4, region_scale=0.2, gating=False, fusion_factor=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.K_sample = K_sample
        self.region_scale = region_scale
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_projection = nn.Linear(embed_dim, reduction_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(drop_rate_adapter)
        self.up_projection = nn.Linear(reduction_dim, embed_dim)
        self.gating = gating
        if self.gating:
            self.gate = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.gate = 1
        self.offset_head = nn.Linear(embed_dim, K_sample * 2)
        self.weight_head = nn.Linear(embed_dim, K_sample)
        self.reference_norm = nn.LayerNorm(embed_dim)
        self.fusion_factor = fusion_factor
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        if self.gating:
            nn.init.zeros_(self.gate)
        nn.init.xavier_uniform_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.xavier_uniform_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def create_grid(self, B, H, W, device):
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def forward(self, x, x_refs, img_shape, patch_size, scale=0.5, **kwargs):
        B, M, HW, C = x_refs.shape
        img_H, img_W = img_shape
        H, W = (img_H // patch_size, img_W // patch_size)
        self_view_out = self.layer_norm(x)
        self_view_out = self.down_projection(self_view_out)
        self_view_out = self.activate(self_view_out)
        self_view_out = self.dropout(self_view_out)
        self_view_out = self.up_projection(self_view_out)
        x_refs = self.reference_norm(x_refs)
        x_refs = x_refs.view(B, M, H, W, C).permute(0, 1, 4, 2, 3)
        offset = self.offset_head(x)
        weight = self.weight_head(x)
        offset = offset.view(B, H, W, self.K_sample, 2)
        offset = torch.tanh(offset) * self.region_scale
        weight = weight.view(B, H, W, self.K_sample)
        weight = F.softmax(weight, dim=-1)
        base_grid = self.create_grid(B, H, W, x.device)
        base_grid = base_grid.unsqueeze(3)
        sampling_location = base_grid + offset
        sampling_location = sampling_location.clamp(-1.0, 1.0)
        sampling_location = sampling_location.unsqueeze(1).expand(-1, M, -1, -1, -1, -1)
        sampling_location = sampling_location.reshape(B * M * self.K_sample, H, W, 2)
        x_refs = x_refs.unsqueeze(2).expand(-1, -1, self.K_sample, -1, -1, -1)
        x_refs = x_refs.reshape(B * M * self.K_sample, C, H, W)
        sampled_refs = F.grid_sample(x_refs, sampling_location, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_refs = sampled_refs.view(B, M, self.K_sample, C, H, W)
        weight = weight.permute(0, 3, 1, 2)
        weight = weight.unsqueeze(1).unsqueeze(3)
        cross_view_out = (sampled_refs * weight).sum(2).sum(1)
        cross_view_out = cross_view_out.permute(0, 2, 3, 1).reshape(B, HW, C)
        output = ((1 - self.fusion_factor) * self_view_out + self.fusion_factor * cross_view_out) * scale
        output = output * self.gate
        return output

class DeformableAdapterLowRank(nn.Module):

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, K_sample=4, region_scale=0.1, gating=False, fusion_factor=0.2, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.K_sample = K_sample
        self.region_scale = region_scale
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_projection = nn.Linear(embed_dim, reduction_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(drop_rate_adapter)
        self.up_projection = nn.Linear(reduction_dim, embed_dim)
        self.gating = gating
        if gating:
            self.gate = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.gate = 1
        self.offset_head = nn.Linear(reduction_dim, K_sample * 2)
        self.weight_head = nn.Linear(reduction_dim, K_sample)
        self.reference_norm = nn.LayerNorm(embed_dim)
        self.fusion_factor = fusion_factor
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        if self.gating:
            nn.init.zeros_(self.gate)
        nn.init.xavier_uniform_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.xavier_uniform_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def create_grid(self, B, H, W, device):
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def forward(self, x, x_refs, img_shape, patch_size, scale=0.5, **kwargs):
        B, M, HW, C = x_refs.shape
        img_H, img_W = img_shape
        H, W = (img_H // patch_size, img_W // patch_size)
        self_view_out = self.layer_norm(x)
        self_view_out = self.down_projection(self_view_out)
        self_view_out = self.activate(self_view_out)
        self_view_out = self.dropout(self_view_out)
        offset = self.offset_head(self_view_out)
        weight = self.weight_head(self_view_out)
        offset = offset.view(B, H, W, self.K_sample, 2)
        offset = torch.tanh(offset) * self.region_scale
        weight = weight.view(B, H, W, self.K_sample)
        weight = F.softmax(weight, dim=-1)
        base_grid = self.create_grid(B, H, W, x.device)
        base_grid = base_grid.unsqueeze(3)
        sampling_location = base_grid + offset
        sampling_location = sampling_location.clamp(-1.0, 1.0)
        sampling_location = sampling_location.unsqueeze(1).expand(-1, M, -1, -1, -1, -1)
        sampling_location = sampling_location.reshape(B * M * self.K_sample, H, W, 2)
        x_refs = self.reference_norm(x_refs)
        x_refs = x_refs.view(B, M, H, W, C).permute(0, 1, 4, 2, 3)
        x_refs = x_refs.unsqueeze(2).expand(-1, -1, self.K_sample, -1, -1, -1)
        x_refs = x_refs.reshape(B * M * self.K_sample, C, H, W)
        sampled_refs = F.grid_sample(x_refs, sampling_location, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_refs = sampled_refs.view(B, M, self.K_sample, C, H, W)
        weight = weight.permute(0, 3, 1, 2)
        weight = weight.unsqueeze(1).unsqueeze(3)
        cross_view_out = (sampled_refs * weight).sum(2).sum(1)
        cross_view_out = cross_view_out.permute(0, 2, 3, 1).reshape(B, HW, C)
        self_view_out = self.up_projection(self_view_out)
        output = ((1 - self.fusion_factor) * self_view_out + self.fusion_factor * cross_view_out) * scale
        output = output * self.gate
        return output

def sliding_window_mask(length: int, width: int, device: torch.device, diagonal=True):
    linear_grid = torch.arange(length, device=device)
    mask = torch.abs(linear_grid[:, None] - linear_grid[None, :]) <= width
    if not diagonal:
        mask = mask & (linear_grid[:, None] != linear_grid[None, :])
    return mask

def sliding_window_indices_no_repeat(length, width, device):
    idx = torch.arange(length, device=device)
    left = idx - width
    right = idx + width
    shift_right = torch.clamp(-left, min=0)
    shift_left = torch.clamp(right - (length - 1), min=0)
    shift = shift_right - shift_left
    left = left + shift
    right = right + shift
    offsets = torch.arange(0, 2 * width + 1, device=device)
    ref_idx = left[:, None] + offsets[None, :]
    return ref_idx
from adapter3r import Adapter3R
