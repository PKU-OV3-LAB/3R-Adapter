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

    def forward(self, x, scale = 0.5):
        x = self.layer_norm(x)
        x = self.down_projection(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.up_projection(x)*scale
        return x


class ModLN(nn.Module):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    def modulate(x, shift, scale):
        
        x = x*(1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:

        shift, scale = self.mlp(mod).chunk(2, dim=-1)  # [N, D]
        x = self.modulate(self.norm(x), shift, scale)  # [N, L, D]
        return  x


class DeformableAdapterHighRank(nn.Module):
    """
    Deformable Cross-View Adapter
    
    x: (B, H*W, C)
    
    x_refs: (B, M, H*W, C)
    """

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, K_sample=4, region_scale=0.2, gating=False, fusion_factor=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.K_sample = K_sample
        self.region_scale = region_scale

        # Adapter MLP branch
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_projection = nn.Linear(embed_dim, reduction_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(drop_rate_adapter)
        self.up_projection = nn.Linear(reduction_dim, embed_dim)

        # learnable residual gate
        self.gating = gating
        if self.gating:
            self.gate = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.gate = 1

        # Offset and Weight Heads
        self.offset_head = nn.Linear(embed_dim, K_sample * 2)
        self.weight_head = nn.Linear(embed_dim, K_sample)
        self.reference_norm = nn.LayerNorm(embed_dim)
        self.fusion_factor = fusion_factor
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.zeros_(self.up_projection.weight)
        # nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        if self.gating:
            nn.init.zeros_(self.gate)
        # important: start from identity sampling
        # nn.init.zeros_(self.offset_head.weight)
        nn.init.xavier_uniform_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        # nn.init.zeros_(self.weight_head.weight)
        nn.init.xavier_uniform_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def create_grid(self, B, H, W, device):
        """
        Create normalized base grid in [-1, 1]
        Return shape: (B, H, W, 2)
        """
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        return grid

    def forward(self, x, x_refs, img_shape, patch_size, scale=0.5, **kwargs):
        """
        Args
            x: (B, H*W, C)
            x_refs: (B, M, H*W, C)
            img_shape: (img_H, img_W)
            patch_size: int

        Returns
            out: (B, H*W, C)
        """

        B, M, HW, C = x_refs.shape
        img_H, img_W = img_shape
        H, W = img_H//patch_size, img_W//patch_size

        # Adapter MLP branch
        self_view_out = self.layer_norm(x)
        self_view_out = self.down_projection(self_view_out)
        self_view_out = self.activate(self_view_out)
        self_view_out = self.dropout(self_view_out)
        self_view_out = self.up_projection(self_view_out)

        # Reshape to feature map
        x_refs = self.reference_norm(x_refs)
        x_refs = x_refs.view(B, M, H, W, C).permute(0, 1, 4, 2, 3)  # (B,M,C,H,W)

        # Predict offset + weight from token space
        offset = self.offset_head(x)  # (B, HW, 2K)
        weight = self.weight_head(x)  # (B, HW, K)
        offset = offset.view(B, H, W, self.K_sample, 2)
        offset = torch.tanh(offset) * self.region_scale  # stabilize
        weight = weight.view(B, H, W, self.K_sample)
        weight = F.softmax(weight, dim=-1)

        # Build base grid
        base_grid = self.create_grid(B, H, W, x.device)
        base_grid = base_grid.unsqueeze(3)  # (B,H,W,1,2)

        # Sampling location
        sampling_location = base_grid + offset  # (B,H,W,K,2)
        sampling_location = sampling_location.clamp(-1.0, 1.0)

        # expand to M reference views
        sampling_location = sampling_location.unsqueeze(1).expand(-1, M, -1, -1, -1, -1)
        sampling_location = sampling_location.reshape(B * M * self.K_sample, H, W, 2)

        # Prepare reference feature maps
        x_refs = x_refs.unsqueeze(2).expand(-1, -1, self.K_sample, -1, -1, -1)
        x_refs = x_refs.reshape(B*M*self.K_sample, C, H, W)

        # Deformable sampling
        sampled_refs = F.grid_sample(x_refs, sampling_location, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_refs = sampled_refs.view(B, M, self.K_sample, C, H, W)

        # Weighted aggregation
        weight = weight.permute(0, 3, 1, 2)  # (B,K,H,W)
        weight = weight.unsqueeze(1).unsqueeze(3)  # (B,1,K,1,H,W)

        cross_view_out = (sampled_refs * weight).sum(2).sum(1)  # sum K and M
        cross_view_out = cross_view_out.permute(0, 2, 3, 1).reshape(B, HW, C)  # Back to token

        # Final residual fusion
        output = ((1-self.fusion_factor)*self_view_out + self.fusion_factor*cross_view_out) * scale
        output = output * self.gate  # gated residual

        return output


class DeformableAdapterLowRank(nn.Module):
    """
    Deformable Cross-View Adapter
    
    x: (B, H*W, C)
    
    x_refs: (B, M, H*W, C)
    """

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, K_sample=4, region_scale=0.1, gating=False, fusion_factor=0.2, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.K_sample = K_sample
        self.region_scale = region_scale

        # Adapter MLP branch
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_projection = nn.Linear(embed_dim, reduction_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(drop_rate_adapter)
        self.up_projection = nn.Linear(reduction_dim, embed_dim)

        # learnable residual gate
        self.gating = gating
        if gating:
            self.gate = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.gate = 1

        # Offset and Weight Heads
        self.offset_head = nn.Linear(reduction_dim, K_sample * 2)
        self.weight_head = nn.Linear(reduction_dim, K_sample)
        self.reference_norm = nn.LayerNorm(embed_dim)
        self.fusion_factor = fusion_factor
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.zeros_(self.up_projection.weight)
        # nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        if self.gating:
            nn.init.zeros_(self.gate)
        # important: start from identity sampling
        # nn.init.zeros_(self.offset_head.weight)
        nn.init.xavier_uniform_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        # nn.init.zeros_(self.weight_head.weight)
        nn.init.xavier_uniform_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def create_grid(self, B, H, W, device):
        """
        Create normalized base grid in [-1, 1]
        Return shape: (B, H, W, 2)
        """
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        return grid

    def forward(self, x, x_refs, img_shape, patch_size, scale=0.5, **kwargs):
        """
        Args
            x: (B, H*W, C)
            x_refs: (B, M, H*W, C)
            img_shape: (img_H, img_W)
            patch_size: int

        Returns
            out: (B, H*W, C)
        """

        B, M, HW, C = x_refs.shape
        img_H, img_W = img_shape
        H, W = img_H//patch_size, img_W//patch_size

        # adapter MLP branch
        self_view_out = self.layer_norm(x)
        self_view_out = self.down_projection(self_view_out)
        self_view_out = self.activate(self_view_out)
        self_view_out = self.dropout(self_view_out)
        

        # predict offset + weight from token space
        offset = self.offset_head(self_view_out)  # (B, HW, 2K)
        weight = self.weight_head(self_view_out)  # (B, HW, K)
        offset = offset.view(B, H, W, self.K_sample, 2)
        offset = torch.tanh(offset) * self.region_scale  # stabilize
        weight = weight.view(B, H, W, self.K_sample)
        weight = F.softmax(weight, dim=-1)

        # build base grid
        base_grid = self.create_grid(B, H, W, x.device)
        base_grid = base_grid.unsqueeze(3)  # (B,H,W,1,2)
        # sampling location
        sampling_location = base_grid + offset  # (B,H,W,K,2)
        sampling_location = sampling_location.clamp(-1.0, 1.0)
        # expand to M reference views
        sampling_location = sampling_location.unsqueeze(1).expand(-1, M, -1, -1, -1, -1)
        sampling_location = sampling_location.reshape(B * M * self.K_sample, H, W, 2)

        # prepare reference feature maps
        x_refs = self.reference_norm(x_refs)
        x_refs = x_refs.view(B, M, H, W, C).permute(0, 1, 4, 2, 3)  # (B,M,C,H,W)
        x_refs = x_refs.unsqueeze(2).expand(-1, -1, self.K_sample, -1, -1, -1)
        x_refs = x_refs.reshape(B*M*self.K_sample, C, H, W)

        # deformable sampling
        sampled_refs = F.grid_sample(x_refs, sampling_location, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_refs = sampled_refs.view(B, M, self.K_sample, C, H, W)

        # weighted aggregation
        weight = weight.permute(0, 3, 1, 2)  # (B,K,H,W)
        weight = weight.unsqueeze(1).unsqueeze(3)  # (B,1,K,1,H,W)

        cross_view_out = (sampled_refs * weight).sum(2).sum(1)  # sum K and M
        cross_view_out = cross_view_out.permute(0, 2, 3, 1).reshape(B, HW, C)  # Back to token

        # resume to high-dimensional feature
        self_view_out = self.up_projection(self_view_out)

        # final residual fusion
        output = ((1-self.fusion_factor)*self_view_out + self.fusion_factor*cross_view_out) * scale
        output = output * self.gate  # gated residual

        return output


def sliding_window_mask(length:int, width:int, device:torch.device, diagonal=True):
    '''
    Args:
        length: sequence length (B)
        width: window radius k (keep |i-j| <= k)
        device: torch device
        diagonal: whether to keep the main diagonal

    Returns:
        mask: (length, length) bool tensor
    '''
    linear_grid = torch.arange(length, device=device)
    mask = (torch.abs(linear_grid[:,None] - linear_grid[None,:]) <= width)
    if not diagonal:
        mask = mask & (linear_grid[:,None] != linear_grid[None,:])
    
    return mask


def sliding_window_indices_no_repeat(length, width, device):
    """
    Return ref_idx with shape [length, 2 * width_eff + 1].
    Includes self.
    No repeat, no out-of-bounds.
    If length is small, width is automatically reduced.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    width = min(width, (length - 1) // 2)

    idx = torch.arange(length, device=device)

    left = idx - width
    right = idx + width

    shift_right = torch.clamp(-left, min=0)
    shift_left = torch.clamp(right - (length - 1), min=0)
    shift = shift_right - shift_left

    left = left + shift

    offsets = torch.arange(0, 2 * width + 1, device=device)
    ref_idx = left[:, None] + offsets[None, :]

    assert ref_idx.min().item() >= 0
    assert ref_idx.max().item() < length

    return ref_idx.long()


class Adapter3R(nn.Module):
    """Low-rank deformable cross-view adapter."""

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, K_sample=4, region_scale=0.2, gating=False, fusion_factor=0.1, grid_layout='pixel_center', sampling_layout='reference_major', **kwargs):
        super().__init__()
        if grid_layout not in {'legacy', 'pixel_center'}:
            raise ValueError(f'unknown grid_layout: {grid_layout}')
        if sampling_layout not in {'legacy', 'reference_major'}:
            raise ValueError(f'unknown sampling_layout: {sampling_layout}')
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
        if self.grid_layout == 'legacy':
            ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
            xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2 / height - 1
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2 / width - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack((grid_x, grid_y), -1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def normalize_to_grid(self, positions, height, width):
        """Convert ``(y, x)`` indices to normalized ``(x, y)`` coordinates."""
        if min(height, width) <= 0:
            raise ValueError('position bounds must be positive')
        y, x = (positions[..., 0], positions[..., 1])
        if self.grid_layout == 'legacy':
            if min(height, width) <= 1:
                raise ValueError('legacy sparse coordinates require height/width > 1')
            grid_x, grid_y = (2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1)
        else:
            grid_x, grid_y = (2 * (x + 0.5) / width - 1, 2 * (y + 0.5) / height - 1)
        return torch.stack((grid_x, grid_y), -1)

    def forward(self, x, x_refs, img_shape, patch_size, scale=0.5, **kwargs):
        if x.ndim != 3 or x_refs.ndim != 4:
            raise ValueError('inputs must have shapes (B,Q,C) and (B,M,HW,C)')
        batch_size, references, token_count, channels = x_refs.shape
        height = img_shape[0] // patch_size
        width = img_shape[1] // patch_size
        if height * width != token_count or x.shape[0] != batch_size or x.shape[2] != channels:
            raise ValueError('input dimensions do not match the patch grid')
        queries = x.shape[1]
        self_out = self.dropout(self.activate(self.down_projection(self.layer_norm(x))))
        offset = torch.tanh(self.offset_head(self_out).view(batch_size, queries, 1, self.K_sample, 2)) * self.region_scale
        raw_weight = F.softmax(self.weight_head(self_out).view(batch_size, queries, 1, self.K_sample), -1)
        sparse = kwargs.get('sparse_pos')
        bound = kwargs.get('pos_bound')
        if sparse is not None or bound is not None:
            if sparse is None or bound is None:
                raise ValueError('sparse_pos and pos_bound must be provided together')
            if sparse.shape != (batch_size, queries, 2) or len(bound) != 2:
                raise ValueError('sparse_pos must have shape (B,Q,2), with two bounds')
            base = self.normalize_to_grid(sparse.to(x.device), *bound)
            base = base.unsqueeze(2).unsqueeze(3)
        elif queries == token_count:
            base = self.create_grid(batch_size, height, width, x.device).reshape(batch_size, queries, 1, 1, 2)
        else:
            raise ValueError('sparse queries require sparse_pos and pos_bound')
        locations = (base + offset).clamp(-1, 1)
        expanded_grid = locations.unsqueeze(1).expand(-1, references, -1, -1, -1, -1)
        if self.sampling_layout == 'reference_major':
            expanded_grid = expanded_grid.permute(0, 1, 4, 2, 3, 5)
        grid = expanded_grid.reshape(batch_size * references * self.K_sample, queries, 1, 2)
        refs = self.reference_down_projection(self.reference_norm(x_refs))
        refs = refs.view(batch_size, references, height, width, self.reduction_dim).permute(0, 1, 4, 2, 3).unsqueeze(2)
        refs = refs.expand(-1, -1, self.K_sample, -1, -1, -1).reshape(batch_size * references * self.K_sample, self.reduction_dim, height, width)
        sampled = F.grid_sample(refs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled = sampled.view(batch_size, references, self.K_sample, self.reduction_dim, queries, 1)
        weight = raw_weight.permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(3)
        cross = (sampled * weight).sum(2).sum(1).permute(0, 2, 3, 1).reshape(batch_size, queries, self.reduction_dim)
        fused = (1 - self.fusion_factor) * self_out + self.fusion_factor * cross
        out = self.up_projection(fused * scale) * self.gate
        return out
