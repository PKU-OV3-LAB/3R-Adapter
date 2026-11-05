import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    @staticmethod
    def modulate(x, shift, scale):

        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:

        shift, scale = self.mlp(mod).chunk(2, dim=-1)  # [N, D]
        x = self.modulate(self.norm(x), shift, scale)  # [N, L, D]
        return  x

class Adapter(nn.Module):
    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, **kwargs):
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
        # nn.init.zeros_(self.up_projection.weight)
        nn.init.xavier_uniform_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward(self, x, scale=0.5, **kwargs):
        x = self.layer_norm(x)
        x = self.down_projection(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.up_projection(x)*scale
        return x

class GatedAdapter(nn.Module):
    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_dim = reduction_dim
        self.dropout = nn.Dropout(p=drop_rate_adapter)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.down_projection = nn.Linear(self.embed_dim, self.reduction_dim)
        self.activate = nn.GELU()
        self.up_projection = nn.Linear(self.reduction_dim, self.embed_dim)
        self.gate = nn.Parameter(torch.zeros(self.embed_dim))
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.xavier_uniform_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        nn.init.zeros_(self.gate)

    def forward(self, x, **kwargs):
        x = self.layer_norm(x)
        x = self.down_projection(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.up_projection(x)*self.gate
        return x


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

class CrossViewAdapter(nn.Module):
    """
    Deformable Cross-View Adapter
    
    x: (B, H*W, C)
    
    x_refs: (B, M, H*W, C)
    """

    def __init__(self, embed_dim, reduction_dim, drop_rate_adapter=0, K_sample=4, region_scale=0.2, gating=False, **kwargs):
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
        if gating:
            self.gate = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.gate = None

        # Offset and Weight Heads
        self.offset_head = nn.Linear(embed_dim, K_sample * 2)
        self.weight_head = nn.Linear(embed_dim, K_sample)
        self.reference_norm = nn.LayerNorm(embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.zeros_(self.up_projection.weight)
        # nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        if self.gate:
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

        # Back to token
        cross_view_out = cross_view_out.permute(0, 2, 3, 1).reshape(B, HW, C)

        # Final residual fusion
        output = self_view_out*scale
        
        if self.gate:
            output = cross_view_out * self.gate  # gated residual

        return output


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


class Adapter3R(nn.Module):
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
        if gating:
            self.gate = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.gate = 1

        # Offset and Weight Heads
        self.offset_head = nn.Linear(reduction_dim, K_sample * 2)
        self.weight_head = nn.Linear(reduction_dim, K_sample)
        
        # Reference mapping
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
        x_refs = self.reference_down_projection(x_refs)
        x_refs = x_refs.view(B, M, H, W, self.reduction_dim).permute(0, 1, 4, 2, 3)  # (B,M,C,H,W)
        x_refs = x_refs.unsqueeze(2).expand(-1, -1, self.K_sample, -1, -1, -1)
        x_refs = x_refs.reshape(B*M*self.K_sample, self.reduction_dim, H, W)

        # deformable sampling
        sampled_refs = F.grid_sample(x_refs, sampling_location, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_refs = sampled_refs.view(B, M, self.K_sample, self.reduction_dim, H, W)

        # weighted aggregation
        weight = weight.permute(0, 3, 1, 2)  # (B,K,H,W)
        weight = weight.unsqueeze(1).unsqueeze(3)  # (B,1,K,1,H,W)

        cross_view_out = (sampled_refs * weight).sum(2).sum(1)  # sum K and M
        cross_view_out = cross_view_out.permute(0, 2, 3, 1).reshape(B, HW, self.reduction_dim)  # Back to token

        # final residual fusion
        self_view_out = ((1-self.fusion_factor)*self_view_out + self.fusion_factor*cross_view_out) * scale
        
        # resume to high-dimensional feature
        self_view_out = self.up_projection(self_view_out)

        self_view_out = self_view_out * self.gate  # gated residual

        return self_view_out