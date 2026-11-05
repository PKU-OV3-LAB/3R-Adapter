"""Helpers for Human3R-style tensors without an upstream dependency."""

from torch import nn

from ..core import Adapter3R
from ..windows import reference_window


def make_human3r_adapters(
    layers, embed_dim, reduction_dim=128, **kwargs
):
    """Construct adapters for selected Human3R decoder layers."""
    return nn.ModuleDict(
        {
            str(layer): Adapter3R(embed_dim, reduction_dim, **kwargs)
            for layer in layers
        }
    )


def apply_human3r_adapter(
    tokens,
    adapter,
    img_shape,
    patch_size=14,
    radius=2,
    scale=0.3,
):
    """Apply an adapter to ``(B,N,HW,C)`` patch tokens."""
    if tokens.ndim != 4:
        raise ValueError("Human3R tokens must have shape (B,N,HW,C)")
    batch_size, views, token_count, channels = tokens.shape
    flat = tokens.reshape(batch_size * views, token_count, channels)
    indices = reference_window(
        views, radius, tokens.device, include_self=True
    )
    references = tokens[:, indices].reshape(
        batch_size * views, indices.shape[1], token_count, channels
    )
    out = adapter(
        flat,
        references,
        img_shape,
        patch_size,
        scale=scale,
    )
    merged = (flat + out).reshape_as(tokens)
    return merged
