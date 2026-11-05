"""Pi3 integration helpers that do not import or modify Pi3."""

import torch
from torch import nn

from ..core import Adapter3R
from ..windows import reference_window


def make_pi3_adapters(
    embed_dim=1024,
    reduction_dim=128,
    layers=(17, 18, 19, 20, 21, 22, 23, 24),
    **kwargs,
):
    """Construct the eight Pi3 adapters used by the released checkpoint."""
    return nn.ModuleDict(
        {
            str(layer): Adapter3R(embed_dim, reduction_dim, **kwargs)
            for layer in layers
        }
    )


def pi3_reference_tokens(patch_tokens, batch_size, views, radius=2):
    """Arrange ``(B*N,HW,C)`` tokens as the adapter reference tensor."""
    if patch_tokens.shape[0] != batch_size * views:
        raise ValueError("B*N does not match the token batch")
    indices = reference_window(
        views, radius, patch_tokens.device, include_self=True
    )
    shaped = patch_tokens.reshape(
        batch_size, views, *patch_tokens.shape[1:]
    )
    references = shaped[:, indices].reshape(
        batch_size * views, indices.shape[1], *patch_tokens.shape[1:]
    )
    return references, indices


def apply_pi3_adapter(
    hidden,
    adapter,
    batch_size,
    views,
    img_shape,
    patch_size=14,
    patch_start_idx=5,
    radius=2,
    scale=0.3,
):
    """Apply an adapter while preserving Pi3 special tokens and layout."""
    original_shape = hidden.shape
    flat = hidden.reshape(batch_size * views, -1, hidden.shape[-1])
    special, patch = flat[:, :patch_start_idx], flat[:, patch_start_idx:]
    references, _ = pi3_reference_tokens(
        patch, batch_size, views, radius
    )
    out = adapter(
        patch,
        references,
        img_shape,
        patch_size,
        scale=scale,
    )
    merged = torch.cat((special, patch + out), 1).reshape(original_shape)
    return merged
