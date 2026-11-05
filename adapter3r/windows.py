"""Reference-view window construction."""

import torch


def reference_window(
    length: int, radius: int, device=None, include_self: bool = True
):
    """Return bounded, duplicate-free, fixed-width reference indices."""
    if length <= 0:
        raise ValueError("length must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if include_self:
        width = min(radius, (length - 1) // 2)
        indices = torch.arange(length, device=device)
        left, right = indices - width, indices + width
        left = (
            left
            + torch.clamp(-left, min=0)
            - torch.clamp(right - (length - 1), min=0)
        )
        return (
            left[:, None]
            + torch.arange(2 * width + 1, device=device)[None]
        ).long()
    count = min(2 * radius, length - 1)
    rows = []
    for index in range(length):
        candidates = sorted(
            (candidate for candidate in range(length) if candidate != index),
            key=lambda candidate: (abs(candidate - index), candidate),
        )
        rows.append(candidates[:count])
    return torch.tensor(rows, dtype=torch.long, device=device)


def sliding_window_indices_no_repeat(length, width, device):
    """Backward-compatible alias for the historical API."""
    return reference_window(length, width, device, include_self=True)
