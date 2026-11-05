"""Standalone 3R-Adapter implementation."""

from .core import Adapter3R
from .manifest import AdapterManifest, adapter_state_dict, unwrap_checkpoint
from .windows import reference_window, sliding_window_indices_no_repeat

__all__ = [
    "Adapter3R",
    "AdapterManifest",
    "adapter_state_dict",
    "unwrap_checkpoint",
    "reference_window",
    "sliding_window_indices_no_repeat",
]
