"""Helpers for integrating 3R-Adapter with external models."""

from .human3r import apply_human3r_adapter, make_human3r_adapters
from .pi3 import apply_pi3_adapter, make_pi3_adapters, pi3_reference_tokens

__all__ = [
    "apply_human3r_adapter",
    "make_human3r_adapters",
    "apply_pi3_adapter",
    "make_pi3_adapters",
    "pi3_reference_tokens",
]
