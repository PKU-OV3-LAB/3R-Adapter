"""Model configuration and checkpoint helpers."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class AdapterManifest:
    format_version: int = 1
    implementation: str = "adapter3r.Adapter3R"
    grid_layout: str = "legacy"
    sampling_layout: str = "legacy"
    patch_size: int = 14
    reference_radius: int = 2
    include_self: bool = True

    def __post_init__(self):
        if self.grid_layout not in {"legacy", "pixel_center"}:
            raise ValueError("invalid grid_layout")
        if self.sampling_layout not in {"legacy", "reference_major"}:
            raise ValueError("invalid sampling_layout")

    def dump(self, path):
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def unwrap_checkpoint(checkpoint):
    """Support raw state dictionaries and common wrappers."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a path or dictionary")
    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def adapter_state_dict(checkpoint, prefix="dec_adapters."):
    """Select adapter entries while leaving adapter-only weights unchanged."""
    state = unwrap_checkpoint(checkpoint)
    selected = {key: value for key, value in state.items() if prefix in key}
    return selected or state
