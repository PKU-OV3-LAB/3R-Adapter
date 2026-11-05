"""Install the fixed adapter before running the upstream Pi3 evaluation entry."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from ..core import Adapter3R
from .runtime import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-entry", required=True, type=Path)
    parser.add_argument("--grid-layout", choices=["legacy", "pixel_center"], default="legacy")
    parser.add_argument("--sampling-layout", choices=["legacy", "reference_major"], default="legacy")
    parser.add_argument("--run-record", type=Path)
    args, remaining = parser.parse_known_args()
    source = args.source_entry.resolve(strict=True)
    repo = source.parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(source.parent))
    core = ROOT / "adapter3r/core.py"
    if args.run_record:
        args.run_record.write_text(json.dumps({
            "grid_layout": args.grid_layout, "sampling_layout": args.sampling_layout,
            "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
            "source_entry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "arguments": remaining,
        }, indent=2) + "\n")
    import pi3.models.layers.modules as modules
    class ConfiguredAdapter(Adapter3R):
        def __init__(self, *values, **kwargs):
            kwargs.setdefault("grid_layout", args.grid_layout)
            kwargs.setdefault("sampling_layout", args.sampling_layout)
            super().__init__(*values, **kwargs)
    modules.Adapter3R = ConfiguredAdapter
    # Some package initializers import the model before the assignment above.
    model = sys.modules.get("pi3.models.pi3_Adapter3R")
    if model is not None and hasattr(model, "Adapter3R"):
        model.Adapter3R = ConfiguredAdapter
    print(f"Adapter3R runtime: grid={args.grid_layout}, sampling={args.sampling_layout}, core={core}")
    sys.argv = [str(source), *remaining]
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
