"""Resolve release presets without inferring layouts from compatible tensor shapes."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("grid_layout", "sampling_layout", "region_scale",
          "fusion_factor", "residual_scale")


def add_release_arguments(parser, backbone=None):
    parser.add_argument("--release-config", default=(
        str(ROOT / "configs/release" / f"{backbone}_3seed.json") if backbone else None),
        help="release JSON preset, or 'none' for explicit historical/custom settings")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weights-root", type=Path,
                        help="directory containing separately distributed seed checkpoints")


def configure(args, backbone, historical_scale=0.3, training=False):
    selected = args.release_config
    config = None
    if selected and selected != "none":
        config = json.loads(Path(selected).read_text())
        if config["backbone"] != backbone or config["schema_version"] != 1:
            raise ValueError("release preset does not match the selected backbone")
        args.seed = config["default_seed"] if args.seed is None else args.seed
        if args.seed not in config["seeds"]:
            raise ValueError(f"seed must be one of {config['seeds']}")
    defaults = dict(grid_layout="pixel_center" if training else "legacy",
                    sampling_layout="reference_major" if training else "legacy",
                    region_scale=0.2, fusion_factor=0.1, residual_scale=historical_scale)
    for field in FIELDS:
        wanted = config[field] if config else defaults[field]
        explicit = getattr(args, field, None)
        if config and explicit is not None and explicit != wanted:
            raise ValueError(f"{field} conflicts with the release preset; use --release-config none for a custom run")
        setattr(args, field, wanted if explicit is None else explicit)
    # Both backbone implementations can pass the residual scale explicitly.
    for field in ("region_scale", "fusion_factor", "residual_scale"):
        os.environ["ADAPTER3R_" + field.upper()] = str(getattr(args, field))
    if config and not training:
        if args.checkpoint is None:
            if args.weights_root is None:
                raise ValueError("provide --weights-root or --checkpoint")
            args.checkpoint = args.weights_root / config["checkpoint_template"].format(seed=args.seed)
        from runtime import sha
        manifest = json.loads((ROOT / "results/release_weights.json").read_text())
        matches = [item for item in manifest["checkpoints"]
                   if item["backbone"] == backbone and item["seed"] == args.seed]
        if len(matches) != 1 or sha(args.checkpoint.resolve(strict=True)) != matches[0]["sha256"]:
            raise ValueError("checkpoint hash does not match the release preset and seed")
    elif not training and args.checkpoint is None:
        raise ValueError("custom/historical evaluation requires --checkpoint")
    return config


def training_overrides(config, seed, overrides):
    """Fix seed/budget/model settings, while allowing portable dataset paths."""
    if not config:
        return overrides
    training = config["training"]
    fixed = dict(training["overrides"])
    fixed[training["seed_key"]] = str(seed)
    remaining = []
    index = 0
    while index < len(overrides):
        item = overrides[index]
        if item == "--config-name":
            if index + 1 >= len(overrides) or overrides[index + 1] != training["config_name"]:
                raise ValueError("training config conflicts with the release preset")
            index += 2
            continue
        if item.startswith("--config-name="):
            if item.split("=", 1)[1] != training["config_name"]:
                raise ValueError("training config conflicts with the release preset")
        else:
            key, separator, value = item.partition("=")
            if key.lstrip("+~") in fixed:
                if key != key.lstrip("+~") or value != fixed[key]:
                    raise ValueError(f"{key} conflicts with the release preset")
            else:
                if separator and "(" in value and not value.startswith(('"', "'")):
                    item = key + "=" + json.dumps(value)
                remaining.append(item)
        index += 1
    # Quote entire expressions for Hydra, without shell evaluation.
    def encode(value):
        return json.dumps(value) if "(" in value else value
    return ["--config-name", training["config_name"], *remaining,
            *(f"{key}={encode(value)}" for key, value in fixed.items())]
