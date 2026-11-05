"""Train adapters with a prepared upstream checkout and explicit Hydra overrides."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from runtime import paths, install_adapter, claim, record, bundled
from release_config import ROOT, add_release_arguments, configure, training_overrides


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["pi3", "cut3r", "human3r"], required=True)
    p.add_argument("--repo", type=Path)
    p.add_argument("--model-assets", type=Path)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--pretrained-human", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--grid-layout", choices=["legacy", "pixel_center"])
    p.add_argument("--sampling-layout", choices=["legacy", "reference_major"])
    p.add_argument("--region-scale", type=float)
    p.add_argument("--fusion-factor", type=float)
    p.add_argument("--residual-scale", type=float)
    add_release_arguments(p)
    p.add_argument("--preflight", action="store_true")
    args, overrides = p.parse_known_args()
    if args.release_config is None and args.backbone in {"pi3", "human3r"}:
        args.release_config = str(ROOT / "configs/release" / f"{args.backbone}_3seed.json")
    config = configure(args, args.backbone, historical_scale=0.3 if args.backbone == "pi3" else 0.5,
                       training=True)
    overrides = training_overrides(config, args.seed, overrides)
    if not overrides:
        raise ValueError("provide the upstream Hydra training config and dataset/budget overrides")
    if config is None and args.seed is not None:
        seed_key = "random_seed" if args.backbone == "pi3" else "seed"
        overrides.append(f"{seed_key}={args.seed}")
    if any("resume=" in value and not value.endswith("resume=null") for value in overrides):
        raise ValueError("this entry requires fresh training from base weights")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = (args.repo or bundled("pi3_train" if args.backbone == "pi3" else args.backbone)).resolve(strict=True)
    if args.backbone == "human3r":
        if args.model_assets is None:
            raise ValueError("Human3R requires --model-assets")
        if config and args.pretrained_human is None:
            raise ValueError("the Human3R release preset requires --pretrained-human base weights")
        os.environ["HUMAN3R_MODEL_ROOT"] = str(args.model_assets.resolve(strict=True))
    checkpoint = args.pretrained.resolve(strict=True)
    entry = repo / ("scripts/train_pi3.py" if args.backbone == "pi3" else "src/train.py")
    if not entry.is_file():
        raise FileNotFoundError(entry)
    if args.backbone == "pi3":
        command = [str(Path(__import__("sys").executable).parent / "torchrun"),
                   "--standalone", "--nproc_per_node=1", str(entry),
                   *overrides,
                   f"model.ckpt={checkpoint}", f"log.output_dir={args.output_dir.resolve()}",
                   f"log.ckpt_dir={args.output_dir.resolve()}/ckpts",
                   "train.resume=null", "train.auto_resume=false",
                   "model.peft_layers=[[],[17,18,19,20,21,22,23,24]]"]
    else:
        command = [__import__("sys").executable, str(entry), *overrides,
                   f"pretrained={checkpoint}", f"output_dir={args.output_dir.resolve()}",
                   "hydra.job.chdir=false"]
        if args.pretrained_human:
            command.append(f"pretrained_mhmr={args.pretrained_human.resolve(strict=True)}")
        command += ["resume=null", f"logdir={args.output_dir.resolve()}/logs",
                    f"hydra.run.dir={args.output_dir.resolve()}/hydra"]
    # Keep each Hydra key once; base initialization and output paths above win.
    seen = set()
    unique = []
    for item in reversed(command):
        key, separator, _ = item.partition("=")
        if separator:
            if key in seen:
                continue
            seen.add(key)
        unique.append(item)
    command = list(reversed(unique))
    # Subprocesses install the configured adapter before upstream model imports.
    command = [__import__("sys").executable, str(Path(__file__).with_name("training_entry.py")),
               "--repo", str(repo), "--backbone", args.backbone,
               "--grid-layout", args.grid_layout, "--sampling-layout", args.sampling_layout,
               "--region-scale", str(args.region_scale), "--fusion-factor", str(args.fusion_factor),
               "--residual-scale", str(args.residual_scale),
               "--", *command]
    if args.preflight:
        print(json.dumps({"command": command, "fresh_initialization": True, "resume": None,
                          "release_config": config, "seed": args.seed}, indent=2))
        return
    output = claim(args.output_dir)
    (output / "manifest.json").write_text(json.dumps(
        record(repo, checkpoint, [entry, Path(__file__)], dict(backbone=args.backbone,
            grid_layout=args.grid_layout, sampling_layout=args.sampling_layout,
            region_scale=args.region_scale, fusion_factor=args.fusion_factor,
            residual_scale=args.residual_scale, seed=args.seed, release_config=config,
            fresh_initialization=True, command=command)), indent=2) + "\n")
    with (output / "run.log").open("w") as log:
        subprocess.run(command, cwd=repo, check=True, stdout=log, stderr=subprocess.STDOUT)
    (output / ".complete").write_text("training complete\n")


if __name__ == "__main__":
    main()
