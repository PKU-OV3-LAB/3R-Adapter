"""Pi3 scene reconstruction and relative camera pose evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime import bundled, claim, record, verify_bundle, python_command, package_environment
from .release_config import add_release_arguments, configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=bundled("pi3"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--output-dir", "--output-root", dest="output_root", type=Path, required=True)
    p.add_argument("--task", choices=["mv", "relpose", "all"], default="all")
    p.add_argument("--grid-layout", choices=["legacy", "pixel_center"])
    p.add_argument("--sampling-layout", choices=["legacy", "reference_major"])
    p.add_argument("--region-scale", type=float)
    p.add_argument("--fusion-factor", type=float)
    p.add_argument("--residual-scale", type=float)
    add_release_arguments(p, "pi3")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    config = configure(args, "pi3")
    repo = args.repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    pretrained = args.pretrained.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if (repo / "SOURCE.json").is_file():
        verify_bundle(repo)
    tasks = ["mv", "relpose"] if args.task == "all" else [args.task]
    output = None if args.preflight else claim(args.output_root)
    for task in tasks:
        source = repo / ("mv_recon/eval.py" if task == "mv" else "relpose/eval_dist.py")
        out = args.output_root.resolve() / task
        command = [*python_command("pi3_runtime"),
                   "--source-entry", str(source),
                   "--grid-layout", args.grid_layout, "--sampling-layout", args.sampling_layout,
                   f"evaluation={'mv_recon' if task == 'mv' else 'relpose-distance'}",
                   "eval_datasets=[OxfordDayAndNight]", f"pi3.ckpt_path={checkpoint}",
                   f"pi3.pretrained_model_name_or_path={pretrained}", "device=cuda:0",
                   f"data.OxfordDayAndNight.cfg.data_dir={data}",
                   f"data.OxfordDayAndNight.cfg.cache_file={out}/dataset_cache.npy",
                   f"data.OxfordDayAndNight.seq_id_map={repo}/datasets/seq-id-maps/oxforddayandnight_mv-recon_seq-id-map-kf2.json",
                   f"output_dir={out}", f"hydra.run.dir={out}/hydra"]
        if args.preflight:
            command += ["--cfg", "job"]
            subprocess.run(command, cwd=repo, env=package_environment(), check=True)
            continue
        metadata = record(repo, checkpoint, [source, Path(__file__)],
                          dict(task=task, decoder_layers=list(range(17,25)),
                               grid_layout=args.grid_layout, sampling_layout=args.sampling_layout,
                               region_scale=args.region_scale, fusion_factor=args.fusion_factor,
                               residual_scale=args.residual_scale, seed=args.seed,
                               release_config=config,
                               command=command))
        (output / f"{task}-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (output / f"{task}.log").open("w") as log:
            subprocess.run(command, cwd=repo, env=package_environment(), check=True,
                           stdout=log, stderr=subprocess.STDOUT)
    if output:
        (output / ".complete").write_text("Pi3 Oxford evaluation complete\n")


if __name__ == "__main__":
    main()
