"""BEDLAM human, scene and depth evaluation with the paper's 15m scene cutoff."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
from .runtime import paths, install_adapter, source_tree, record, claim, wrap_loader, bundled
from .release_config import add_release_arguments, configure


def scene_mask(depth, valid, cutoff=15.0):
    return valid & (depth > 0) & (depth <= cutoff)


def configure_tree(source):
    tree = source_tree(source)
    masks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "depth_evaluation":
            for keyword in node.keywords:
                if keyword.arg == "max_depth":
                    keyword.value = ast.Constant(value=16)
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "valid_masks" for t in node.targets)):
            masks.append(node)
    if len(masks) != 1:
        raise ValueError("unsupported BEDLAM GT mask construction")
    masks[0].value = ast.parse(
        "[scene_mask(gt['depthmap'], gt['valid_mask'], 15.0) for gt in gts]", mode="eval").body
    return ast.fix_missing_locations(tree)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=bundled("human3r"))
    p.add_argument("--model-assets", type=Path, required=True,
                   help="separately acquired Human3R model assets directory")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--grid-layout", choices=["legacy", "pixel_center"])
    p.add_argument("--sampling-layout", choices=["legacy", "reference_major"])
    p.add_argument("--region-scale", type=float)
    p.add_argument("--fusion-factor", type=float)
    p.add_argument("--residual-scale", type=float)
    add_release_arguments(p, "human3r")
    p.add_argument("--max-sequences", type=int)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    config = configure(args, "human3r")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HUMAN3R_MODEL_ROOT"] = str(args.model_assets.resolve(strict=True))
    repo = paths(args.repo)
    checkpoint = args.checkpoint.resolve(strict=True)
    root = args.data_root.resolve(strict=True)
    source = repo / "eval/global_human/launch.py"
    install_adapter(args.grid_layout, args.sampling_layout, args.region_scale,
                    args.fusion_factor, args.residual_scale)
    wrap_loader()
    from eval.global_human.metadata import dataset_metadata
    metadata = dataset_metadata["bedlam"]
    metadata.update(img_path=str(root), split="Test", subsample=25)
    sequences = sorted(p.name for p in (root / "Test").iterdir() if p.is_dir())
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError("max-sequences must be positive")
        sequences = sequences[:args.max_sequences]
    if not sequences:
        raise ValueError("no BEDLAM Test sequences")
    metadata.update(seq_list=sequences, full_seq=False)
    tree = configure_tree(source)
    cli = ["--weights", str(checkpoint), "--output_dir", str(args.output_dir.resolve()),
           "--eval_dataset", "bedlam", "--eval_depth", "--eval_reconstruction",
           "--reset_interval", "100000000", "--size", "512"]
    if args.visualize:
        cli.append("--vis")
    scope = {"__name__": "_table3", "__file__": str(source), "scene_mask": scene_mask}
    if args.preflight:
        exec(compile(tree, str(source), "exec"), scope)
        upstream = scope["get_args_parser"]().parse_args(cli)
        print(json.dumps(dict(dataset="BEDLAM", sequences=len(sequences), scene_cutoff_m=15,
                              depth_metric_max_m=16, human_metrics=True,
                              grid_layout=args.grid_layout, sampling_layout=args.sampling_layout,
                              region_scale=args.region_scale, fusion_factor=args.fusion_factor,
                              residual_scale=args.residual_scale, seed=args.seed)))
        return
    output = claim(args.output_dir)
    manifest = record(repo, checkpoint, [source, repo / "eval/global_human/metadata.py", Path(__file__)],
                      dict(dataset="BEDLAM", scene_cutoff_m=15, depth_metric_max_m=16,
                           sequences=sequences, human_metrics=True, subsample=25,
                           grid_layout=args.grid_layout, sampling_layout=args.sampling_layout,
                           region_scale=args.region_scale, fusion_factor=args.fusion_factor,
                           residual_scale=args.residual_scale, seed=args.seed, release_config=config))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "effective_launch.py").write_text(ast.unparse(tree) + "\n")
    sys.argv = [str(source), *cli]
    scope["__name__"] = "__main__"
    exec(compile(tree, str(source), "exec"), scope)
    logs = list(output.glob("_error_log*.txt"))
    if not logs:
        raise ValueError("BEDLAM evaluation produced no metric logs")
    (output / ".complete").write_text("BEDLAM evaluation completed\n")


if __name__ == "__main__":
    main()
