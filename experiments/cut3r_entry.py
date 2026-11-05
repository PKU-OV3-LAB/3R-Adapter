"""Table 1 Oxford Day&Night scene and relative-pose evaluation on CUT3R."""
import argparse
import ast
import json
import math
import os
import importlib.util
from pathlib import Path
import sys
from runtime import paths, install_adapter, source_tree, record, claim, wrap_loader, bundled


def configure_tree(source, root):
    tree = source_tree(source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "OxfordDayAndNight"):
            for keyword in node.keywords:
                if keyword.arg == "ROOT":
                    keyword.value = ast.Constant(value=str(root))
    return ast.fix_missing_locations(tree)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=bundled("cut3r"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--task", choices=["mv", "relpose", "all"], default="all")
    p.add_argument("--grid-layout", choices=["legacy", "pixel_center"], default="legacy")
    p.add_argument("--sampling-layout", choices=["legacy", "reference_major"], default="legacy")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = paths(args.repo)
    checkpoint, root = args.checkpoint.resolve(strict=True), args.data_root.resolve(strict=True)
    install_adapter(args.grid_layout, args.sampling_layout)
    wrap_loader()
    metadata_file = repo / "eval/relpose/metadata.py"
    metadata_tree = source_tree(metadata_file)
    for node in ast.walk(metadata_tree):
        if isinstance(node, ast.Constant) and node.value == "data/oxford-day-and-night":
            node.value = str(root)
    spec = importlib.util.spec_from_file_location("eval.relpose.metadata", metadata_file)
    metadata_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metadata_module
    exec(compile(ast.fix_missing_locations(metadata_tree), str(metadata_file), "exec"), metadata_module.__dict__)
    dataset_metadata = metadata_module.dataset_metadata
    dataset_metadata["oxforddn"]["img_path"] = str(root)
    tasks = ["mv", "relpose"] if args.task == "all" else [args.task]
    output = None if args.preflight else claim(args.output_root)
    for task in tasks:
        source = repo / ("eval/mv_recon/launch.py" if task == "mv" else "eval/relpose/launch.py")
        tree = configure_tree(source, root)
        taskout = args.output_root.resolve() / task
        cli = ["--weights", str(checkpoint), "--output_dir", str(taskout), "--size", "512",
               "--eval_dataset", "oxford" if task == "mv" else "oxforddn"]
        if task == "mv":
            cli += ["--model_name", "ours"]
        scope = {"__name__": "_cut3r", "__file__": str(source)}
        if args.preflight:
            exec(compile(tree, str(source), "exec"), scope)
            parsed = scope["get_args_parser"]().parse_args(cli)
            if task == "mv":
                from eval.mv_recon.data import OxfordDayAndNight
                ds = OxfordDayAndNight(split="test", ROOT=str(root), resolution=(512,512),
                    num_seq=1, full_video=True, kf_every=2, max_thresh=20,
                    light_condition="day_night", mode="within_scene")
                print(json.dumps(dict(task=task, dataset=parsed.eval_dataset, sequences=len(ds))))
            else:
                print(json.dumps(dict(task=task, dataset=parsed.eval_dataset,
                                      pose_eval_stride=parsed.pose_eval_stride)))
            continue
        manifest = record(repo, checkpoint, [source, metadata_file, Path(__file__)],
                          dict(task=task, dataset="OxfordDayAndNight", kf_every=2,
                               max_frames=20, grid_layout=args.grid_layout, sampling_layout=args.sampling_layout,
                               encoder_residual_scale=0.3, decoder_residual_scale=0.5))
        (output / f"{task}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / f"{task}-effective.py").write_text(ast.unparse(tree) + "\n")
        if task == "mv":
            exec(compile(tree, str(source), "exec"), scope)
            scope["main"](scope["get_args_parser"]().parse_args(cli))
        else:
            sys.argv = [str(source), *cli]
            scope["__name__"] = "__main__"
            exec(compile(tree, str(source), "exec"), scope)
            expected = dataset_metadata["oxforddn"]["seq_list"]
            if not expected:
                raise ValueError("Oxford pose split contains no sequences")
            for sequence in expected:
                metric = taskout / f"{sequence}_eval_metric.txt"
                if not metric.is_file():
                    raise ValueError(f"missing pose metrics for {sequence}")
                values = scope["extract_metrics"](str(metric))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"nonfinite pose metrics for {sequence}")
    if output:
        (output / ".complete").write_text("all selected Oxford tasks completed\n")


if __name__ == "__main__":
    main()
