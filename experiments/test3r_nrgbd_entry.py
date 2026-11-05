#!/usr/bin/env python3
"""Run all nine NRGBD sequences using the pinned Test3R evaluator."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import shutil
import sys
import textwrap


def nrgbd_module(source: str, data_root: str) -> ast.Module:
    """Replace only the dataset constructor; retain upstream evaluation math."""
    tree = ast.parse(source)
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise ValueError("expected one upstream main function")
    main = mains[0]
    targets = [
        (i, n) for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "datasets_all" for t in n.targets)
    ]
    if len(targets) != 1:
        raise ValueError("expected one top-level datasets_all assignment")
    index, _ = targets[0]
    replacement = ast.parse(
        'datasets_all = {"NRGBD": NRGBD(split="test", ROOT='
        + repr(data_root)
        + ', resolution=resolution, num_seq=1, full_video=True, kf_every=500)}\n'
        'if len(datasets_all["NRGBD"]) != 9:\n'
        '    raise ValueError("NRGBD protocol requires exactly nine sequences")\n'
    ).body
    main.body[index:index + 1] = replacement
    return ast.fix_missing_locations(tree)


def validate_metrics(namespace: dict, output: Path, dataset="NRGBD", expected_count=9) -> dict:
    paths = sorted((output / dataset).rglob("logs_*.txt"))
    paths = [p for p in paths if p.name != "logs_all.txt"]
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            match = namespace["regex"].match(line)
            if match:
                rows.append(match.groupdict())
    scenes = [row["scene_id"].strip() for row in rows]
    if len(rows) != expected_count or len(set(scenes)) != expected_count:
        raise ValueError(f"expected {expected_count} unique {dataset} results, found {len(rows)} rows / {len(set(scenes))} scenes")
    values = {}
    for key in rows[0]:
        if key == "scene_id":
            continue
        numbers = [float(row[key]) for row in rows]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError(f"non-finite metric: {key}")
        values[key] = sum(numbers) / expected_count
    values["nc"] = (values["nc1"] + values["nc2"]) / 2
    values["chamfer"] = values["acc"] + values["comp"]
    return {"dataset": dataset, "sequence_count": expected_count, "scenes": scenes, "metrics": values}


def historical_protocol(namespace):
    """Restore the loss and scene order used by the audited nine-scene run."""
    import dust3r.inference as inference
    import dust3r.model as model
    import dust3r.modules as modules
    from eval.mv_recon.data import NRGBD
    loss = ast.parse(inspect.getsource(inference.train_criterion))
    calls = [n for n in ast.walk(loss) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "self_depth_consistency_loss"]
    if len(calls) != 1:
        raise ValueError("unsupported upstream TTA loss")
    keyword = next(k for k in calls[0].keywords if k.arg == "with_conf")
    keyword.value = ast.Constant(value=False)
    exec(compile(ast.fix_missing_locations(loss), "<NRGBD loss>", "exec"), inference.__dict__)
    order = ast.parse(textwrap.dedent(inspect.getsource(NRGBD.load_all_scenes)))
    targets = [n for n in ast.walk(order) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "scenes" for t in n.targets)]
    if len(targets) != 1:
        raise ValueError("unsupported upstream NRGBD scene listing")
    targets[0].value = ast.Call(func=ast.Name(id="sorted", ctx=ast.Load()),
                               args=[targets[0].value], keywords=[])
    scope = dict(NRGBD.load_all_scenes.__globals__)
    exec(compile(ast.fix_missing_locations(order), "<NRGBD order>", "exec"), scope)
    NRGBD.load_all_scenes = scope["load_all_scenes"]
    core = Path(__file__).resolve().parents[1] / "adapter3r/core.py"
    if not core.is_file():
        core = Path(__file__).resolve().with_name("core.py")
    spec = importlib.util.spec_from_file_location("_nrgbd_adapter_core", core)
    local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local)
    modules.Adapter3R = local.Adapter3R
    model.Adapter3R = local.Adapter3R
    return core


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-launch", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    known, remaining = parser.parse_known_args(argv)
    launch = known.source_launch.resolve(strict=True)
    repo = launch.parents[2]
    data_root = known.data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError("NRGBD data root must be a directory")
    source = launch.read_text()
    tree = nrgbd_module(source, str(data_root))
    sys.path.insert(0, str(repo))
    namespace = {"__file__": str(launch), "__name__": "_adapter3r_test3r"}
    exec(compile(tree, str(launch), "exec"), namespace)
    core = historical_protocol(namespace)
    args = namespace["get_args_parser"]().parse_args(remaining)
    if args.batch_index != 9:
        raise ValueError("NRGBD requires --batch_index 9")
    if known.preflight:
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        stop = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "accelerator" for t in n.targets))
        function.body = function.body[:stop] + [ast.Return(value=ast.Name(id="datasets_all", ctx=ast.Load()))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(launch), "exec"), namespace)
        datasets = namespace["main"](args)
        print(json.dumps({"dataset": list(datasets), "sequences": len(datasets["NRGBD"]),
                          "indices": list(range(9)), "kf_every": 500}))
        return
    args.peft_parts = ast.literal_eval(args.peft_parts)
    args.peft_layers = ast.literal_eval(args.peft_layers)
    output = Path(args.output_dir).resolve(strict=True)
    # The shell runner claims a fresh directory before redirecting run.log.
    if {p.name for p in output.iterdir()} - {"run.log", ".run-start"}:
        raise ValueError("output directory already contains evaluation artifacts")
    sources = [launch, repo / "dust3r/model.py", repo / "dust3r/modules.py",
               repo / "dust3r/inference.py", repo / "eval/mv_recon/data.py",
               core, Path(__file__).resolve()]
    record = {
        "dataset": "NRGBD", "indices": list(range(9)), "kf_every": 500,
        "data_root": str(data_root), "arguments": remaining,
        "tta_confidence_weighting": False, "scene_order": "sorted",
        "adapter_grid_layout": "legacy", "adapter_sampling_layout": "legacy",
        "checkpoint_sha256": file_sha(Path(args.weights)),
        "source_sha256": {str(p.relative_to(repo)) if p.is_relative_to(repo) else p.name:
                          file_sha(p) for p in sources},
        "effective_launch_sha256": hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    for path in sources:
        shutil.copyfile(path, output / ("upstream_" + path.name if path == launch else path.name))
    (output / "effective_launch.py").write_text(ast.unparse(tree) + "\n")
    for index in range(9):
        args.index = index
        namespace["gloabl_seed"](42)
        namespace["main"](args)
    result = validate_metrics(namespace, output)
    (output / "nrgbd_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "nrgbd_metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
