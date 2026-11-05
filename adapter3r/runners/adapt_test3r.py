"""Self-supervised adaptation on 7-Scenes or NRGBD."""
import argparse
import ast
import json
from pathlib import Path
import sys
from .runtime import claim, record, source_tree, bundled, paths
from .test3r_runtime import historical_protocol, validate_metrics


def dataset_tree(source, dataset, root):
    tree = source_tree(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    targets = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "datasets_all" for t in n.targets)]
    if len(targets) != 1:
        raise ValueError("unsupported Test3R dataset entry")
    name, constructor, count, stride = (
        ("7scenes", "SevenScenes", 18, 200) if dataset == "7scenes"
        else ("NRGBD", "NRGBD", 9, 500))
    block = ast.parse(
        f"datasets_all = {{{name!r}: {constructor}(split='test', ROOT={str(root)!r}, "
        f"resolution=resolution, num_seq=1, full_video=True, kf_every={stride})}}\n"
        f"if len(datasets_all[{name!r}]) != {count}: raise ValueError('unexpected sequence count')\n"
    ).body
    main.body[targets[0]:targets[0]+1] = block
    return ast.fix_missing_locations(tree), name, count, stride


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=bundled("test3r"))
    p.add_argument("--dataset", choices=["7scenes", "nrgbd"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float)
    p.add_argument("--accum-iter", type=int, default=1)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo, root = args.repo.resolve(strict=True), args.data_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    paths(repo)
    source = repo / "eval/mv_recon/launch.py"
    tree, name, count, stride = dataset_tree(source, args.dataset, root)
    scope = {"__name__": "_table2", "__file__": str(source)}
    exec(compile(tree, str(source), "exec"), scope)
    core = historical_protocol(scope)
    # Use fixed ordering for all 18 sequences as well as all nine NRGBD scenes.
    from eval.mv_recon.data import SevenScenes
    import inspect
    import textwrap
    ordering = ast.parse(textwrap.dedent(inspect.getsource(SevenScenes.load_all_scenes)))
    for node in ast.walk(ordering):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "scenes" for t in node.targets)):
            node.value = ast.Call(func=ast.Name(id="sorted", ctx=ast.Load()), args=[node.value], keywords=[])
    env = dict(SevenScenes.load_all_scenes.__globals__)
    exec(compile(ast.fix_missing_locations(ordering), "<7-Scenes order>", "exec"), env)
    SevenScenes.load_all_scenes = env["load_all_scenes"]
    cli = ["--weights", str(checkpoint), "--batch_index", str(count),
           "--epoches", str(args.epochs), "--lr", str(args.lr or (1e-5 if count == 18 else 9e-5)),
           "--accum_iter", str(args.accum_iter), "--test_time_training", "--use_adapter",
           "--reduction_dim", "32", "--peft_parts", "['enc','dec']",
           "--peft_layers", "[list(range(24)),[1,3,5,7,9,11]]",
           "--output_dir", str(args.output_dir.resolve())]
    upstream = scope["get_args_parser"]().parse_args(cli)
    upstream.peft_parts = ["enc", "dec"]
    upstream.peft_layers = [list(range(24)), [1,3,5,7,9,11]]
    if args.preflight:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        stop = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id == "accelerator" for t in n.targets))
        fn.body = fn.body[:stop] + ast.parse("return datasets_all").body
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), scope)
        data = scope["main"](upstream)
        print(json.dumps(dict(dataset=name, sequences=len(data[name]), kf_every=stride,
                              epochs=args.epochs, lr=upstream.lr, accum_iter=args.accum_iter)))
        return
    output = claim(args.output_dir)
    metadata = record(repo, checkpoint, [source, core, Path(__file__)],
                      dict(dataset=name, sequences=count, kf_every=stride, tta_confidence_weighting=False,
                           epochs=args.epochs, lr=upstream.lr, accum_iter=args.accum_iter))
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "effective_launch.py").write_text(ast.unparse(tree) + "\n")
    for index in range(count):
        upstream.index = index
        scope["gloabl_seed"](42)
        scope["main"](upstream)
    result = validate_metrics(scope, output, dataset=name, expected_count=count)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
    (output / ".complete").write_text("all sequences complete\n")


if __name__ == "__main__":
    main()
