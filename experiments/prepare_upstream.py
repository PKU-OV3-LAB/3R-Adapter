"""Apply the shipped table compatibility patches in a fresh upstream worktree."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def normalize(source, suffix=".py"):
    if suffix != ".py":
        return re.sub(r"/(?:home|data|Users)/[A-Za-z0-9_./:+-]+", "external/assets", source).rstrip() + "\n"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = re.sub(r"/(?:home|data|Users)/[A-Za-z0-9_./:+-]+", "external/assets", node.value)
            # Dataset/model paths are supplied by the entry; normalize unrelated
            # machine defaults and omit obsolete documentation in this recipe.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
                    if not node.body:
                        node.body.append(ast.Pass())
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--output-worktree", required=True, type=Path)
    p.add_argument("--backbone", choices=["cut3r", "human3r"], required=True)
    p.add_argument("--asset-source", type=Path,
                   help="optional existing checkout containing separately obtained native submodules and model assets")
    args = p.parse_args()
    recipe_dir = Path(__file__).resolve().parent / "patches"
    recipe = json.loads((recipe_dir / f"{args.backbone}-recipe.json").read_text())
    repo = args.repo.resolve(strict=True)
    dest = args.output_worktree.resolve()
    if dest.exists():
        raise FileExistsError("choose a new upstream worktree")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach",
                    str(dest), recipe["revision"]], check=True)
    for relative, expected in recipe["base_normalized_sha256"].items():
        path = dest / relative
        text = normalize(path.read_text(), path.suffix)
        if digest(text) != expected:
            raise ValueError(f"unexpected upstream source: {relative}")
        path.write_text(text)
    patch = recipe_dir / f"{args.backbone}-tables.patch"
    subprocess.run(["git", "-C", str(dest), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(dest), "apply", str(patch)], check=True)
    for relative, expected in recipe["patched_sha256"].items():
        path = dest / relative
        if digest(path.read_text()) != expected:
            raise ValueError(f"patched source mismatch: {relative}")
        if path.suffix == ".py":
            compile(path.read_text(), str(path), "exec")
    if args.asset_source:
        asset_root = args.asset_source.resolve(strict=True)
        for relative in ("src/croco", "src/dinov2", "src/models"):
            source, target = asset_root / relative, dest / relative
            if source.is_dir() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve(), target_is_directory=True)
            elif source.is_dir() and target.is_dir() and not any(target.iterdir()):
                target.rename(target.with_name(target.name + ".empty-submodule"))
                target.symlink_to(source.resolve(), target_is_directory=True)
    print(json.dumps({"backbone": args.backbone, "worktree": str(dest),
                      "revision": recipe["revision"], "patched_files": list(recipe["patched_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
