"""Shared runtime checks for the public table entrypoints."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch


ROOT = Path(__file__).resolve().parents[1]


def bundled(name):
    return ROOT / "third_party" / name


def verify_bundle(repo):
    repo = Path(repo)
    manifest = json.loads((repo / "SOURCE.json").read_text())
    for relative, item in manifest["files"].items():
        if sha(repo / relative) != item["sha256"]:
            raise ValueError(f"bundled source changed: {relative}")
    return manifest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    (path / ".run-start").write_text("fresh directory\n")
    return path


def paths(repo, source=None):
    repo = Path(repo).resolve(strict=True)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    if (repo / "src/dinov2").is_dir():
        sys.path.insert(0, str(repo / "src/dinov2"))
    if source:
        sys.path.insert(0, str(Path(source).parent))
    # Checkpoint snapshots must not supersede the declared upstream checkout.
    if (repo / "add_ckpt_path.py").is_file():
        import add_ckpt_path
        add_ckpt_path.add_path_to_dust3r = lambda _: None
    return repo


def install_adapter(grid, sampling, region=0.2, fusion=0.1, scale=0.5):
    sys.path.insert(0, str(ROOT))
    from adapter3r import Adapter3R
    import dust3r.modules as modules
    class ConfiguredAdapter(Adapter3R):
        def __init__(self, *args, **kwargs):
            kwargs.update(grid_layout=grid, sampling_layout=sampling,
                          region_scale=region, fusion_factor=fusion)
            super().__init__(*args, **kwargs)
        def forward(self, *args, **kwargs):
            kwargs.setdefault("scale", scale)
            return super().forward(*args, **kwargs)
    modules.Adapter3R = ConfiguredAdapter
    model = sys.modules.get("dust3r.model")
    if model is not None:
        model.Adapter3R = ConfiguredAdapter
    return ConfiguredAdapter


def source_tree(source):
    tree = ast.parse(Path(source).read_text())
    # Respect the physical GPU chosen by the caller, including under CUDA.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and ast.unparse(target.value) == "os.environ"
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "CUDA_VISIBLE_DEVICES"):
                    node.value = ast.Call(func=ast.Attribute(value=target.value, attr="get", ctx=ast.Load()),
                                          args=[ast.Constant(value="CUDA_VISIBLE_DEVICES"), node.value],
                                          keywords=[])
    return ast.fix_missing_locations(tree)


def record(repo, checkpoint, sources, protocol):
    repo = Path(repo)
    result = dict(protocol=protocol, checkpoint_sha256=sha(checkpoint),
                  source_sha256={str(p): sha(p) for p in sources},
                  core_sha256=sha(ROOT / "adapter3r/core.py"))
    if (repo / "SOURCE.json").is_file():
        bundle = verify_bundle(repo)
        result.update(upstream_head=bundle["base_revision"],
                      upstream_diff_sha256=bundle["source_diff_sha256"],
                      bundled_manifest_sha256=sha(repo / "SOURCE.json"))
        return result
    result["upstream_head"] = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result["upstream_diff_sha256"] = hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"])).hexdigest()
    return result


def assert_model(model):
    from adapter3r import Adapter3R
    count = sum(isinstance(m, Adapter3R) for m in model.modules())
    if count == 0:
        raise ValueError("checkpoint/model has no 3R-Adapter modules")
    print(f"Loaded {count} 3R-Adapter modules", flush=True)
    return model


def wrap_loader():
    from dust3r.model import ARCroco3DStereo
    original = ARCroco3DStereo.from_pretrained
    ARCroco3DStereo.from_pretrained = classmethod(
        lambda cls, *args, **kwargs: assert_model(original(*args, **kwargs)))
