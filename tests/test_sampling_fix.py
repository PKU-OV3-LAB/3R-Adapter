import importlib.util
from pathlib import Path
import subprocess

import pytest
import torch
from torch.nn import functional as F
from adapter3r import Adapter3R

ROOT = Path(__file__).parents[1]


def entry():
    spec = importlib.util.spec_from_file_location("nrgbd_entry", ROOT / "experiments/test3r_nrgbd_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ["legacy", "reference_major"])
def test_sampling_layout_preserves_consumed_grid(layout, monkeypatch):
    real = F.grid_sample
    captured = {}
    def capture(inputs, grid, **kwargs):
        captured["grid"] = grid.detach().clone()
        return real(inputs, grid, **kwargs)
    monkeypatch.setattr(F, "grid_sample", capture)
    model = Adapter3R(8, 4, K_sample=4, sampling_layout=layout).eval()
    with torch.no_grad():
        model.offset_head.weight.zero_()
        model.offset_head.bias.zero_()
    x, refs = torch.ones(1, 12, 8), torch.randn(1, 3, 12, 8)
    with torch.no_grad():
        out = model(x, refs, (3, 4), 1)
    assert out.shape == x.shape
    locations = model.create_grid(1, 3, 4, x.device).reshape(1, 12, 1, 2).expand(-1, -1, 4, -1)
    expected = locations.unsqueeze(1).expand(-1, 3, -1, -1, -1)
    if layout == "reference_major":
        expected = expected.permute(0, 1, 3, 2, 4)
    assert torch.equal(captured["grid"], expected.reshape(12, 12, 1, 2))


def test_full_reordered_queries_preserve_selection():
    torch.manual_seed(12)
    model = Adapter3R(8, 4, grid_layout="pixel_center", sampling_layout="reference_major").eval()
    x, refs = torch.randn(1, 12, 8), torch.randn(1, 3, 12, 8)
    order = torch.randperm(12)
    pos = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij"), -1).reshape(1, 12, 2)
    with torch.no_grad():
        dense = model(x, refs, (3, 4), 1)
        selected = model(x[:, order], refs, (3, 4), 1, sparse_pos=pos[:, order], pos_bound=(3, 4))
    assert torch.equal(selected, dense[:, order])
    with pytest.raises(ValueError):
        model(x, refs, (3, 4), 1, sparse_pos=pos)


def test_bfloat16_dense_grid_stays_float32(monkeypatch):
    model = Adapter3R(8, 4, grid_layout="pixel_center", sampling_layout="reference_major").eval()
    captured = {}
    create_grid = model.create_grid
    def capture_grid(*args, **kwargs):
        grid = create_grid(*args, **kwargs)
        captured["base"] = grid
        return grid
    monkeypatch.setattr(model, "create_grid", capture_grid)
    x, refs = torch.randn(1, 37*37, 8).bfloat16(), torch.randn(1, 2, 37*37, 8).bfloat16()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x, refs, (37, 37), 1)
    assert captured["base"].dtype == torch.float32
    assert torch.equal(captured["base"], create_grid(1, 37, 37, x.device))


def test_dataset_transform_preserves_evaluation_body():
    import ast
    source = 'def main(args):\n    resolution=(512,384)\n    datasets_all={"wrong": SevenScenes()}\n    return datasets_all\n'
    tree = entry().nrgbd_module(source, "dataset")
    seen = {}
    class Dataset:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def __len__(self):
            return 9
    namespace = {"NRGBD": Dataset}
    exec(compile(tree, "<test>", "exec"), namespace)
    assert list(namespace["main"](None)) == ["NRGBD"]
    assert seen == dict(split="test", ROOT="dataset", resolution=(512,384),
                        num_seq=1, full_video=True, kf_every=500)
    assert ast.unparse(tree).endswith("return datasets_all")


def test_nrbgd_validation_rejects_duplicate_and_missing_results(tmp_path):
    import re
    output = tmp_path / "NRGBD/512"
    output.mkdir(parents=True)
    log = output / "logs_0.txt"
    pattern = re.compile(r"(?P<scene_id>\w+): (?P<acc>[\d.]+), (?P<comp>[\d.]+), (?P<nc1>[\d.]+), (?P<nc2>[\d.]+)")
    log.write_text("".join(f"s{i}: 0.1, 0.2, 0.8, 0.9\n" for i in range(9)))
    assert entry().validate_metrics({"regex": pattern}, tmp_path)["sequence_count"] == 9
    log.write_text(log.read_text().replace("s8", "s0"))
    with pytest.raises(ValueError, match="unique"):
        entry().validate_metrics({"regex": pattern}, tmp_path)


def test_runner_refuses_existing_output_without_launch(tmp_path, monkeypatch):
    import os
    repo = tmp_path / "repo"
    (repo / "eval/mv_recon").mkdir(parents=True)
    (repo / "eval/mv_recon/launch.py").write_text("raise AssertionError('must not launch')\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("test")
    output = tmp_path / "output"
    output.mkdir()
    (output / "historic.txt").write_text("preserve")
    tools = tmp_path / "tools"
    tools.mkdir()
    fake_git = tools / "git"
    fake_git.write_text("#!/bin/sh\nprintf 'test-head\\n'\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    command = ["bash", str(ROOT / "experiments/run_table2_test3r_nrgbd.sh"),
               "--repo", str(repo), "--commit", "HEAD", "--checkpoint", str(checkpoint),
               "--gpu", "1", "--data-root", str(tmp_path), "--output-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (output / "run.log").exists()
    assert (output / "historic.txt").read_text() == "preserve"


def test_pi3_entry_installs_fixed_adapter_before_model_import(tmp_path):
    import json
    import sys
    repo = tmp_path / "pi3_repo"
    package = repo / "pi3/models/layers"
    package.mkdir(parents=True)
    for folder in [repo / "pi3", repo / "pi3/models", package]:
        (folder / "__init__.py").write_text("")
    (package / "modules.py").write_text("class Adapter3R:\n    pass\n")
    (repo / "pi3/models/pi3_Adapter3R.py").write_text(
        "from pi3.models.layers.modules import Adapter3R\n"
    )
    (repo / "mv_recon").mkdir()
    source = repo / "mv_recon/eval.py"
    source.write_text(
        "from pi3.models.pi3_Adapter3R import Adapter3R\n"
        "import torch\n"
        "m = Adapter3R(8, 4)\n"
        "assert m.grid_layout == 'pixel_center'\n"
        "assert m.sampling_layout == 'reference_major'\n"
        "assert m.create_grid(1, 37, 37, torch.device('cpu')).dtype == torch.float32\n"
    )
    record = tmp_path / "runtime.json"
    subprocess.run([sys.executable, str(ROOT / "experiments/pi3_entry.py"),
                    "--source-entry", str(source), "--grid-layout", "pixel_center",
                    "--sampling-layout", "reference_major", "--run-record", str(record)],
                   check=True)
    assert len(json.loads(record.read_text())["core_sha256"]) == 64
