import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "experiments" / "collect_results.py"
SPEC = importlib.util.spec_from_file_location("collect_results", SCRIPT)
collect_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_results)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_normalise_released_table_metrics_and_ignore_unknown_fields():
    metrics = collect_results.normalise_metrics(
        {
            "Acc-mean": "0.1",
            "Comp-mean": 0.2,
            "NC-mean": "0.8",
            "UnsupportedMetric": 99,
        }
    )
    assert metrics == {
        "accuracy": 0.1,
        "completeness": 0.2,
        "chamfer": pytest.approx(0.3),
        "normal_consistency": 0.8,
    }


def test_load_relpose_csv_uses_last_row(tmp_path):
    path = tmp_path / "relpose.csv"
    path.write_text(
        "ATE,RPE trans,RPE rot\n1,2,3\n0.258,0.199,3.199\n",
        encoding="utf-8",
    )
    assert collect_results.load_metrics(path) == {
        "ate": 0.258,
        "rpe_t": 0.199,
        "rpe_r": 3.199,
    }


def test_strict_collection_rejects_incomplete_old_and_external_metrics(tmp_path):
    import os
    from types import SimpleNamespace
    root = tmp_path / "run"
    root.mkdir()
    metric = root / "metric.json"
    metric.write_text('{"acc": 0.1, "comp": 0.2}')
    args = SimpleNamespace(template=None, metrics_file=[f"scene={metric}"], run_root=root)
    with pytest.raises(ValueError, match="completion"):
        collect_results.build_manifest(args)
    (root / ".run-start").write_text("start")
    (root / ".complete").write_text("done")
    os.utime(metric, (1, 1))
    with pytest.raises(ValueError, match="belong"):
        collect_results.build_manifest(args)
    outside = tmp_path / "outside.json"
    outside.write_text(metric.read_text())
    args.metrics_file = [f"scene={outside}"]
    with pytest.raises(ValueError, match="belong"):
        collect_results.build_manifest(args)


def test_cli_records_provenance_and_metrics(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-m", "initial")
    revision = git(repo, "rev-parse", "HEAD").stdout.strip()
    tracked.write_text("v2\n", encoding="utf-8")
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"checkpoint")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({"accuracy": 0.08, "complete": 0.12}),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    assert (
        collect_results.main(
            [
                "--experiment",
                "table2-test3r-nrgbd",
                "--table",
                "Table 2",
                "--repo",
                str(repo),
                "--checkpoint",
                str(checkpoint),
                "--command-json",
                '["python","eval.py"]',
                "--metrics-file",
                f"nrgbd={metrics}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["repository"]["revision"] == revision
    assert manifest["repository"]["dirty"] is True
    assert len(manifest["repository"]["diff_sha256"]) == 64
    assert "path" not in manifest["repository"]
    assert len(manifest["checkpoint"]["sha256"]) == 64
    assert manifest["checkpoint"]["id"] == "model.bin"
    assert "path" not in manifest["checkpoint"]
    assert "host" not in manifest
    assert "command" not in manifest
    assert len(manifest["command_sha256"]) == 64
    assert manifest["metric_sources"][0]["file"] == "metrics.json"
    assert "path" not in manifest["metric_sources"][0]
    assert manifest["metrics"]["nrgbd"] == {
        "accuracy": 0.08,
        "completeness": 0.12,
        "chamfer": pytest.approx(0.2),
    }
