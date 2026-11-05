#!/usr/bin/env python3
"""Collect released table metrics together with reproducibility metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "accuracy": "accuracy",
    "acc": "accuracy",
    "accmean": "accuracy",
    "complete": "completeness",
    "completeness": "completeness",
    "comp": "completeness",
    "compmean": "completeness",
    "chamfer": "chamfer",
    "normal": "normal_consistency",
    "normalconsistency": "normal_consistency",
    "nc": "normal_consistency",
    "ncmean": "normal_consistency",
    "ate": "ate",
    "rpet": "rpe_t",
    "rpetrans": "rpe_t",
    "rpetranslation": "rpe_t",
    "rper": "rpe_r",
    "rperot": "rpe_r",
    "rperotation": "rpe_r",
}


def _run_git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=text,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    if (repo / "SOURCE.json").is_file():
        from ..runners.runtime import verify_bundle
        bundle = verify_bundle(repo)
        return {"revision": bundle["base_revision"], "dirty": False,
                "diff_sha256": bundle["source_diff_sha256"], "untracked_files": [],
                "bundled_manifest_sha256": sha256_file(repo / "SOURCE.json")}
    revision = str(_run_git(repo, "rev-parse", "HEAD")).strip()
    diff = bytes(_run_git(repo, "diff", "--binary", "HEAD", text=False))
    untracked_raw = bytes(
        _run_git(
            repo,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            text=False,
        )
    )
    untracked = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_raw.split(b"\0")
        if item
    )
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff)
    for relative in untracked:
        digest.update(b"\0untracked\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        path = repo / relative
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_symlink():
            digest.update(
                os.readlink(path).encode("utf-8", errors="surrogateescape")
            )
    return {
        "revision": revision,
        "dirty": bool(diff or untracked),
        "diff_sha256": digest.hexdigest(),
        "untracked_files": untracked,
    }


def _normalise_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def normalise_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        canonical = METRIC_ALIASES.get(_normalise_key(key))
        if canonical is None or value in (None, ""):
            continue
        try:
            metrics[canonical] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"metric {key!r} is not numeric: {value!r}"
            ) from error
    if "chamfer" not in metrics and {
        "accuracy",
        "completeness",
    } <= metrics.keys():
        metrics["chamfer"] = (
            metrics["accuracy"] + metrics["completeness"]
        )
    if not metrics:
        raise ValueError("no supported released table metric was found")
    return metrics


def load_metrics(path: Path) -> dict[str, float]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON metrics must be an object: {path}")
        return normalise_metrics(payload)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"CSV has no data rows: {path}")
        return normalise_metrics(rows[-1])
    raise ValueError(f"only .json and .csv metric files are supported: {path}")


def parse_metric_specs(
    specs: list[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metrics: dict[str, Any] = {}
    sources: list[dict[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--metrics-file must use LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        if not label or not raw_path:
            raise ValueError("metric label and path must be non-empty")
        path = Path(raw_path).expanduser().resolve(strict=True)
        if label in metrics:
            raise ValueError(f"duplicate metric label: {label}")
        metrics[label] = load_metrics(path)
        sources.append(
            {"label": label, "file": path.name, "sha256": sha256_file(path)}
        )
    return metrics, sources


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    template: dict[str, Any] = {}
    if args.template:
        loaded = json.loads(args.template.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("manifest template must be a JSON object")
        template = loaded
    metrics, metric_sources = parse_metric_specs(args.metrics_file)
    run_root = getattr(args, "run_root", None)
    if run_root is not None:
        run_root = run_root.resolve(strict=True)
        if not (run_root / ".complete").is_file() or not (run_root / ".run-start").is_file():
            raise ValueError("run root has no successful completion record")
        started = (run_root / ".run-start").stat().st_mtime
        for spec in args.metrics_file:
            path = Path(spec.split("=", 1)[1]).expanduser().resolve(strict=True)
            if not path.is_relative_to(run_root) or path.stat().st_mtime < started:
                raise ValueError("metric file does not belong to this completed run")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("--command-json must be an array of strings")
    canonical_command = json.dumps(
        command, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    manifest = dict(template)
    manifest.update(
        {
            "schema_version": 1,
            "experiment": args.experiment,
            "table": args.table,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "repository": repository_state(args.repo),
            "checkpoint": {
                "id": checkpoint.name,
                "sha256": sha256_file(checkpoint),
            },
            "command_sha256": hashlib.sha256(canonical_command).hexdigest(),
            "metrics": metrics,
            "metric_sources": metric_sources,
        }
    )
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--command-json", required=True)
    parser.add_argument(
        "--metrics-file",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--template", type=Path)
    parser.add_argument("--run-root", type=Path,
                        help="require fresh metric files inside a completed runner directory")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
