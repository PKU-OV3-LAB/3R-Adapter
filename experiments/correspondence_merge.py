"""Merge disjoint, completed shards of one fixed Oxford measurement run."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from correspondence_validation import validate_rows


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifests, checks, attention, offsets, ranks, rank_sums, raster_sums = [], [], [], [], [], [], []
    for folder in a.inputs:
        if not (folder / "complete").is_file():
            p.error(f"incomplete input: {folder}")
        m = json.loads((folder / "manifest.json").read_text())
        manifests.append(m)
        att = read_csv(folder / "attention_per_sample.csv")
        off = read_csv(folder / "offset_per_sample.csv")
        models = tuple(sorted({r["model"] for r in att}))
        validate_rows(att, off, m["sample_ids"], int(m["seed"]), models)
        ids = set(map(str, m["sample_ids"]))
        if set(r["sample_id"] for r in att) != ids or set(r["sample_id"] for r in off) != ids:
            p.error(f"CSV sample IDs differ from manifest: {folder}")
        attention.extend(att)
        offsets.extend(off)
        checks.extend(json.loads((folder / "checks.json").read_text()))
        if m.get("export_profile"):
            ranks.extend(read_csv(folder / "rank_metadata.csv"))
            with np.load(folder / "full_token_sums.npz") as data:
                rank_sums.append(data["rank_sum"])
                raster_sums.append(data["raster_sum"])
    first = manifests[0]
    fields = ("seed", "checkpoint_sha256", "samples_sha256", "code_sha256", "blocks",
              "protocol", "shards", "readout_chunk", "export_profile")
    if any(any(m[k] != first[k] for k in fields) for m in manifests):
        p.error("shard measurement contracts differ")
    if len({tuple(sorted({r["model"] for r in read_csv(folder / "attention_per_sample.csv")}))
            for folder in a.inputs}) != 1:
        p.error("all shards must contain the same models")
    if sorted(m["shard"] for m in manifests) != list(range(first["shards"])):
        p.error("require every shard exactly once")
    ids = [sid for m in manifests for sid in m["sample_ids"]]
    selection = json.loads((Path(__file__).resolve().parents[1] /
                            "configs/oxford_correspondence_samples.json").read_text())
    expected = (selection["pilot32_ids"] if first["protocol"] == "pilot" else
                [s["sample_id"] for s in selection["samples"]])
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        p.error("sample IDs overlap or do not cover the fixed protocol")
    def unique_rows(rows, keys):
        values = [tuple(r[k] for k in keys) for r in rows]
        return len(values) == len(set(values))
    if not unique_rows(attention, ("sample_id", "model", "block", "group")) or not unique_rows(
            offsets, ("sample_id", "block")):
        p.error("duplicate measurement rows")
    if a.output.exists():
        p.error("output must be a fresh directory")
    a.output.mkdir(parents=True)
    write_csv(a.output / "attention_per_sample.csv", attention)
    write_csv(a.output / "offset_per_sample.csv", offsets)
    (a.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if ranks:
        write_csv(a.output / "rank_metadata.csv", ranks)
        np.savez_compressed(a.output / "full_token_sums.npz",
                            rank_sum=np.concatenate(rank_sums), raster_sum=np.concatenate(raster_sums))
    first.update(sample_ids=sorted(ids), sample_count=len(ids), shard=0, shards=1,
                 merged_shards=len(manifests), source_manifests=[str(x / "manifest.json") for x in a.inputs])
    (a.output / "manifest.json").write_text(json.dumps(first, indent=2) + "\n")
    (a.output / "complete").write_text("all fixed inputs completed in disjoint shards\n")
    print(json.dumps({"samples": len(ids), "seed": first["seed"], "output": str(a.output)}))


if __name__ == "__main__":
    main()
