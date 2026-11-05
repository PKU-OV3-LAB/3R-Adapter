"""Validate complete measurement rows and their paired GT support."""
import math
import json
from pathlib import Path

BLOCKS = ("17", "19", "21", "23", "25", "31")
OFFSETS = (*(str(b) for b in range(17,25)), "all_layers", "all_layers_all_refs")


def validate_rows(attention, offsets, sample_ids, seed, models):
    ids = {str(i) for i in sample_ids}
    expected = {(sid, model, block, group) for sid in ids for model in models
                for block in BLOCKS for group in ("GTValid","GTInvalid")}
    keys = [(r["sample_id"],r["model"],r["block"],r["group"]) for r in attention]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("attention rows do not cover the required sample/model/block/group product")
    off_keys = [(r["sample_id"],r["block"]) for r in offsets]
    expected_offsets = {(sid,block) for sid in ids for block in OFFSETS}
    if len(off_keys) != len(set(off_keys)) or set(off_keys) != expected_offsets:
        raise ValueError("offset rows do not cover every sample and all ten block/union labels")
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           "configs/oxford_correspondence_samples.json").read_text())
    metadata = {str(s["sample_id"]):(s["scene"],s["sequence"],s["stratum"].capitalize())
                for s in manifest["samples"]}
    for row in [*attention, *offsets]:
        values = (row["site"],row["sequence"],row["domain"])
        if metadata.get(row["sample_id"]) != values:
            raise ValueError("sample site, sequence or domain differs from the fixed input selection")
    support = {}
    for row in attention:
        if int(row["seed"]) != seed:
            raise ValueError("attention seed differs from its manifest")
        den = int(row["denominator"])
        if den < 0:
            raise ValueError("negative attention support")
        key = (row["sample_id"],row["group"])
        if key in support and support[key] != den:
            raise ValueError("GT support differs between models or blocks")
        support[key] = den
    index = {(r["sample_id"],r["block"]):r for r in offsets}
    for row in offsets:
        if int(row["seed"]) != seed:
            raise ValueError("offset seed differs from its manifest")
        den = int(row["denominator"])
        pair_den = support[(row["sample_id"],"GTValid")]
        query_union = row["block"] == "all_layers_all_refs"
        mechanism = "any_reference" if query_union else "all_offsets"
        if row["mechanism"] != mechanism:
            raise ValueError("offset mechanism does not match its counting unit")
        if den < 0 or (not query_union and den != pair_den) or (
                query_union and not (den <= pair_den <= 2*den)):
            raise ValueError("offset support is inconsistent with the fixed two-reference GT support")
        field = "candidates_per_reference" if query_union else "candidates"
        expected_budget = 32 if row["block"].startswith("all_layers") else 4
        if int(row[field]) != expected_budget:
            raise ValueError("candidate budget does not match the block/union label")
        for field, value in row.items():
            if field.endswith("_hits") and value:
                hits = float(value)
                if not math.isfinite(hits) or not hits.is_integer() or not 0 <= hits <= den:
                    raise ValueError("invalid offset hit count")
        if not query_union:
            union = index[(row["sample_id"],"all_layers")]
            for radius in range(5):
                field = f"local_r{radius}_hits"
                if int(row[field]) > int(union[field]):
                    raise ValueError("all-layer coverage is below constituent-layer coverage")
