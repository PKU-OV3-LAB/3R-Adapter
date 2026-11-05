"""Summarize completed three-seed Oxford measurements; paired sequence bootstrap."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .correspondence_validation import validate_rows

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--bootstrap", type=int, default=10000)
a = p.parse_args()
if a.bootstrap < 1:
    p.error("bootstrap must be positive")
if a.output.exists():
    p.error("output must be a fresh directory")
SEEDS = (1111, 1112, 1113)
BLOCKSETS = {str(b): (str(b),) for b in (17, 19, 21, 23, 25, 31)}
BLOCKSETS["pool17_19_21_23"] = ("17", "19", "21", "23")
BLOCKSETS["pool19_21_23"] = ("19", "21", "23")
CI_METRICS = (
    "match_k1_r0", "match_k1_r1", "match_k1_r2", "match_k1_r3", "match_k1_r4",
    "match_k4_r1", "match_k4_r2", "match_k4_r3", "match_k4_r4",
    "gt_mass_r0", "gt_mass_r1", "gt_mass_r2", "gt_mass_r3", "gt_mass_r4",
    "coarse_match1_g2", "coarse_match1_g4", "coarse_gt_mass_g2", "coarse_gt_mass_g4",
    "top5pct_mass_g1", "top20pct_mass_g1", "top5pct_mass_g2", "top20pct_mass_g2",
    "top5pct_mass_g4", "top20pct_mass_g4", "entropy_g1", "entropy_g2", "entropy_g4",
    "reference_patch_share", "error_k1_px", "error_k4_px",
)


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def write(path, rows):
    names = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w") as f:
        w = csv.DictWriter(f, names)
        w.writeheader()
        w.writerows(rows)


attention, offsets = [], []
manifests = {}
for seed in SEEDS:
    folder = a.data / f"full{seed}"
    assert (folder / "complete").exists(), f"incomplete {seed}"
    m = json.loads((folder / "manifest.json").read_text())
    assert m["sample_count"] == 204 and m["seed"] == seed
    assert m["protocol"] == "full" and len(m["sample_ids"]) == len(set(m["sample_ids"])) == 204
    assert set(m["sample_ids"]) == set(range(204)), "require the complete fixed Camera Ready selection"
    manifests[seed] = m
    attention.extend(read(folder / "attention_per_sample.csv"))
    offsets.extend(read(folder / "offset_per_sample.csv"))
assert len({m["samples_sha256"] for m in manifests.values()}) == 1
assert len({tuple(m["sample_ids"]) for m in manifests.values()}) == 1
for seed in SEEDS:
    rows = [r for r in attention if r["model"] == "adapted" and int(r["seed"]) == seed]
    assert len(rows) == 204*6*2
    expected = {(str(sid), str(block), group) for sid in range(204)
                for block in (17,19,21,23,25,31) for group in ("GTValid","GTInvalid")}
    assert {(r["sample_id"],r["block"],r["group"]) for r in rows} == expected
    seed_offsets = [r for r in offsets if int(r["seed"]) == seed]
    expected_offsets = {(str(sid), block) for sid in range(204) for block in
                        [*(str(b) for b in range(17,25)), "all_layers", "all_layers_all_refs"]}
    assert len(seed_offsets) == len(expected_offsets)
    assert {(r["sample_id"],r["block"]) for r in seed_offsets} == expected_offsets
frozen = [r for r in attention if r["model"] == "frozen"]
assert len(frozen) == 204*6*2
assert all(int(r["seed"]) == 1111 for r in frozen)
assert {(r["sample_id"],r["block"],r["group"]) for r in frozen} == expected
for seed in SEEDS:
    validate_rows([r for r in attention if int(r["seed"]) == seed],
                  [r for r in offsets if int(r["seed"]) == seed],
                  manifests[seed]["sample_ids"], seed,
                  ("frozen","adapted") if seed == 1111 else ("adapted",))
support = {}
for r in attention:
    key = (r["sample_id"], r["block"], r["group"])
    val = (r["site"], r["sequence"], r["domain"], r["denominator"])
    if key in support: assert support[key] == val
    support[key] = val
before17 = {(r["sample_id"], r["group"]): r for r in frozen if r["block"] == "17"}
for r in attention:
    if r["block"] == "17" and r["model"] == "adapted":
        prior = before17[(r["sample_id"], r["group"])]
        assert all(float(v) == float(prior[k]) for k,v in r.items() if k.endswith("_sum") and v)
# Verify exact actual Q/K at the first insertion block across all released seeds.
hashes = {}
hook_audited = True
for seed in SEEDS:
    checks = json.loads((a.data / f"full{seed}" / "checks.json").read_text())
    qk_records = [r for r in checks if "block17_actual_qk_sha256" in r]
    qk_ids = [r["sample_id"] for r in qk_records]
    assert len(qk_ids) == len(set(qk_ids)) == 204 and set(qk_ids) == set(range(204)), "incomplete Q/K audit"
    hook_models = set()
    for check in checks:
        if "passive_hook_output_max_abs_diff" in check:
            diffs = check["passive_hook_output_max_abs_diff"]
            assert diffs and all(float(v) == 0.0 for v in diffs.values()), "passive hooks changed model output"
            hook_models.add(check["model"])
    required_hooks = {"frozen","adapted"} if seed == 1111 else {"adapted"}
    hook_audited = hook_audited and required_hooks.issubset(hook_models)
    for check in checks:
        if "sample_id" not in check: continue
        sid = check["sample_id"]
        value = check["block17_actual_qk_sha256"]["adapted"]
        if sid in hashes: assert hashes[sid] == value
        hashes[sid] = value
        if seed == 1111:
            assert check["block17_actual_qk_sha256"]["frozen"] == value

sequence_meta = sorted({(r["site"], r["sequence"], r["domain"]) for r in frozen})
assert len(sequence_meta) == 51 and len({x[0] for x in sequence_meta}) == 5
index = {(site, sequence): i for i,(site,sequence,domain) in enumerate(sequence_meta)}
rng = np.random.default_rng(20261103)
weights = np.zeros((a.bootstrap, 51), dtype=np.float64)
strata = defaultdict(list)
for i,(site,seq,domain) in enumerate(sequence_meta): strata[(site,domain)].append(i)
for ids in strata.values():
    weights[:, ids] = rng.multinomial(len(ids), np.ones(len(ids))/len(ids), a.bootstrap)

fields = sorted({k[:-4] for r in attention for k in r if k.endswith("_sum") and r[k]})
field_index = {name: i+1 for i,name in enumerate(fields)}
arrays = defaultdict(lambda: np.zeros((51, len(fields)+1)))
for r in attention:
    key = (int(r["seed"]), r["model"], r["block"], r["group"], r["domain"])
    i = index[(r["site"], r["sequence"])]
    target = arrays[key]
    target[i, 0] += float(r["denominator"])
    for metric, j in field_index.items():
        v = r.get(metric+"_sum")
        if v: target[i,j] += float(v)


def get(model, seed, blocks, group, domain):
    domains = ("Day", "Night") if domain == "All" else (domain,)
    out = np.zeros((51,len(fields)+1))
    for b in blocks:
        for d in domains:
            out += arrays[(seed, model, b, group, d)]
    return out


def unit(metric):
    if metric.startswith("error_"): return "px", 1.
    if metric.startswith("entropy_"): return "normalized_entropy", 1.
    if "enrichment" in metric: return "ratio", 1.
    return "percent", 100.


def interval(numerator, denominator):
    boot_num = np.stack([weights @ x for x in numerator], axis=1)
    boot_den = np.stack([weights @ x for x in denominator], axis=1)
    assert (boot_den > 0).all()
    return (boot_num / boot_den).mean(axis=1)


summary, recovery = [], []
for blocklabel, blocks in BLOCKSETS.items():
    for domain in ("Day", "Night", "All"):
        for group in ("GTValid", "GTInvalid"):
            before = get("frozen", 1111, blocks, group, domain)
            after = np.stack([get("adapted", s, blocks, group, domain) for s in SEEDS])
            assert np.all(after[:,:,0] == before[None,:,0])
            if not before[:,0].sum(): continue
            for metric, col in field_index.items():
                if group == "GTInvalid" and metric.startswith(("match_", "error_", "pixel_", "gt_", "uniform_gt_", "coarse_match", "coarse_gt")): continue
                u, scale = unit(metric)
                bval = scale * before[:,col].sum()/before[:,0].sum()
                avals = scale * after[:,:,col].sum(axis=1)/after[:,:,0].sum(axis=1)
                delta = avals-bval
                common = dict(block=blocklabel, domain=domain, group=group, metric=metric, unit=u,
                              valid_pair_layer_count=int(before[:,0].sum()))
                brow = dict(common, model="frozen", mean=bval)
                arow = dict(common, model="adapted", mean=avals.mean(), seed_sd=avals.std(ddof=1),
                            seed1111=avals[0], seed1112=avals[1], seed1113=avals[2])
                drow = dict(common, delta_mean=delta.mean(), delta_seed_sd=delta.std(ddof=1),
                            delta1111=delta[0], delta1112=delta[1], delta1113=delta[2])
                if metric in CI_METRICS:
                    bb = interval(before[None,:,col], before[None,:,0]) * scale
                    ab = interval(after[:,:,col], after[:,:,0]) * scale
                    for row, vec in ((brow,bb), (arow,ab), (drow,ab-bb)):
                        row["ci95_low"], row["ci95_high"] = np.quantile(vec, (.025,.975))
                summary.extend((brow,arow))
                recovery.append(drow)

off_fields = sorted({k for r in offsets for k in r if (k.endswith("_hits") or k=="error_sum") and r[k]})
off_arrays = defaultdict(lambda: np.zeros((51,len(off_fields)+1)))
for r in offsets:
    target = off_arrays[(int(r["seed"]),r["block"],r["mechanism"],r["domain"])]
    i = index[(r["site"],r["sequence"])]
    target[i,0] += float(r["denominator"])
    for j,f in enumerate(off_fields,1):
        if r.get(f): target[i,j] += float(r[f])
offset_summary = []
labels = [str(b) for b in range(17,25)] + ["all_layers", "all_layers_all_refs", "pool17_19_21_23", "pool_all8_layers"]
for label in labels:
    blocks = (("17","19","21","23") if label=="pool17_19_21_23" else tuple(str(x) for x in range(17,25)) if label=="pool_all8_layers" else (label,))
    method = "any_reference" if label=="all_layers_all_refs" else "all_offsets"
    for domain in ("Day","Night","All"):
        domains = ("Day","Night") if domain=="All" else (domain,)
        values = np.stack([sum((off_arrays[(seed,block,method,d)] for block in blocks for d in domains),
                               start=np.zeros((51,len(off_fields)+1))) for seed in SEEDS])
        assert np.all(values[:,:,0] == values[0:1,:,0])
        for j,field in enumerate(off_fields,1):
            if label.startswith("all_layers") and field.startswith(("ranked_", "identity_")): continue
            if method == "any_reference" and not field.startswith("local_"): continue
            metric = field.removesuffix("_hits").removesuffix("_sum")
            if field=="error_sum" and label=="all_layers_all_refs": continue
            if field=="error_sum": scale,u=1.,"px"
            else: scale,u=100.,"percent"
            points = scale*values[:,:,j].sum(axis=1)/values[:,:,0].sum(axis=1)
            boot = scale*interval(values[:,:,j],values[:,:,0])
            lo,hi = np.quantile(boot,(.025,.975))
            offset_summary.append(dict(block=label,domain=domain,mechanism=method,metric=metric,unit=u,
                mean=points.mean(),seed_sd=points.std(ddof=1),seed1111=points[0],seed1112=points[1],seed1113=points[2],
                ci95_low=lo,ci95_high=hi,denominator_per_seed=int(values[0,:,0].sum()),
                candidates_per_reference=(32 if label.startswith("all_layers") else 4),
                denominator_type=("GT-valid query with any valid reference" if label=="all_layers_all_refs" else
                                  "GT-valid query/reference" if label=="all_layers" else "GT-valid query/reference/layer")))

# Save per-site directions for every before/after metric.
site_rows = []
for row in recovery:
    label,domain,group,metric=row["block"],row["domain"],row["group"],row["metric"]
    blocks=BLOCKSETS[label]
    before=get("frozen",1111,blocks,group,domain)
    after=np.stack([get("adapted",s,blocks,group,domain) for s in SEEDS])
    j=field_index[metric]
    for site in sorted({x[0] for x in sequence_meta}):
        ids=[i for i,x in enumerate(sequence_meta) if x[0]==site]
        den=before[ids,0].sum()
        if not den: continue
        scale=unit(metric)[1]
        b=scale*before[ids,j].sum()/den
        av=scale*after[:,ids,j].sum(axis=1)/den
        site_rows.append(dict(block=label,domain=domain,group=group,metric=metric,unit=row["unit"],site=site,
                              frozen=b,adapted_mean=av.mean(),delta_mean=av.mean()-b))

a.output.mkdir(parents=True,exist_ok=False)
write(a.output/"attention_summary.csv",summary)
write(a.output/"attention_recovery.csv",recovery)
write(a.output/"offset_coverage.csv",offset_summary)
write(a.output/"recovery_by_site.csv",site_rows)
def choose(rows, **kwargs):
    result=[r for r in rows if all(r[k]==v for k,v in kwargs.items())]
    assert len(result)==1,(kwargs,len(result))
    return result[0]

key={
    "bootstrap": {"replicates":a.bootstrap,"rng_seed":20261103,
        "strata":"site and Day/Night sequence stratum, fixed sites; paired resampling across before/after/seeds/layers",
        "scope":"sequence sampling within the five observed sites, conditional on the three released checkpoints; not training-seed population CI",
        "sites":5,"sequences":51,"clips":204,"day_sequences":27,"night_sequences":24},
    "audits": {"same_inputs_and_gt_support":True,"passive_model_output_diff_zero":hook_audited,
               "block17_actual_qk_identical_all_seeds":True,"no_missing_samples":True},
    "frozen_tolerance":[],
    "offset_coverage":[],
    "night_recovery":[],
    "attention_concentration":[],
}
for radius in range(5):
    for domain in ("Day","Night"):
        key["frozen_tolerance"].append(choose(summary,block="pool17_19_21_23",domain=domain,group="GTValid",
                                                     model="frozen",metric=f"match_k1_r{radius}"))
    for label in ("pool17_19_21_23","all_layers","all_layers_all_refs"):
        key["offset_coverage"].append(choose(offset_summary,block=label,domain="All",metric=f"local_r{radius}"))
for label in ("pool17_19_21_23","pool19_21_23","23","25","31"):
    for metric in ("match_k1_r1","match_k1_r2","match_k1_r3","gt_mass_r1","gt_mass_r2",
                   "coarse_match1_g2","coarse_match1_g4","coarse_gt_mass_g2","coarse_gt_mass_g4"):
        key["night_recovery"].append(choose(recovery,block=label,domain="Night",group="GTValid",metric=metric))
for domain in ("Day","Night"):
    for model in ("frozen","adapted"):
        for metric in ("gt_mass_r0","gt_mass_r1","gt_mass_r2","top5pct_mass_g1","top20pct_mass_g1",
                       "top20pct_mass_g2","top20pct_mass_g4","entropy_g1","entropy_g2","entropy_g4"):
            key["attention_concentration"].append(choose(summary,block="pool17_19_21_23",domain=domain,
                                                        group="GTValid",model=model,metric=metric))
(a.output/"key_results.json").write_text(json.dumps(key,indent=2,allow_nan=False)+"\n")
table = {"a_frozen_match1": [], "b_offset_coverage": [], "c_night_match1": [],
         "night_block31_limitation": {}}
for radius in range(4):
    row = {"window_size": 2 * radius + 1}
    for domain in ("Day", "Night"):
        row[domain.lower()] = choose(summary, block="pool17_19_21_23", domain=domain,
                                    group="GTValid", model="frozen",
                                    metric=f"match_k1_r{radius}")["mean"]
    table["a_frozen_match1"].append(row)
    row = {"window_size": 2 * radius + 1}
    for column, label in (("single_layer", "pool17_19_21_23"),
                          ("all_layers", "all_layers"), ("all_references", "all_layers_all_refs")):
        row[column] = choose(offset_summary, block=label, domain="All",
                             metric=f"local_r{radius}")["mean"]
    table["b_offset_coverage"].append(row)
for block in (19, 21, 23, 25, 31):
    row = {"block": block}
    for model in ("frozen", "adapted"):
        row[model] = choose(summary, block=str(block), domain="Night",
                            group="GTValid", model=model, metric="match_k1_r1")["mean"]
    if block == 31:
        table["night_block31_limitation"] = row
    else:
        table["c_night_match1"].append(row)
(a.output / "table4.json").write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
print(json.dumps(table, indent=2))
