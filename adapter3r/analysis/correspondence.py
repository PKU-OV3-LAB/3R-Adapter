"""Camera Ready Oxford correspondence measurements through passive model hooks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from .correspondence_geometry import build_pairwise_gt_correspondences, data_path, load_view

ROOT = Path(__file__).resolve().parents[2]

BLOCKS = (17, 19, 21, 23, 25, 31)
RADII = (0, 1, 2, 3, 4)
PIXELS = (14, 28, 42, 56)
PCTS = (1, 5, 10, 20)
KS = (1, 3, 4, 8)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arrsha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def load_helpers():
    sys.path.insert(0, str(ROOT / "third_party/pi3_train"))
    sys.path.insert(0, str(ROOT))
    from pi3.models.pi3 import Pi3
    from pi3.models.pi3_Adapter3R import Pi3_Adapter3R
    from pi3.models.layers.modules import sliding_window_indices_no_repeat
    from datasets.base.base_dataset import BaseDataset
    return Pi3, Pi3_Adapter3R, sliding_window_indices_no_repeat, BaseDataset


def attention_rows(attn, gt, refs, item, block, model, seed):
    """attn is true full-key softmax, averaged over heads, patch-query/key restricted."""
    n, hw, pw, ph = (gt["meta"][k] for k in ("N", "hw", "pw", "ph"))
    gtt = torch.as_tensor(gt["gt_ref_token"])
    gtp = torch.as_tensor(gt["gt_ref_pixel"])
    valid = torch.as_tensor(gt["valid"]).bool()
    rows = []
    per_group = {}
    for group in ("GTValid", "GTInvalid"):
        per_group[group] = dict(sample_id=item["sample_id"], site=item["scene"],
            sequence=item["sequence"], domain=item["stratum"].capitalize(),
            block=block, model=model, seed=seed, group=group, denominator=0)
    for i in range(n):
        for j in refs[i].tolist():
            if i == j:
                continue
            absolute = attn[i, :, j, :]
            share = absolute.sum(-1)
            assert (share > 0).all() and torch.isfinite(share).all()
            prob = absolute / share[:, None]
            assert torch.allclose(prob.sum(-1), torch.ones(hw), atol=2e-6)
            target = gtt[i, j].clamp(min=0)
            gr, gc = target // pw, target % pw
            vr = valid[i, j]
            top = prob.topk(8, dim=-1).indices
            pred_r, pred_c = top // pw, top % pw
            # GT token discretization exactly follows the inherited fixed protocol.
            dist_inf = torch.maximum((pred_r - gr[:, None]).abs(), (pred_c - gc[:, None]).abs())
            pred_xy = torch.stack((pred_c * 14 + 7, pred_r * 14 + 7), -1)
            err = (pred_xy - gtp[i, j][:, None, :]).square().sum(-1).sqrt()
            metric = {"reference_patch_share": share}
            for k in KS:
                for radius in RADII:
                    metric[f"match_k{k}_r{radius}"] = (dist_inf[:, :k] <= radius).any(-1).float()
                metric[f"error_k{k}_px"] = err[:, :k].min(-1).values
                for px in PIXELS:
                    metric[f"pixel_hit_k{k}_{px}"] = (err[:, :k] <= px).any(-1).float()
            q = torch.arange(hw)
            dr, dc = torch.meshgrid(torch.arange(-4,5),torch.arange(-4,5),indexing="ij")
            dr,dc = dr.flatten(),dc.flatten()
            rr,cc = gr[:,None]+dr,gc[:,None]+dc
            inside=(rr>=0)&(rr<ph)&(cc>=0)&(cc<pw)
            idx=rr.clamp(0,ph-1)*pw+cc.clamp(0,pw-1)
            neighbor_mass=prob[q[:,None],idx]
            for radius in RADII:
                keep=inside & (dr.abs()<=radius) & (dc.abs()<=radius)
                mass=(neighbor_mass*keep).sum(-1)
                area=keep.sum(-1)
                metric[f"gt_mass_r{radius}"] = mass
                metric[f"uniform_gt_mass_r{radius}"] = area / hw
                metric[f"gt_mass_enrichment_r{radius}"] = mass / (area / hw)
            for grid in (1, 2, 4):
                if grid == 1:
                    coarse = prob
                    gh, gw = ph, pw
                else:
                    gh, gw = math.ceil(ph / grid), math.ceil(pw / grid)
                    image = F.pad(prob.reshape(hw, 1, ph, pw), (0, gw*grid-pw, 0, gh*grid-ph))
                    coarse = F.avg_pool2d(image, grid, grid).flatten(1) * grid**2
                # Non-overlapping cell aggregation conserves all probability mass.
                assert torch.allclose(coarse.sum(-1), torch.ones(hw), atol=3e-6)
                sorted_mass = coarse.sort(-1, descending=True).values
                entropy = -(coarse * coarse.clamp(min=1e-30).log()).sum(-1)
                metric[f"entropy_g{grid}"] = entropy / math.log(gh * gw)
                metric[f"peak_mass_g{grid}"] = sorted_mass[:, 0]
                cell = (gr // grid) * gw + gc // grid
                metric[f"coarse_match1_g{grid}"] = (coarse.argmax(-1) == cell).float()
                metric[f"coarse_gt_mass_g{grid}"] = coarse[q, cell]
                for pct in PCTS:
                    number = max(1, math.ceil(gh * gw * pct / 100))
                    metric[f"top{pct}pct_mass_g{grid}"] = sorted_mass[:, :number].sum(-1)
            metric_names = list(metric)
            packed = torch.stack(list(metric.values())).double().numpy()
            for group, mask in (("GTValid", vr), ("GTInvalid", ~vr)):
                row = per_group[group]
                row["denominator"] += int(mask.sum())
                totals = packed[:, mask.numpy()].sum(-1)
                for name, total in zip(metric_names, totals):
                    # Geometry metrics on invalid queries are undefined, not zero.
                    if group == "GTInvalid" and (name.startswith(("match_", "error_", "pixel_", "gt_", "uniform_gt_", "coarse_match", "coarse_gt"))):
                        continue
                    row[name + "_sum"] = row.get(name + "_sum", 0.0) + float(total)
    for row in per_group.values():
        rows.append(row)
    return rows


class PassiveCapture:
    def __init__(self, model, gt, refs, item, name, seed, chunk=512, export_profile=False):
        self.model, self.gt, self.refs, self.item = model, gt, refs, item
        self.name, self.seed, self.chunk = name, seed, chunk
        self.active = None
        self.qk = []
        self.rows, self.offsets, self.handles, self.qk_hashes = [], {}, [], {}
        self.offset_raw, self.weight_raw = {}, {}
        self.export_profile = export_profile
        self.rank_rows, self.rank_sums, self.raster_sums = [], [], []
        self.handles.append(model.rope.register_forward_hook(self.rope_hook))
        for b in BLOCKS:
            attn = model.decoder[b].attn
            self.handles.append(attn.register_forward_pre_hook(self.pre(b), with_kwargs=True))
            self.handles.append(attn.register_forward_hook(self.after(b), with_kwargs=True))
        if name == "adapted":
            for b, adapter in zip(model.peft_layers[1], model.dec_adapters):
                self.handles.append(adapter.offset_head.register_forward_hook(self.offset_hook(b)))
                self.handles.append(adapter.weight_head.register_forward_hook(self.weight_hook(b)))
                self.handles.append(adapter.register_forward_hook(self.adapter_hook(b), with_kwargs=True))

    def pre(self, block):
        def hook(module, args, kwargs):
            assert self.active is None
            assert kwargs.get("attn_bias") is None
            self.active = block
            self.qk = []
        return hook

    def rope_hook(self, module, args, output):
        if self.active is not None:
            self.qk.append(output.detach())

    def after(self, block):
        def hook(module, args, kwargs, output):
            started=time.time()
            assert self.active == block and len(self.qk) == 2
            q, k = self.qk
            assert q.dtype == torch.bfloat16 and q.shape == k.shape
            if block == 17:
                self.qk_hashes[str(block)] = [arrsha(q), arrsha(k)]
            n, ph, pw, hw = (self.gt["meta"][v] for v in ("N", "ph", "pw", "hw"))
            t = hw + self.model.patch_start_idx
            patch_ids = torch.cat([torch.arange(i*t+self.model.patch_start_idx, (i+1)*t,
                                                device=q.device) for i in range(n)])
            pieces = []
            # Passive readout uses actual post-RoPE Q/K from the unmodified forward.
            # Full softmax includes every view and register token; no temperature change.
            with torch.autocast("cuda", enabled=False):
                kf = k.float()
                for ids in patch_ids.split(self.chunk):
                    logits = (q[:, :, ids].float() * (q.shape[-1] ** -.5)) @ kf.transpose(-1, -2)
                    mean = logits.softmax(-1).mean(1)[0]
                    pieces.append(mean.cpu())
                del kf
            full = torch.cat(pieces).reshape(n, hw, n, t)
            patch = full[:, :, :, self.model.patch_start_idx:]
            self.rows.extend(attention_rows(patch, self.gt, self.refs, self.item, block, self.name, self.seed))
            if self.export_profile and self.name == "frozen" and block in (17, 19, 21, 23):
                rank = torch.zeros(hw, dtype=torch.float64)
                raster = torch.zeros(hw, dtype=torch.float64)
                count = 0
                valid = torch.as_tensor(self.gt["valid"]).bool()
                for i in range(n):
                    for j in self.refs[i].tolist():
                        if i == j:
                            continue
                        absolute = patch[i, :, j, :]
                        prob = absolute / absolute.sum(-1, keepdim=True)
                        prob = prob[valid[i, j]]
                        count += len(prob)
                        rank += prob.sort(-1, descending=True).values.double().sum(0)
                        raster += prob.double().sum(0)
                assert abs(rank.sum().item() - count) < max(1e-4, count * 1e-6)
                self.rank_rows.append(dict(sample_id=self.item["sample_id"], site=self.item["scene"],
                    sequence=self.item["sequence"], domain=self.item["stratum"].capitalize(),
                    block=block, model=self.name, seed=self.seed, group="GTValid", denominator=count))
                self.rank_sums.append(rank.numpy())
                self.raster_sums.append(raster.numpy())
            if self.item["sample_id"] == 0:
                print(json.dumps({"profile_model":self.name,"block":block,"readout_seconds":round(time.time()-started,3)}),flush=True)
            self.active = None
            self.qk = []
        return hook

    def offset_hook(self, block):
        def hook(module, args, output):
            self.offset_raw[block] = output.detach()
        return hook

    def weight_hook(self, block):
        def hook(module, args, output):
            self.weight_raw[block] = output.detach()
        return hook

    def adapter_hook(self, block):
        def hook(module, args, kwargs, output):
            x = args[0]
            n, hw, pw, ph = (self.gt["meta"][v] for v in ("N", "hw", "pw", "ph"))
            assert x.shape[0] == n and x.shape[1] == hw
            assert module.grid_layout == "pixel_center" and module.sampling_layout == "reference_major"
            assert module.K_sample == 4
            offset = torch.tanh(self.offset_raw.pop(block).view(n, hw, 1, 4, 2)) * module.region_scale
            weight = F.softmax(self.weight_raw.pop(block).view(n, hw, 1, 4), -1)
            base = module.create_grid(n, ph, pw, x.device).reshape(n, hw, 1, 1, 2)
            locations = (base + offset).clamp(-1, 1).detach().float().cpu().squeeze(2)
            self.offsets[block] = (locations, weight.detach().float().cpu().squeeze(2))
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def offsets_rows(offsets, gt, refs, item, seed):
    n, hw, pw, ph = (gt["meta"][k] for k in ("N", "hw", "pw", "ph"))
    gtt = torch.as_tensor(gt["gt_ref_token"])
    gtp = torch.as_tensor(gt["gt_ref_pixel"])
    valid = torch.as_tensor(gt["valid"]).bool()
    def decode(loc):
        col = ((loc[..., 0] + 1) * pw - 1) / 2
        row = ((loc[..., 1] + 1) * ph - 1) / 2
        tokens = row.round().long().clamp(0, ph-1) * pw + col.round().long().clamp(0, pw-1)
        pixels = torch.stack((col * 14 + 7, row * 14 + 7), -1)
        return tokens, pixels
    candidates = {b: decode(v[0]) for b, v in offsets.items()}
    candidates["all_layers"] = (torch.cat([v[0] for v in candidates.values()], -1),
                                torch.cat([v[1] for v in candidates.values()], -2))
    rows = []
    union_hits = {r: torch.zeros(n, hw, dtype=torch.bool) for r in RADII}
    query_valid = torch.zeros(n, hw, dtype=torch.bool)
    q = torch.arange(hw)
    identity = torch.stack((q % pw * 14 + 7, q // pw * 14 + 7), -1)
    for block, (tokens, pixels) in candidates.items():
        row = dict(sample_id=item["sample_id"], site=item["scene"], sequence=item["sequence"],
                   domain=item["stratum"].capitalize(), seed=seed, block=block,
                   mechanism="all_offsets", candidates=tokens.shape[-1], denominator=0)
        for i in range(n):
            for j in refs[i].tolist():
                if i == j:
                    continue
                m = valid[i, j]
                target = gtt[i, j].clamp(min=0)
                gr, gc = target // pw, target % pw
                distance = torch.maximum((tokens[i] // pw - gr[:, None]).abs(),
                                         (tokens[i] % pw - gc[:, None]).abs())
                error = (pixels[i] - gtp[i, j][:, None, :]).square().sum(-1).sqrt().min(-1).values
                row["denominator"] += int(m.sum())
                for radius in RADII:
                    hit = (distance <= radius).any(-1)
                    row[f"local_r{radius}_hits"] = row.get(f"local_r{radius}_hits", 0) + int((hit & m).sum())
                    if block == "all_layers":
                        union_hits[radius][i] |= hit & m
                        query_valid[i] |= m
                row["error_sum"] = row.get("error_sum", 0.) + float(error[m].sum(dtype=torch.float64))
                for px in PIXELS:
                    row[f"pixel_{px}_hits"] = row.get(f"pixel_{px}_hits", 0) + int(((error <= px) & m).sum())
                if block != "all_layers":
                    order = offsets[block][1][i].argsort(dim=-1, descending=True, stable=True)
                    ranked = distance.gather(-1, order)
                    for k in (1, 3):
                        row[f"ranked_k{k}_r1_hits"] = row.get(f"ranked_k{k}_r1_hits", 0) + int(((ranked[:, :k] <= 1).any(-1) & m).sum())
                    # Spatial priors use the identical geometry support.
                    identity_d = torch.maximum((q // pw - gr).abs(), (q % pw - gc).abs())
                    for r in (1, 2, 3, 4):
                        row[f"identity_r{r}_hits"] = row.get(f"identity_r{r}_hits", 0) + int(((identity_d <= r) & m).sum())
        rows.append(row)
    row = dict(sample_id=item["sample_id"], site=item["scene"], sequence=item["sequence"],
               domain=item["stratum"].capitalize(), seed=seed, block="all_layers_all_refs",
               mechanism="any_reference", candidates_per_reference=32, denominator=int(query_valid.sum()))
    for radius, hits in union_hits.items():
        row[f"local_r{radius}_hits"] = int((hits & query_valid).sum())
    rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--samples", type=Path, default=ROOT / "configs/oxford_correspondence_samples.json")
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(1111, 1112, 1113), required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", choices=("pilot", "full"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", action="store_true")
    p.add_argument("--verify-hooks", action="store_true")
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--export-profile", action="store_true")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    if args.gpu < 0 or args.limit < 0 or args.chunk < 1:
        p.error("gpu and limit must be nonnegative; chunk must be positive")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        p.error("shards must be positive and shard must be in [0, shards)")
    if args.export_profile and not args.base:
        p.error("--export-profile requires --base")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_LAUNCH_BLOCKING="0",
        ADAPTER3R_REGION_SCALE="0.2", ADAPTER3R_FUSION_FACTOR="0.1", ADAPTER3R_RESIDUAL_SCALE="0.3")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    selection = json.loads(args.samples.read_text())
    if sha(args.samples) != sha(ROOT / "configs/oxford_correspondence_samples.json"):
        p.error("this evaluator requires the fixed Camera Ready sample manifest")
    samples = selection["samples"]
    if args.protocol == "pilot":
        samples = [s for s in samples if s["sample_id"] in selection["pilot32_ids"]]
    assert len(samples) == (32 if args.protocol == "pilot" else 204)
    if args.shards > len(samples):
        p.error("number of shards must not exceed the selected sample count")
    samples = samples[args.shard::args.shards]
    if args.limit:
        samples=samples[:args.limit]
    assert all(s["stratum"] in ("day", "night") for s in samples)
    release = json.loads((ROOT / "results/release_weights.json").read_text())
    ck = next(c for c in release["checkpoints"] if c["backbone"] == "pi3" and c["seed"] == args.seed)
    checkpoint = args.weights_root / ck["file"]
    assert sha(checkpoint) == ck["sha256"]
    for item in samples:
        assert len(item["frames"]) == 4
        for frame in item["frames"]:
            for field in ("img_path", "depth_path"):
                data_path(args.data_root, frame[field])
    from ..runners.runtime import verify_bundle
    verify_bundle(ROOT / "third_party/pi3_train")
    if args.output.exists():
        p.error("output must be a fresh directory")
    if args.preflight:
        print(json.dumps({"checkpoint_sha256": ck["sha256"], "sample_count": len(samples),
                          "samples_sha256": sha(args.samples), "protocol": args.protocol}))
        return
    Pi3, Adapted, refs_fn, Dataset = load_helpers()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("measurement requires a CUDA device with BF16 support")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    manifest = {
        "protocol": args.protocol, "seed": args.seed, "gpu": args.gpu,
        "checkpoint": str(checkpoint), "checkpoint_sha256": ck["sha256"],
        "samples_sha256": sha(args.samples), "sample_ids": [s["sample_id"] for s in samples],
        "sample_count": len(samples), "code_root": str(ROOT), "data_root": str(args.data_root),
        "shard": args.shard, "shards": args.shards,
        "export_profile": args.export_profile, "readout_chunk": args.chunk,
        "code_sha256": {str(x): sha(x) for x in (
            Path(__file__), Path(__file__).with_name("correspondence_geometry.py"),
            ROOT / "adapter3r/core.py",
            ROOT / "third_party/pi3_train/pi3/models/pi3_Adapter3R.py",
            ROOT / "third_party/pi3_train/pi3/models/pi3.py",
            ROOT / "third_party/pi3_train/pi3/models/layers/modules.py",
            ROOT / "third_party/pi3_train/pi3/models/layers/attention.py",
            ROOT / "third_party/pi3_train/pi3/utils/geometry.py",
            ROOT / "third_party/pi3_train/pi3/utils/cropping.py",
            ROOT / "third_party/pi3_train/datasets/base/base_dataset.py")},
        "blocks": BLOCKS, "adapter_blocks": list(range(17, 25)), "patch_size": 14,
        "matching_radii_tokens": RADII, "pixel_tolerances": PIXELS,
        "coarse_nonoverlapping_cell_factors": [1, 2, 4],
        "attention_definition": "actual BF16 post-RoPE Q/K; FP32 scaled dot products and full-key softmax; mean of all heads; conditional normalization per query/reference over patch keys",
        "concentration": "top percent of reference cells; coarse cells sum fine probabilities; unchanged temperature",
        "all_offsets": "any of four actual clamped positions per query/reference/layer; all_layers is any of 32 positions; any_reference has query-level denominator",
        "gt": "independent fixed dataset depth/camera and inherited cropping/discretization",
        "started_at": time.time(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" in state: state = state["model"]
    elif "state_dict" in state: state = state["state_dict"]
    adp = Adapted(ckpt=None, peft_layers=[[], list(range(17, 25))])
    adp.load_state_dict(state, strict=True)
    models = []
    if args.base:
        base = Pi3()
        base.load_state_dict({k: v for k, v in state.items() if not k.startswith("dec_adapters.")}, strict=True)
        models.append(("frozen", base.cuda().eval().requires_grad_(False)))
    models.append(("adapted", adp.cuda().eval().requires_grad_(False)))
    del state
    ds = Dataset(mode="test", resolution=[(518, 518)], frame_num=4)
    refs = refs_fn(4, width=2, device="cpu")
    all_attn, all_offsets, checks = [], [], []
    rank_rows, rank_sums, raster_sums = [], [], []
    for pos, item in enumerate(samples):
        start = time.time()
        rng = np.random.default_rng(item["crop_seed"])
        views = [load_view(ds, args.data_root, fr, (518, 518), rng) for fr in item["frames"]]
        gt = build_pairwise_gt_correspondences(views, patch_size=14, device="cpu")
        image = torch.stack([torch.from_numpy(v["img"]).permute(2, 0, 1).float()/255 for v in views])[None].cuda()
        model_hashes = {}
        for name, model in models:
            expected = None
            if pos == 0 and args.verify_hooks:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model(image)
            cap = PassiveCapture(model, gt, refs, item, name, args.seed, chunk=args.chunk,
                                 export_profile=args.export_profile)
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(image)
            finally:
                cap.close()
            if expected is not None:
                diffs = {k: float((v - expected[k]).abs().max()) for k, v in output.items()
                         if isinstance(v, torch.Tensor)}
                assert all(v == 0 for v in diffs.values()), diffs
                checks.append({"model": name, "passive_hook_output_max_abs_diff": diffs})
            all_attn.extend(cap.rows)
            rank_rows.extend(cap.rank_rows)
            rank_sums.extend(cap.rank_sums)
            raster_sums.extend(cap.raster_sums)
            if name == "adapted":
                assert len(cap.offsets) == 8
                all_offsets.extend(offsets_rows(cap.offsets, gt, refs, item, args.seed))
            model_hashes[name] = cap.qk_hashes["17"]
            del cap, output, expected
        if args.base:
            assert model_hashes["frozen"] == model_hashes["adapted"], "first adapter block Q/K must be identical"
        checks.append({"sample_id": item["sample_id"], "block17_actual_qk_sha256": model_hashes})
        del image
        torch.cuda.empty_cache()
        write_csv(args.output / "attention_per_sample.csv", all_attn)
        write_csv(args.output / "offset_per_sample.csv", all_offsets)
        if rank_rows:
            write_csv(args.output / "rank_metadata.csv", rank_rows)
            np.savez_compressed(args.output / "full_token_sums.npz",
                                rank_sum=np.stack(rank_sums), raster_sum=np.stack(raster_sums))
        (args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps({"progress": pos+1, "total": len(samples), "sample": item["sample_id"],
                          "domain": item["stratum"], "site": item["scene"],
                          "seconds": round(time.time()-start, 2)}), flush=True)
    for path, digest in manifest["code_sha256"].items():
        assert sha(path) == digest, "source changed during measurement"
    manifest["completed_at"] = time.time()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "complete").write_text("all selected samples completed; no silent skips\n")


if __name__ == "__main__":
    main()
