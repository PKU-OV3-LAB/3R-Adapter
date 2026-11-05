"""Recompute and render the full-reference attention profile with compressed axes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def recompute(run):
    if not (run / "complete").is_file():
        raise ValueError("profile run is incomplete")
    meta = rows(run / "rank_metadata.csv")
    keys = [(r["sample_id"], r["block"]) for r in meta]
    if len(keys) != 816 or len(set(keys)) != 816 or {r["block"] for r in meta} != {"17", "19", "21", "23"}:
        raise ValueError("require all 204 inputs and four measured blocks exactly once")
    if {r["sample_id"] for r in meta} != {str(i) for i in range(204)}:
        raise ValueError("profile does not cover the fixed Camera Ready sample IDs")
    if any(r["model"] != "frozen" or r["group"] != "GTValid" for r in meta):
        raise ValueError("profile must use frozen attention on valid GT support")
    with np.load(run / "full_token_sums.npz") as data:
        rank, raster = data["rank_sum"], data["raster_sum"]
    if rank.shape != (816, 1369) or raster.shape != rank.shape:
        raise ValueError("full-reference arrays must contain 1369 tokens per measurement")
    den = np.array([int(r["denominator"]) for r in meta])
    if not np.allclose(rank.sum(-1), den, atol=1e-4, rtol=1e-6):
        raise ValueError("rank sums do not conserve probability")
    att = rows(run / "attention_per_sample.csv")
    att = [r for r in att if r["model"] == "frozen" and r["group"] == "GTValid" and r["block"] in {"17","19","21","23"}]
    day = [r for r in att if r["domain"] == "Day"]
    anchor = sum(float(r["gt_mass_r0_sum"]) for r in day) / sum(int(r["denominator"]) for r in day)
    result = {"attention_rank": np.arange(1, 1370)}
    for domain in ("Day", "Night"):
        mask = np.array([r["domain"] == domain for r in meta])
        support = den[mask].sum()
        if support <= 0:
            raise ValueError(f"missing {domain} support")
        for label, data in (("rank", rank), ("raster", raster)):
            prob = data[mask].sum(0) / support
            result[f"{domain.lower()}_{label}_probability"] = prob
            result[f"{domain.lower()}_{label}_day_gt_percent"] = prob / anchor * 100
    return result, anchor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path)
    p.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[2] /
                   "results/correspondence/full_token_attention_profiles.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--day-color", default="#E6612D")
    p.add_argument("--night-color", default="#2166AC")
    a = p.parse_args()
    if a.output.exists():
        p.error("output must be a fresh directory")
    if a.run:
        data, anchor = recompute(a.run)
    else:
        records = rows(a.csv)
        data = {k: np.array([float(r[k]) for r in records]) for k in records[0]}
        anchor = data["day_rank_probability"][0] / data["day_rank_day_gt_percent"][0] * 100
    x = data["attention_rank"]
    if not np.array_equal(x, np.arange(1, 1370)):
        p.error("profile must cover every reference-token rank from 1 to 1369")
    a.output.mkdir(parents=True)
    if a.run:
        with (a.output / "full_token_attention_profiles.csv").open("w") as stream:
            writer = csv.DictWriter(stream, list(data))
            writer.writeheader()
            writer.writerows({k: v[i] for k,v in data.items()} for i in range(1369))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(7.6,3.9), sharex="col", sharey="row",
                             gridspec_kw={"width_ratios":[4.8,1.5], "height_ratios":[4,1],
                                          "wspace":.045,"hspace":.06})
    colors = {"Day": a.day_color, "Night": a.night_color}
    fine = np.linspace(1, 1369, 20000)
    for domain in ("Day", "Night"):
        y = data[f"{domain.lower()}_rank_day_gt_percent"]
        if not np.isfinite(y).all() or (y <= 0).any() or (np.diff(y) > 1e-10).any():
            p.error("rank profile must be positive, finite and nonincreasing")
        smooth = PchipInterpolator(x, y)(fine)
        for ax in axes.flat:
            ax.plot(fine, smooth, color=colors[domain], label=domain, lw=1.8)
    for row in range(2):
        for col in range(2):
            ax = axes[row,col]
            ax.set_yscale("log")
            ax.set_xlim((1,128) if col == 0 else (128,1369))
            ax.set_ylim((10,300) if row == 0 else (.8,10))
            ax.spines["right" if col == 0 else "left"].set_visible(False)
            ax.spines["bottom" if row == 0 else "top"].set_visible(False)
            ax.grid(alpha=.16, lw=.5)
            ax.tick_params(axis="both", which="both", labelbottom=row==1, labelleft=col==0,
                           left=col==0, bottom=row==1)
    axes[0,0].set_yticks([10,30,100,300], labels=["10","30","100","300"])
    axes[1,0].set_yticks([1,3], labels=["1","3"])
    axes[1,0].set_xticks([1,32,64,96,128])
    axes[1,1].set_xticks([500,1000,1369])
    axes[0,0].axhline(100, color=".4", ls=":", lw=.8)
    axes[0,0].legend(frameon=False, loc="upper right")
    # Marks denote a change in display scale; no measured ranks/ranges are dropped.
    d=.012
    for row in range(2):
        for col in range(2):
            ax=axes[row,col]
            boundary=1 if col==0 else 0
            for y in (0,1):
                ax.plot([boundary-d,boundary+d],[y-d,y+d], transform=ax.transAxes,
                        color=".3", clip_on=False, lw=.8)
    fig.supxlabel("Attention rank over all 1,369 reference tokens", y=.025)
    fig.supylabel("Attention / Day frozen GT exact mass (%)", x=.005)
    fig.subplots_adjust(left=.105,right=.98,top=.96,bottom=.17)
    for suffix in ("pdf","svg","png"):
        fig.savefig(a.output/f"full_token_attention_rank_order_folded.{suffix}",dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    (a.output/"profile_definition.json").write_text(json.dumps({
        "normalization_probability":anchor,"tokens":1369,"x_segments":[[1,128],[128,1369]],
        "y_segments":[[.8,10],[10,300]],"interpolation":"PCHIP through all measured ranks",
        "omitted_ranks":[],"colors":colors},indent=2)+"\n")


if __name__ == "__main__":
    main()
