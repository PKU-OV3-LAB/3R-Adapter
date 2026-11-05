# Cross-view correspondence and attention

Run from the source repository after installing Pi3's dependencies. This
evaluation uses the recorded Pi3 checkpoints, corrected sampling, decoder
adapter indices 17–24, and scales 0.2 / 0.1 / 0.3.
On RTX 50-series GPUs, use the root README's CUDA 12.8 installation profile
instead of the historical Torch 2.5.1 requirements.

```bash
python -m adapter3r.analysis.correspondence \
  --data-root OXFORD_DATA --weights-root RELEASE_WEIGHTS \
  --seed 1111 --gpu 0 --protocol pilot --base --verify-hooks --output NEW_PILOT
python -m adapter3r.analysis.correspondence \
  --data-root OXFORD_DATA --weights-root RELEASE_WEIGHTS \
  --seed 1111 --gpu 0 --protocol full --base --verify-hooks --output NEW_RUNS/full1111
python -m adapter3r.analysis.correspondence \
  --data-root OXFORD_DATA --weights-root RELEASE_WEIGHTS \
  --seed 1112 --gpu 1 --protocol full --verify-hooks --output NEW_RUNS/full1112
python -m adapter3r.analysis.correspondence \
  --data-root OXFORD_DATA --weights-root RELEASE_WEIGHTS \
  --seed 1113 --gpu 2 --protocol full --verify-hooks --output NEW_RUNS/full1113
python -m adapter3r.analysis.correspondence_summary --data NEW_RUNS --output NEW_SUMMARY
```

`--preflight` checks selected frame paths, source inventories and checkpoint
SHA256 without running inference. `--limit` supports small smoke runs; such
runs cannot be used by the full 204-input summary. Outputs must be new
directories. A CUDA GPU with BF16 support is required. `--chunk` controls the
FP32 attention query chunk size; the validated setting is 512.

For parallel inference, launch independent runs with `--shards N --shard I`
for every `I` from 0 to `N-1`, then merge them before full-run aggregation:

```bash
python -m adapter3r.analysis.correspondence_merge \
  --inputs RUN_SHARD0 RUN_SHARD1 RUN_SHARD2 RUN_SHARD3 --output NEW_RUNS/full1111
```

The fixed manifest contains 204 four-view inputs from 51 sequences at five
sites, with 27 day and 24 night sequences. Frame paths begin with `aria/` and
are relative to `--data-root`. Camera matrices and crop seeds preserve the
exact measurement selection. Original dataset metadata hashes are provenance
only because machine-specific path strings can change their raw bytes.
No images, depth files or model weights are included.

Geometry comes exclusively from dataset depth and cameras. Oxford depth is
converted from millimeters and truncated above 200m before cropping. This is
an Oxford input setting, independent of the 15m BEDLAM scene cutoff.
The inherited projection, `align_corners=False` depth sampling and token
rounding convention are retained to reproduce the reported measurement.
The visibility mask requires positive source depth, an interior projection
and less than 5% relative depth inconsistency.

Attention is read passively from actual BF16 post-RoPE Q/K. The readout uses
FP32 scaled logits, the full key softmax including all views and special
tokens, averages heads, then normalizes patch keys within each reference.
Model forward returns are untouched. `--verify-hooks` verifies exact output
equality on the first selected input. Block 17 Q/K hashes provide a negative
control for every input, since that block precedes the first adapter update.
The frozen model uses the checkpoint's frozen backbone tensors.

Table 4(a) pools valid pair counts over blocks 17/19/21/23. Match@1 uses
Chebyshev distance at radius `(w-1)/2` in the 37×37 patch grid. Window size is
a geometric tolerance, with the original 14-pixel model patches unchanged.

Table 4(b)'s single-layer result pools blocks 17/19/21/23 with four candidates
per query/reference/layer. All-layers uses the union of all eight adapter
layers, with 32 candidates per query/reference. All-references counts each
query once if any valid reference has a hit, with up to 64 candidates.
These are coverage statistics with different budgets and denominators.
The optional ranked Top-1/3 diagnostics sort only by learned sampling weights.

Table 4(c) uses paired valid support and nighttime Match@1 at `w=3`, averaged
over seeds 1111/1112/1113. Decoder indices are zero based. Block 31 is also
reported as a limitation: its improvement is not guaranteed.
GT mass and entropy remain supplementary; adaptation need not increase
overall attention concentration.

The summary first sums hits and support within each seed, then averages seed
percentages. It includes seed SD and a paired bootstrap over sequences
stratified by site and domain. The interval is conditional on five observed
sites and three checkpoints. Paper-facing values use means without ±.
`results/correspondence/` contains the original numerical CSVs, sanitized
provenance manifests and frozen-backbone hash evidence. Recompute the paper
table from these records without a GPU:

```bash
python -m adapter3r.analysis.correspondence_summary \
  --data results/correspondence --output NEW_RECORDED_SUMMARY
```

Full portable runs for all three seeds have been compared against every
original CSV value and actual block17 Q/K hash. All values match exactly.
`results/correspondence/port_validation.json` records the validation scope and
runtime versions. This does not validate training with the corrected Oxford
loader or full Tables 1–3 evaluation through the release entrypoints.

Add `--export-profile` to the frozen/adapted run with `--base` to export the
full reference-image distribution at blocks 17/19/21/23. Every query/reference
distribution is sorted independently before averaging, over all 1,369 tokens.
The normalized curves share the Day frozen GT exact-token mean as denominator.
The profile tool renders PDF, SVG and PNG with compressed axes, preserving
every measured rank and using PCHIP interpolation. SciPy and Matplotlib are
required for rendering.

```bash
python -m adapter3r.analysis.correspondence_profile --run NEW_RUNS/full1111 --output NEW_PROFILE
python -m adapter3r.analysis.correspondence_profile --output NEW_RECORDED_PROFILE
```

The second command uses the included original profile CSV. Day is orange and
Night is blue by default; `--day-color` and `--night-color` can set the legend
colors without changing numerical values.
