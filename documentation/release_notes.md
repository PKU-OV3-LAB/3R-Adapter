# Release history

## v0.1.1

Training, evaluation and test-time adaptation now use one `adapter3r` package
and the `train`, `evaluate` and `adapt` commands. Training accepts `--data-root`
for portable Oxford/BEDLAM configuration. Older entrypoints forward to the
same package implementations. Process documentation is under `documentation/`.

The existing 34 tests, command help, actual Hydra configuration composition
and package runtime smoke checks passed. The shared core and five bundled
backbone source inventories are unchanged. These checks do not constitute
full CUDA training or evaluation.

Adapter checkpoints are still unpublished. The README links the official
Pi3/CUT3R base weights and provides adaptation commands.

## v0.1.0

The README follows the lab's GAPrompt++ repository format. The current project
page and supplied two-minute video are hosted as Release assets and deployed
to GitHub Pages. Website assets never enter Git history, so ordinary code
clones do not download them.

The repository adds the final Oxford correspondence protocol, its fixed
dataset-relative input manifest, passive attention capture, offset coverage,
three-seed aggregation, full-token distribution export and original numerical
evidence. Table 4 means can be recomputed from the included CSVs.

Release preparation also fixes missing install dependencies, a Git ignore
rule that excluded bundled dataset loaders, CUT3R historical decoder residual
scale, relative Oxford pose image paths, pose failures being labeled complete,
and BEDLAM launch paths and pilot sequence selection. Metric-only BEDLAM runs
disable optional visualization by default.

An additional Oxford **training-loader indexing defect** was found in both
the prepared release and the historical training source. Sequence-local frame
positions were incorrectly used as indices into the global concatenated frame
array. The public training loader now maps those positions to the intended
sequence's global IDs, including after short sequences are excluded.

Pi3 and CUT3R have now each completed three fresh training runs from base
weights with corrected `pixel_center` and `reference_major` sampling. Each run
evaluated all 51 Oxford test sequences. The frame-index correction applies to
Pi3; CUT3R's training loader already used the correct global-ID mapping.
Historical checkpoint inventories remain unchanged. November numerical
records are separately included in `results/oxford_20261105`.

The new full-image evaluation uses Umeyama and ICP 5m, a project Oxford
extension of Pi3's generic evaluation code. Pi3 upstream does not define an
official Oxford protocol. The bundled CUT3R scene entry retains center-224
scoring, ScaleShiftInv and ICP 0.1m. New full-image metrics must not be presented
as results of that entry. The November center-224 comparisons quantify these
protocol differences and do not reproduce the historical paper checkpoints.

Table 4 evaluation uses an independent, fixed input manifest and dataset GT
geometry, so it does not use the training loader's frame selection. The
correspondence measurement port is validated separately against its original
records. All three seeds have now completed full fixed-input runs: every value
in 9,792 attention rows and 6,120 offset rows matches the original records
exactly, and all 204 actual block17 Q/K hashes per seed match. Passive hooks
leave model outputs unchanged on each shard's first input. All 1,369 raw
rank/raster profile probabilities also match exactly; normalized values differ
by at most 5.69e-14 from floating-point summation order.
`results/correspondence/port_validation.json` records scope and provenance,
including the two input guards added after seed1111 ran.

Tables 1–3 paper values and corrected-sampling reruns remain separate
in the result manifests, including the numerical BEDLAM scene differences.

The Pi3 CUDA 12.8 installation profile records the Torch 2.9.0 environment
used on RTX 5090. Fresh installation of all backbone dependencies remains
unverified after isolated server environments could not reach PyPI.

Model weights are distributed separately; their download endpoints remain
unavailable. The original code has no repository-wide open-source license
yet, as stated in `../LICENSE_STATUS.md`. Third-party licenses are retained.
