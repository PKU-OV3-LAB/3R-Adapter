# November Oxford retraining and protocol comparisons

These numerical records describe fresh Pi3 and CUT3R adapter runs for seeds
1111, 1112 and 1113. Each model started from base pretrained weights with
`pixel_center` and `reference_major` sampling. All 73 training sequences and
51 validation sequences were covered; frozen backbone tensors stayed
unchanged. Each evaluation uses 51 test sequences and 20 frames per sequence.

| Training setting | Pi3 | CUT3R |
|---|---|---|
| Epochs / updates per epoch | 20 / 800 | 20 / 500 |
| Learning rate | 1e-4 | 2e-3 |
| Adapter layers | decoder 17–24 | encoder 20–23, decoder 8–11 |
| Residual scale | 0.3 | encoder 0.3, decoder 0.5 |

Pi3's sequence-local positions are mapped through the selected sequence's
global frame IDs. CUT3R's original training loader already used that mapping.

`author_methods.json` contains the two author-method means and sample standard
deviations. `full_image_per_sequence.csv` contains the six runs' numerical
records, without machine paths or logs.

The full-image scene protocol scores valid GT regions on a 518 grid, uses
Umeyama coarse alignment and ICP 5m, and retains model-native 512 or 518 inputs.
This is a project Oxford extension of the generic Pi3 evaluation, not an
official Pi3 Oxford protocol. The bundled CUT3R scene runner instead preserves
the original center-224 entry.

`cut3r_center_per_sequence.csv` and `cut3r_center_summary.csv` compare the same
three new CUT3R checkpoints under:

| Protocol | Accuracy | Completeness | Chamfer | NC |
|---|---:|---:|---:|---:|
| Full image, Umeyama + ICP 5m | 0.224 ± 0.001 | 0.518 ± 0.021 | 0.742 ± 0.020 | 0.5310 ± 0.0007 |
| Center 224@518, Umeyama + ICP 5m | 0.198 ± 0.001 | 0.350 ± 0.008 | 0.548 ± 0.009 | 0.5565 ± 0.0004 |
| Original center 224@512 entry, ScaleShiftInv + ICP 0.1m | 0.341 ± 0.009 | 0.446 ± 0.017 | 0.788 ± 0.025 | 0.5432 ± 0.0002 |

Distances are meters; NC is dimensionless. Values are means ± sample standard
deviations over three independent seeds (`ddof=1`). Each seed first averages
its 51 sequences equally.

The center@518 comparison reuses the same predictions and refits alignment
on the selected region. The original center@512 entry also changes
preprocessing, alignment and the ICP threshold. Its difference cannot be
attributed to cropping alone. Neither center result reproduces the historical
paper checkpoint; paper values remain separate in `results/public_results.json`.
