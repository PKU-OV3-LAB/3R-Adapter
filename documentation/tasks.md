# Task setup

Install with `python -m pip install -e .` from the cloned repository.
Install the selected backbone's requirements in a separate environment.
The task commands are in the [README](../README.md); append `--help` to list
options and `--preflight` to inspect configuration before a full run.
Output directories must be fresh.

## Oxford Day&Night

Use the processed Oxford data expected by the bundled loaders, including
`aria/` frame paths, depth and camera annotations. Pass its root with
`--data-root`. Pi3 training automatically sets both train and validation
paths. CUT3R training replaces the dataset paths while retaining its
multiresolution training and center-224 validation configurations.

Pi3 adapters use decoder layers 17–24, residual scale 0.3 and 20 epochs of
800 updates. CUT3R uses encoder layers 20–23 with scale 0.3 and decoder layers
8–11 with scale 0.5; its preset trains for 20 epochs with 4,000 samples per
epoch, batch size 8 and learning rate `2e-3`.

The public evaluation command uses corrected sampling for newly trained
checkpoints. It accepts your checkpoint directly. For historical checkpoints,
explicitly pass `--grid-layout legacy --sampling-layout legacy`.
The separate `--release-config` option selects a recorded checkpoint preset
and verifies its hash. [Checkpoint inventory](weights.md) describes these
records; none of those files are publicly downloadable yet.

Pi3 evaluation requires the original base model directory with
`model.safetensors` and `config.json`. Reconstruction and relative pose run
together by default; `--task mv` or `--task relpose` selects one.
The bundled CUT3R scene evaluator uses center-224 scoring, ScaleShiftInv and
ICP 0.1 m. November full-image results in `results/oxford_20261105` use a
separate protocol and must not be attributed to that evaluator.

## 7-Scenes and NRGBD

Prepare data in the layout expected by the bundled Test3R loaders.
`python -m adapter3r adapt --dataset 7scenes` covers all 18 sequences with
stride 200; `--dataset nrgbd` covers nine sequences with stride 500.
Each sequence initializes from base DUSt3R weights.

Defaults are two optimization epochs, accumulation 1, reduction dimension 32,
encoder layers 0–23 and odd decoder layers 1–11. Learning rates are `1e-5`
for 7-Scenes and `9e-5` for NRGBD. The loss uses `with_conf=False`.
`--epochs`, `--lr` and `--accum-iter` override optimization controls.

## BEDLAM

The processed data root contains `Test/SEQUENCE/` with RGB, camera, depth,
segmentation and SMPL annotations, plus the training split.
The Human3R asset root contains `smpl/`, `smplx/`, `smpl_mean_params.npz` and
the model files required by the backbone.

BEDLAM uses the CUT3R scene backbone with Human3R's joint reconstruction
implementation and Multi-HMR human components. The `--backbone human3r`
selector identifies this implementation, not the source of pretrained weights.
Training initializes from CUT3R base weights and Multi-HMR weights, rather
than a trained Human3R checkpoint, and requires `--model-assets`.
`--data-root` configures both BEDLAM training and validation paths.
The preset uses decoder adapters 8–11, residual scale 0.5 and 40 epochs.
Evaluation includes human, scene and depth metrics, with frame subsample 25.
Scene cutoff is 15 m; depth `max_depth=16` is independently configured.
`--max-sequences 32` selects a pilot. Visualization is enabled with `--visualize`.

## Configuration and source

New training uses `pixel_center` and `reference_major` sampling and starts
from base weights with resume disabled. Model settings and budgets live in
`configs/release/` and the bundled backbone configurations. Explicit Hydra
overrides remain available after the regular command options.
Use `--release-config none` for custom training settings.

`--repo` selects an alternative upstream tree. The default uses bundled
source whose revisions and hashes are in `third_party/MANIFEST.json` and
each backbone's `SOURCE.json`.
Run manifests record the source, checkpoint and effective configuration.
The adapter core and upstream bytes are preserved during runner consolidation.
Existing tests and configuration checks do not establish full CUDA training
or a clean installation of every backbone dependency set.

CroCo's optional CUDA RoPE extension is under `croco/models/curope` for
Test3R and `src/croco/models/curope` for CUT3R/Human3R.
