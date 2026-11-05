# Adaptation and evaluation tasks

Run from the repository root after installing this package and the relevant
backbone's dependencies. All required backbone source is in `third_party/`;
no external checkout or Git submodule initialization is needed. `--repo` can
select an alternative source tree. Model weights and data are explicit inputs.
Outputs must be fresh directories.

## Low-light reconstruction and camera pose estimation

```bash
bash experiments/run_table1_pi3.sh \
  --release-config none --checkpoint YOUR_PI3_ADAPTER_WEIGHTS \
  --pretrained PI3_BASE_DIRECTORY \
  --grid-layout pixel_center --sampling-layout reference_major --residual-scale 0.3 \
  --data-root OXFORD_DATA --gpu 1 --output-root NEW_PI3_RESULTS --task all
bash experiments/run_table1_cut3r.sh \
  --checkpoint CUT3R_ADAPTER_WEIGHTS --data-root OXFORD_DATA \
  --grid-layout pixel_center --sampling-layout reference_major \
  --gpu 1 --output-root NEW_CUT3R_RESULTS --task all
```

Pi3 uses decoder layers 17–24. Both methods include scene reconstruction and
relative pose. Oxford uses within-scene day/night inputs, stride 2, with CUT3R
limited to 20 frames. Pi3's default `configs/release/pi3_3seed.json` validates
historical release checkpoint hashes; use `--release-config none` for your own
trained checkpoint. Run seeds 1111, 1112 and 1113 in separate output directories
for the three-seed mean. Historical Pi3 checkpoints require
`--release-config none --checkpoint HISTORICAL_WEIGHTS --grid-layout legacy
--sampling-layout legacy`. CUT3R retains its historical defaults.
Both runners accept `--preflight`; Pi3 resolves the full Hydra configuration.

## Self-supervised test-time adaptation

```bash
bash experiments/run_table2_test3r_7scenes.sh \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root SEVENSCENES_DATA \
  --gpu 1 --output-dir NEW_7SCENES_RESULTS
bash experiments/run_table2_test3r_nrgbd.sh \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root NRGBD_DATA \
  --gpu 1 --output-dir NEW_NRGBD_RESULTS
```

Each sequence initializes from base DUSt3R weights. 7-Scenes covers all 18
sequences with stride 200, NRGBD all nine with stride 500. Online optimization
uses two epochs, accumulation 1, reduction dimension 32, encoder layers 0–23
and odd decoder layers 1–11. Learning rates are `1e-5` and `9e-5`.
The historical TTA loss uses `with_conf=False`. Both runners accept
`--epochs`, `--lr`, `--accum-iter` and `--preflight`.

## Joint human-scene reconstruction

```bash
bash experiments/run_table3_human3r_bedlam.sh \
  --weights-root RELEASE_WEIGHTS --seed 0 --data-root PROCESSED_BEDLAM \
  --model-assets HUMAN3R_MODEL_ASSETS \
  --gpu 1 --output-dir NEW_BEDLAM_RESULTS
```

The processed data root contains `Test/SEQUENCE/` with RGB, camera, depth,
segmentation and SMPL annotations. The model asset root has `smpl/`, `smplx/`,
`smpl_mean_params.npz` and the separately licensed model files expected by
Human3R. Source code for local DINOv2 and Multi-HMR blocks is included.

Metrics include PA-PVE, Metric-PVE, PA-MPJPE, Metric-MPJPE, scene
Accuracy/Completeness/Chamfer and depth. Scene cutoff is 15m, including human
regions; depth `max_depth=16` is separate. Frame subsample is 25.
Use `--max-sequences 32` for a pilot or `--preflight` to check dependencies.
The default `configs/release/human3r_3seed.json` uses residual scale 0.5 and
corrected layouts. Run seeds 0, 1 and 2 separately for the three-seed mean.
Use `--release-config none` with explicit layouts/scales for historical weights.

## Adapter training from official base weights

Official downloads and complete Pi3/CUT3R adaptation commands are in the
[repository README](../README.md#data-and-pretrained-weights). For the
commands below, set `OXFORD_DATA` to the absolute processed dataset path.

```bash
python experiments/train_adapter.py --backbone pi3 \
  --pretrained checkpoints/Pi3/model.safetensors \
  --seed 1111 --gpu 1 --output-dir NEW_TRAINING \
  "train_dataset.OxfordDayAndNight.data_root=$OXFORD_DATA" \
  "test_dataset.OxfordDayAndNight.data_root=$OXFORD_DATA"
python experiments/train_adapter.py --backbone cut3r \
  --pretrained checkpoints/CUT3R/cut3r_512_dpt_4_64.pth \
  --grid-layout pixel_center --sampling-layout reference_major \
  --seed 1111 --gpu 1 --output-dir NEW_CUT_TRAINING \
  --config-name cut3r-3RAdapter \
  "train_dataset=\"4000 @ OxfordDayAndNight(allow_repeat=True, split='train', ROOT='$OXFORD_DATA', aug_crop=16, resolution=[(512,384),(512,336),(512,288),(512,256),(512,208),(512,144),(384,512),(336,512),(288,512),(256,512)], transform=SeqColorJitter, num_views=4, n_corres=0)\"" \
  "test_dataset=\"1000 @ OxfordDayAndNight(split='test', ROOT='$OXFORD_DATA', resolution=224, num_views=4, seed=42, n_corres=0)\""
python experiments/train_adapter.py --backbone human3r \
  --pretrained CUT3R_BASE_WEIGHTS --pretrained-human MULTIHMR_WEIGHTS \
  --model-assets HUMAN3R_MODEL_ASSETS --seed 0 --gpu 1 --output-dir NEW_HUMAN_TRAINING \
  "dataset28=BEDLAM_Multi(allow_repeat=True, split='train', ROOT='PROCESSED_BEDLAM', aug_crop=16, resolution=[(512,384),(512,336),(512,288),(512,256),(512,208),(512,144),(384,512),(336,512),(288,512),(256,512)], transform=SeqColorJitter, num_views=4, n_corres=0)" \
  "test_dataset=1000 @ BEDLAM_Multi(split='test', ROOT='PROCESSED_BEDLAM', resolution=(512,288), num_views=4, seed=42, n_corres=0)"
```

Supply explicit dataset overrides. Pi3 and Human3R use the release preset's
model expression, budget and selected seed. Conflicting fixed overrides fail.
For custom training use `--release-config none` and explicit settings.
CUT3R/Human3R dataset expressions live in their bundled configs. Human3R uses
adapter layers 8–11. Training starts from pretrained base weights with resume
disabled. Use `--preflight` to inspect the command before a full run.

## Cross-view correspondence and attention distributions

See [correspondence_README.md](correspondence_README.md) for the fixed Oxford
selection, passive attention readout, all-offset coverage and paired
three-seed summary. Numerical records are included for CPU recomputation.

## Dependencies and evidence

Install requirements from the chosen bundled backbone. CroCo CUDA RoPE source
is under `croco/models/curope` for Test3R, or `src/croco/models/curope` for
CUT3R/Human3R. Its `setup.py` compiles an optional local extension; a PyTorch
fallback is available.

Source revision and file hashes are in `third_party/MANIFEST.json` and each
backbone's `SOURCE.json`. Runtime manifests record these sources and checkpoint
hashes even without a Git checkout. `collect_results.py` collects JSON/CSV
metrics and can require completed fresh runs. Included paper values and
historical reruns remain separate from newly generated results.
