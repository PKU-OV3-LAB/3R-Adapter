# 3R-Adapter: Retrieval, Rewiring, and Refinement for Efficient Adaptation of 3D Reconstruction Model

<div align="center">
  <a href="https://azx030512.github.io/">Zixiang Ai</a><sup>1</sup>&emsp;
  Zhenyu Cui<sup>2</sup>&emsp;
  Xu Zou<sup>3</sup>&emsp;
  <a href="https://zhoujiahuan1991.github.io/">Jiahuan Zhou</a><sup>1,†</sup>
</div>

<div align="center">
  <sup>1</sup>Peking University&emsp;
  <sup>2</sup>Tsinghua University&emsp;
  <sup>3</sup>Huazhong University of Science and Technology
</div>

<p align="center"><sup>†</sup>Corresponding author</p>

<p align="center">
  <strong>NeurIPS 2026 Spotlight</strong><br>
  <a href="https://pku-ov3-lab.github.io/NeurIPS2026-3R-Adapter/">Project Page</a> ·
  <a href="https://pku-ov3-lab.github.io/NeurIPS2026-3R-Adapter/assets/3r-adapter-paper.pdf">Paper</a> ·
  <a href="https://pku-ov3-lab.github.io/NeurIPS2026-3R-Adapter/#video">Video</a> ·
  <a href="https://github.com/PKU-OV3-LAB/NeurIPS2026-3R-Adapter/releases/tag/v0.1.0">Release</a>
</p>

<p align="center">
  <img src="https://pku-ov3-lab.github.io/NeurIPS2026-3R-Adapter/assets/pipeline.png" width="100%" alt="3R-Adapter pipeline">
</p>

Official implementation of **3R-Adapter**, a relation-centric framework for
parameter-efficient adaptation of pretrained 3D reconstruction models.
Deformable Retrieval searches for cross-view evidence, Relational Rewiring
constructs sparse associations, and Consistency Refinement injects the
recovered relations into a frozen backbone. We evaluate low-light
reconstruction, test-time adaptation, and joint human-scene reconstruction.

## Installation

```bash
git clone https://github.com/PKU-OV3-LAB/NeurIPS2026-3R-Adapter.git
cd NeurIPS2026-3R-Adapter
python -m pip install -e ".[test]"
python -m pytest
```

Python 3.10+ and PyTorch 2.0+ are required. Install the selected backbone's
dependencies from its bundled requirements and use a separate environment for
each backbone. All required backbone source is included in `third_party/`.

For Pi3 on RTX 50-series GPUs, the audited CUDA 12.8 profile uses Python 3.11:

```bash
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-pi3-cu128.txt
python -m pip install -e ".[test]"
```

The historical Torch 2.5.1 wheels lack RTX 5090's `sm_120` target. Optional
CUDA RoPE source is bundled with the relevant backbone. Fresh installation
of every backbone dependency set remains unverified; the profile records the
existing environment used for the audited runs.

## Data and pretrained weights

Prepare Oxford Day&Night, 7-Scenes, NRGBD and processed BEDLAM according to
their original dataset terms and the layouts in
[experiments/README.md](experiments/README.md). Supply data, checkpoint and
Human3R model-asset paths explicitly. No datasets or model weights are stored
in Git.

Trained **3R-Adapter checkpoints are not yet published**. Download the official
base weights below, then train the adapter using the adaptation commands.

- **Pi3:** [official model](https://huggingface.co/yyfz233/Pi3),
  [model.safetensors](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors).
  Use the original Pi3 model for the bundled integration.
- **CUT3R:** [official 512 DPT checkpoint](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view),
  `cut3r_512_dpt_4_64.pth`.

```bash
mkdir -p checkpoints/Pi3 checkpoints/CUT3R

curl -fL https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors \
  -o checkpoints/Pi3/model.safetensors
curl -fL https://huggingface.co/yyfz233/Pi3/resolve/main/config.json \
  -o checkpoints/Pi3/config.json

python -m pip install gdown
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD \
  -O checkpoints/CUT3R/cut3r_512_dpt_4_64.pth
```

Keep Pi3's `config.json` beside `model.safetensors`; its evaluation loader
uses this directory to initialize the backbone. [WEIGHTS.md](WEIGHTS.md)
contains the historical checkpoint inventory and provenance.

## Low-light reconstruction and camera pose estimation

Run the following commands from the repository root, with the processed
Oxford Day&Night dataset:

```bash
export OXFORD_DATA=/absolute/path/to/processed/oxford-day-and-night
```

### Pi3 + 3R-Adapter

Start from the official Pi3 weights and train only the adapter:

```bash
python experiments/train_adapter.py --backbone pi3 \
  --pretrained checkpoints/Pi3/model.safetensors \
  --seed 1111 --gpu 0 --output-dir outputs/pi3_seed1111 \
  "train_dataset.OxfordDayAndNight.data_root=$OXFORD_DATA" \
  "test_dataset.OxfordDayAndNight.data_root=$OXFORD_DATA"
```

The preset uses decoder layers **17–24**, residual scale **0.3**, and 20 epochs
of 800 updates. The selected checkpoint is
`outputs/pi3_seed1111/ckpts/best_model/pytorch_model.bin`.

Evaluate the newly trained model:

```bash
bash experiments/run_table1_pi3.sh \
  --release-config none \
  --checkpoint outputs/pi3_seed1111/ckpts/best_model/pytorch_model.bin \
  --pretrained checkpoints/Pi3 \
  --grid-layout pixel_center --sampling-layout reference_major \
  --residual-scale 0.3 \
  --data-root "$OXFORD_DATA" --gpu 0 \
  --output-root outputs/pi3_seed1111_eval --task all
```

`--release-config none` accepts your own trained checkpoint. The historical
release preset checks checkpoint hashes against the archived inventory.

### CUT3R + 3R-Adapter

Start from the official CUT3R 512 DPT weights. Explicit dataset overrides
replace the example paths in the bundled training configuration:

```bash
python experiments/train_adapter.py --backbone cut3r \
  --pretrained checkpoints/CUT3R/cut3r_512_dpt_4_64.pth \
  --grid-layout pixel_center --sampling-layout reference_major \
  --seed 1111 --gpu 0 --output-dir outputs/cut3r_seed1111 \
  --config-name cut3r-3RAdapter \
  "train_dataset=\"4000 @ OxfordDayAndNight(allow_repeat=True, split='train', ROOT='$OXFORD_DATA', aug_crop=16, resolution=[(512,384),(512,336),(512,288),(512,256),(512,208),(512,144),(384,512),(336,512),(288,512),(256,512)], transform=SeqColorJitter, num_views=4, n_corres=0)\"" \
  "test_dataset=\"1000 @ OxfordDayAndNight(split='test', ROOT='$OXFORD_DATA', resolution=224, num_views=4, seed=42, n_corres=0)\""
```

The configuration trains adapters at encoder layers **20–23** and decoder
layers **8–11**, with residual scales **0.3** and **0.5**, respectively.
Training uses 20 epochs, 4,000 samples per epoch, batch size 8 and learning rate
`2e-3`. The selected checkpoint is
`outputs/cut3r_seed1111/checkpoint-best.pth`.

Evaluate the newly trained model:

```bash
bash experiments/run_table1_cut3r.sh \
  --checkpoint outputs/cut3r_seed1111/checkpoint-best.pth \
  --grid-layout pixel_center --sampling-layout reference_major \
  --data-root "$OXFORD_DATA" --gpu 0 \
  --output-root outputs/cut3r_seed1111_eval --task all
```

Both backbones freeze their pretrained parameters during adaptation and use
`pixel_center` / `reference_major` sampling. Repeat with seeds 1112 and 1113
and distinct output directories to report a three-seed mean and sample
standard deviation. Append `--preflight` to a training command to inspect its
resolved arguments without starting training.

The CUT3R scene runner preserves center-224 scoring with ScaleShiftInv and
ICP 0.1 m. The November full-image comparisons use a separate Oxford
protocol; see [results/oxford_20261105](results/oxford_20261105).

## Self-supervised test-time adaptation

Use the Test3R integration to optimize 3R-Adapter from a base DUSt3R checkpoint
on each unlabeled test sequence:

```bash
bash experiments/run_table2_test3r_7scenes.sh \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root SEVENSCENES_DATA \
  --gpu 0 --output-dir outputs/tta_7scenes

bash experiments/run_table2_test3r_nrgbd.sh \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root NRGBD_DATA \
  --gpu 0 --output-dir outputs/tta_nrgbd
```

The runners cover all 18 7-Scenes sequences and all nine NRGBD sequences,
with strides 200 and 500. Each sequence starts from base weights; online
optimization uses two epochs. See [experiments/README.md](experiments/README.md)
for learning rates and optional controls.

## Joint human-scene reconstruction

The Human3R integration evaluates human mesh, scene and depth reconstruction
on processed BEDLAM. Train an adapted model using the Human3R commands in
[experiments/README.md](experiments/README.md), then supply the resulting
checkpoint and the required SMPL/SMPL-X model assets:

```bash
bash experiments/run_table3_human3r_bedlam.sh \
  --release-config none --checkpoint HUMAN3R_ADAPTER_CHECKPOINT \
  --grid-layout pixel_center --sampling-layout reference_major \
  --residual-scale 0.5 \
  --data-root PROCESSED_BEDLAM --model-assets HUMAN3R_MODEL_ASSETS \
  --gpu 0 --output-dir outputs/bedlam_eval
```

The scene cutoff is **15 m**; depth `max_depth=16` is a separate metric setting.

## Cross-view correspondence analysis

Use [experiments/correspondence_README.md](experiments/correspondence_README.md)
for fixed Oxford inputs, passive attention measurements, learned offset
retrieval diagnostics and three-seed aggregation. Original numerical records
are included in `results/correspondence/`.

Historical checkpoints require `legacy` sampling layouts; newly trained
checkpoints require their training layouts. Paper values, historical
checkpoint metrics and November retraining results are recorded separately.

## Citation

```bibtex
@inproceedings{ai2026threeRAdapter,
  title     = {3R-Adapter: Retrieval, Rewiring, and Refinement
               for Efficient Adaptation of 3D Reconstruction Model},
  author    = {Ai, Zixiang and Cui, Zhenyu and Zou, Xu and Zhou, Jiahuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of [Pi3](https://github.com/yyfz/Pi3),
[CUT3R](https://github.com/CUT3R/CUT3R),
[Test3R](https://github.com/nopQAQ/Test3R),
[Human3R](https://github.com/fanegg/Human3R),
[DUSt3R](https://github.com/naver/dust3r),
[CroCo](https://github.com/naver/croco),
[DINOv2](https://github.com/facebookresearch/dinov2) and
[Multi-HMR](https://github.com/naver/multi-hmr) for sharing their work.
