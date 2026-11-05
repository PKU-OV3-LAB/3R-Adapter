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
  <a href="https://pku-ov3-lab.github.io/3R-Adapter/">Project Page</a> ·
  <a href="https://pku-ov3-lab.github.io/3R-Adapter/assets/3r-adapter-paper.pdf">Paper</a> ·
  <a href="https://pku-ov3-lab.github.io/3R-Adapter/#video">Video</a>
</p>

<p align="center">
  <img src="https://pku-ov3-lab.github.io/3R-Adapter/assets/pipeline.png" width="100%" alt="3R-Adapter pipeline">
</p>

Official implementation of **3R-Adapter**, a relation-centric framework for
parameter-efficient adaptation of pretrained 3D reconstruction models.
Deformable Retrieval searches for cross-view evidence, Relational Rewiring
constructs sparse associations, and Consistency Refinement injects the
recovered relations into a frozen backbone.

## Installation

```bash
git clone https://github.com/PKU-OV3-LAB/3R-Adapter.git
cd 3R-Adapter
python -m pip install -e .
```

Use Python 3.10+ and install the selected backbone's dependencies from
`third_party/`. For Pi3 on RTX 50-series GPUs:

```bash
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-pi3-cu128.txt
```

## Code structure

```text
adapter3r/
  core.py          # Shared 3R-Adapter module
  integrations/    # Backbone integration helpers
  runners/         # Training, evaluation and online adaptation
  analysis/        # Correspondence measurements and result processing
configs/           # Adapter configurations
third_party/       # Bundled backbone source
documentation/     # Task setup and reference material
tests/             # Validation
```

All tasks use `python -m adapter3r`. Run a command with `--help` to see its
options. [Task setup](documentation/tasks.md) describes data layouts and
additional configuration.

## Pretrained weights

**Trained 3R-Adapter weights have not yet been published.** Download the
official base weights and train the adapter with the commands below.

- **Pi3:** [model.safetensors](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors)
  and [config.json](https://huggingface.co/yyfz233/Pi3/resolve/main/config.json).
- **CUT3R:** [512 DPT checkpoint](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view).

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

## Low-light reconstruction and camera pose estimation

Prepare the processed Oxford Day&Night dataset:

```bash
export OXFORD_DATA=/absolute/path/to/processed/oxford-day-and-night
```

### Pi3 + 3R-Adapter

Train from the official base weights, then evaluate reconstruction and pose:

```bash
python -m adapter3r train --backbone pi3 \
  --pretrained checkpoints/Pi3/model.safetensors \
  --data-root "$OXFORD_DATA" --seed 1111 --gpu 0 \
  --output-dir outputs/pi3

python -m adapter3r evaluate --backbone pi3 \
  --checkpoint outputs/pi3/ckpts/best_model/pytorch_model.bin \
  --pretrained checkpoints/Pi3 --data-root "$OXFORD_DATA" \
  --gpu 0 --output-dir outputs/pi3_eval
```

### CUT3R + 3R-Adapter

```bash
python -m adapter3r train --backbone cut3r \
  --pretrained checkpoints/CUT3R/cut3r_512_dpt_4_64.pth \
  --data-root "$OXFORD_DATA" --seed 1111 --gpu 0 \
  --output-dir outputs/cut3r

python -m adapter3r evaluate --backbone cut3r \
  --checkpoint outputs/cut3r/checkpoint-best.pth \
  --data-root "$OXFORD_DATA" --gpu 0 \
  --output-dir outputs/cut3r_eval
```

## Self-supervised test-time adaptation

Start each test sequence from a base DUSt3R checkpoint:

```bash
python -m adapter3r adapt --dataset 7scenes \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root SEVENSCENES_DATA \
  --gpu 0 --output-dir outputs/tta_7scenes

python -m adapter3r adapt --dataset nrgbd \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root NRGBD_DATA \
  --gpu 0 --output-dir outputs/tta_nrgbd
```

## Joint human-scene reconstruction

Following Human3R's joint reconstruction pipeline, initialize the scene
backbone from **CUT3R** and the human branch from **Multi-HMR**, then adapt
on BEDLAM. In these commands, `human3r` selects the joint reconstruction
implementation; it does not denote a pretrained Human3R checkpoint.

```bash
python -m adapter3r train --backbone human3r \
  --pretrained CUT3R_BASE_WEIGHTS --pretrained-human MULTIHMR_WEIGHTS \
  --model-assets HUMAN3R_MODEL_ASSETS --data-root PROCESSED_BEDLAM \
  --seed 0 --gpu 0 --output-dir outputs/bedlam

python -m adapter3r evaluate --backbone human3r \
  --checkpoint outputs/bedlam/checkpoint-best.pth \
  --data-root PROCESSED_BEDLAM --model-assets HUMAN3R_MODEL_ASSETS \
  --gpu 0 --output-dir outputs/bedlam_eval
```

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
