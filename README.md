# 3R-Adapter: Retrieval, Rewiring, and Refinement for Efficient Adaptation of 3D Reconstruction Model

<div align="center">
  <a href="https://azx030512.github.io/">Zixiang Ai</a><sup>1</sup>&emsp;
  Zhenyu Cui<sup>2</sup>&emsp;
  Xu Zou<sup>3</sup>&emsp;
  <a href="https://zhoujiahuan1991.github.io/">Jiahuan Zhou</a><sup>†</sup><sup>1</sup><br>
  <small>
    <sup>1</sup> Peking University<br>
    <sup>2</sup> Tsinghua University<br>
    <sup>3</sup> Huazhong University of Science and Technology
  </small>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2611.11964"><img src="https://img.shields.io/badge/Arxiv-2611.11964-A42C25.svg?logo=arXiv" alt="arXiv:2611.11964"></a>
  <a href="https://github.com/PKU-OV3-LAB/3R-Adapter"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2FPKU-OV3-LAB%2F3R-Adapter&amp;label=3R-Adapter&amp;icon=github&amp;color=%230a58ca" alt="3R-Adapter repository visits"></a>
</p>

<div align="center">
  Official implementation of <b>3R-Adapter: Retrieval, Rewiring, and Refinement for Efficient Adaptation of 3D Reconstruction Model</b>.
  <br><br>
  The paper has been accepted as <b>NeurIPS 2026 Spotlight</b>.
</div>

<p align="center">
  <a href="https://pku-ov3-lab.github.io/3R-Adapter/">Project Page</a> ·
  <a href="https://arxiv.org/pdf/2611.11964">Paper PDF</a> ·
  <a href="https://pku-ov3-lab.github.io/3R-Adapter/#video">Video</a>
</p>

## Overview

**3R-Adapter** is a relation-centric framework for
parameter-efficient adaptation of pretrained 3D reconstruction models.
Deformable Retrieval searches for cross-view evidence, Relational Rewiring
constructs sparse associations, and Consistency Refinement injects the
recovered relations into a frozen backbone.

We evaluate 3R-Adapter on low-light reconstruction and camera pose estimation,
self-supervised test-time adaptation, and joint human-scene reconstruction.

<p align="center">
  <img src="https://pku-ov3-lab.github.io/3R-Adapter/assets/pipeline.png" width="100%" alt="3R-Adapter pipeline">
</p>

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

## Usage

All tasks use `python -m adapter3r`. Append `--help` to a command to see its
options. [Task setup](documentation/tasks.md) describes data layouts and
additional configuration.

### Low-light reconstruction and camera pose estimation

Prepare the processed Oxford Day&Night dataset:

```bash
export OXFORD_DATA=/absolute/path/to/processed/oxford-day-and-night
```

#### Pi3 + 3R-Adapter

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

#### CUT3R + 3R-Adapter

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

### Self-supervised test-time adaptation

Start each test sequence from a base DUSt3R checkpoint:

```bash
python -m adapter3r adapt --dataset 7scenes \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root SEVENSCENES_DATA \
  --gpu 0 --output-dir outputs/tta_7scenes

python -m adapter3r adapt --dataset nrgbd \
  --checkpoint DUST3R_BASE_WEIGHTS --data-root NRGBD_DATA \
  --gpu 0 --output-dir outputs/tta_nrgbd
```

### Joint human-scene reconstruction

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

## Citation

If you find our paper and code useful in your research, please cite:

```bibtex
@misc{ai20263radapterretrievalrewiringrefinement,
  title         = {3R-Adapter: Retrieval, Rewiring, and Refinement
                   for Efficient Adaptation of 3D Reconstruction Model},
  author        = {Zixiang Ai and Zhenyu Cui and Xu Zou and Jiahuan Zhou},
  year          = {2026},
  eprint        = {2611.11964},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.11964}
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
