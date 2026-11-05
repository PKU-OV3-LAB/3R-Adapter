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
# Run the remaining commands from the cloned repository.
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

[WEIGHTS.md](WEIGHTS.md) records the available checkpoint inventories and
their provenance. Download endpoints for model checkpoints are not yet
available. Historical checkpoint records and the November Oxford retraining
results are kept separate.

## Tasks

| Paper table | Backbone / dataset | Entry point |
|---|---|---|
| Table 1 | Pi3 / Oxford Day&Night | `experiments/run_table1_pi3.sh` |
| Table 1 | CUT3R / Oxford Day&Night | `experiments/run_table1_cut3r.sh` |
| Table 2 | Test3R / 7-Scenes | `experiments/run_table2_test3r_7scenes.sh` |
| Table 2 | Test3R / NRGBD | `experiments/run_table2_test3r_nrgbd.sh` |
| Table 3 | Human3R / BEDLAM | `experiments/run_table3_human3r_bedlam.sh` |
| Table 4 | Pi3 / Oxford Day&Night | `experiments/run_correspondence.sh` |

Run commands from the repository root:

```bash
bash experiments/run_table1_pi3.sh \
  --checkpoint PI3_ADAPTER_CHECKPOINT --pretrained PI3_BASE_DIRECTORY \
  --data-root OXFORD_DATA --gpu 0 --output-root NEW_PI3_RESULTS --task all

bash experiments/run_table1_cut3r.sh \
  --checkpoint CUT3R_ADAPTER_CHECKPOINT --data-root OXFORD_DATA \
  --grid-layout pixel_center --sampling-layout reference_major \
  --gpu 0 --output-root NEW_CUT3R_RESULTS --task all
```

Pi3 adapters must be attached to decoder layers **17–24**. CUT3R retains
encoder residual scale **0.3** and decoder residual scale **0.5**. The CUT3R
scene runner preserves the original center-224 evaluation entry; the
November full-image results use a separately documented Oxford protocol.

[experiments/README.md](experiments/README.md) provides the complete
evaluation and fresh-training commands.
[experiments/correspondence_README.md](experiments/correspondence_README.md)
describes the independent fixed-input correspondence measurements.
BEDLAM uses a **15 m scene cutoff**; depth `max_depth=16` is a separate setting.

## Sampling and result provenance

```python
from adapter3r import Adapter3R

adapter = Adapter3R(
    1024, 128, K_sample=4,
    grid_layout="pixel_center",
    sampling_layout="reference_major",
)
```

Use `legacy` layouts for checkpoints trained with the historical forward
path. Corrected layouts are `pixel_center` and `reference_major`; parameter
shapes remain compatible, but layouts must match the training run.

Pi3 and CUT3R have each completed three fresh Oxford training runs with the
corrected sampling path, starting from base pretrained weights. Pi3's
sequence-to-global frame indexing was corrected; CUT3R's training loader
already used the correct mapping. Numerical records and protocol definitions
are in [results/oxford_20261105](results/oxford_20261105).

The original paper values remain in `results/public_results.json`, historical
checkpoint metrics in `results/release_3seed_results.json`, and the fixed-input
Table 4 evidence in `results/correspondence/`. Different evaluation protocols
must not be mixed into a single ranking.

## Project page and video

The website, paper, figures, fonts and supplied two-minute video are
distributed as **GitHub Release assets**, outside Git history.
GitHub Pages deploys `project-page.zip` from the release. An ordinary
`git clone`, including a clone fetching all branches, downloads the code
without the project page or video.

To download the website separately:

```bash
gh release download v0.1.0 \
  --repo PKU-OV3-LAB/NeurIPS2026-3R-Adapter \
  --pattern project-page.zip --dir WEBSITE_DOWNLOAD
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

We thank the authors of Pi3, CUT3R, Test3R, Human3R, DUSt3R, CroCo, DINOv2 and
Multi-HMR. Pinned sources, original notices and licenses are retained in
`third_party/`; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

No repository-wide open-source license has been granted for the original
3R-Adapter code. See [LICENSE_STATUS.md](LICENSE_STATUS.md). Third-party source,
derivative patches, datasets and model assets retain their respective terms,
including applicable non-commercial restrictions.
