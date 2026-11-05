# Third-party notices

The required third-party source is included under `third_party/`. Original
copyright notices and license files are retained. Source inventories record the
pinned upstream revisions, original hashes and published hashes. Local changes
include adapter integration, table protocols, GPU selection and portable paths.

| Directory | Included components | License retained |
|---|---|---|
| `pi3`, `pi3_train` | Pi3 models, Oxford data loaders, evaluation and training | Pi3 `LICENSE` |
| `cut3r` | CUT3R model, Oxford evaluation and training | CC BY-NC-SA 4.0 |
| `test3r` | Test3R/DUSt3R model, online adaptation and scene evaluation | CC BY-NC-SA 4.0, `NOTICE` |
| `human3r` | Human3R model, BEDLAM evaluation/training and Multi-HMR components | Human3R `LICENSE`, including NAVER non-commercial terms |
| Nested `croco` | CroCo models and optional CUDA RoPE source | CroCo `LICENSE` and `NOTICE` |
| Human3R `src/dinov2` | Local DINOv2 source | DINOv2 `LICENSE` and nested notices |
| Release website `_shared/fonts` | Linux Libertine, Work Sans, Red Hat Mono | Libertine GPL with font exception and original dual-license notice; Work Sans / Red Hat Mono OFL |

These components retain their individual terms. The project's pending license
does not override third-party licenses or grant commercial rights.

Model weights, SMPL/SMPL-X assets and datasets are acquired separately under
their own terms. This repository grants no rights to those assets.

Linux Libertine's original license text and attribution are included beside
the unmodified website fonts. Font terms apply independently to those assets.
