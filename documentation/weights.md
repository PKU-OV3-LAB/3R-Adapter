# Pretrained weights and adapter checkpoints

Trained 3R-Adapter checkpoints are not yet available for public download.
The current GitHub Release contains website and video assets only.

Official base weights for adaptation:

- [Pi3 model.safetensors](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors)
  and [config.json](https://huggingface.co/yyfz233/Pi3/resolve/main/config.json).
- [CUT3R 512 DPT checkpoint](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view),
  `cut3r_512_dpt_4_64.pth`.

The [README adaptation commands](../README.md#low-light-reconstruction-and-camera-pose-estimation)
start from these base weights and produce a new adapted checkpoint.

## Historical three-seed checkpoint inventory

The inventory contains six independent validation-selected checkpoints using
the fixed presets in `configs/release/`. Their metadata is included for
provenance; the checkpoint files are not included in this release.

| Backbone | Seeds | Checkpoint size per seed | Selection |
|---|---|---|---|
| Pi3 | 1111, 1112, 1113 | 3,848,092,877 bytes | Minimum validation loss |
| Human3R | 0, 1, 2 | 4,833,084,106 bytes | Minimum validation median loss |

The directory structure is:

```text
WEIGHTS_ROOT/
  pi3/seed1111/pytorch_model.bin
  pi3/seed1112/pytorch_model.bin
  pi3/seed1113/pytorch_model.bin
  human3r/seed0/checkpoint-best.pth
  human3r/seed1/checkpoint-best.pth
  human3r/seed2/checkpoint-best.pth
```

These are original full checkpoint snapshots, rather than extracted adapter
tensors. Each file retains its original hash. SHA256, bytes, run identifiers,
checkpoint epoch and base-weight hashes are in `results/release_weights.json`.
Pretrained base weights and Human3R model assets are acquired separately under
their respective licenses.

Mean and sample standard deviation in `results/release_3seed_results.json`
refer to metrics from all three independent models. No averaged checkpoint,
ensemble inference or single-seed guarantee is implied. Default seeds 1111 and
0 make single-model inference convenient; use all seeds to compute the mean.

The paper's numerical tables remain unchanged and separately recorded.
The October experiments provide evidence for these checkpoints. Full reruns
through the new packaged evaluation entries have a separate verification status.
No public download endpoint is configured by this preparation step.

## November Oxford retraining

Pi3 and CUT3R were each retrained for seeds 1111, 1112 and 1113 with corrected
sampling. Their November metrics are in `results/oxford_20261105`; they do not
refer to the six Pi3/Human3R checkpoint snapshots listed above. These newer
checkpoint files also have no public download endpoint in this release.
