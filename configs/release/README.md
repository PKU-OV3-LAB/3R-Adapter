# Fixed configurations for the three-seed release

| Backbone | Seeds | Region scale | Fusion factor | Residual scale | Decoder layers |
|---|---|---|---|---|---|
| Pi3 | 1111, 1112, 1113 | 0.2 | 0.1 | 0.3 | 17–24 |
| Human3R | 0, 1, 2 | 0.2 | 0.1 | 0.5 | 8–11 |

Both presets use `pixel_center` and `reference_major`, reduction dimension
128 and four samples per reference. They describe the configuration shared by
three independent runs. A metric mean is computed after evaluating all three
checkpoints. No weights are averaged. Default seeds 1111 and 0 are convenience
defaults, without selecting the best test result.

The presets are loaded by the Pi3 and BEDLAM table runners and by
`train_adapter.py`. A preset fixes the sampling layout, hyperparameters,
architecture, seed and training budget. Conflicting overrides fail rather than
silently creating a different experiment. Checkpoint hashes are checked against
`results/release_weights.json`. Dataset and model asset paths remain explicit.

Pi3 trains for 20 epochs with 800 iterations per epoch. Human3R trains for
40 epochs on `4000 @ BEDLAM_Multi`, with validation on 1000 test samples.
Its preset overrides the bundled historical 32-dimensional model expression
with the actual 128-dimensional model, focal setting and head/fusion ratios
used by these checkpoints. Each run starts from the declared base weights.

Historical/custom runs require `--release-config none`, an explicit checkpoint
and the matching layouts/scales. CUT3R and Test3R retain their existing
configurations; three-seed evidence has not been established for them.

See `experiments/README.md` for commands. Paper values remain in
`results/public_results.json`; release results are recorded separately in
`results/release_3seed_results.json`.
