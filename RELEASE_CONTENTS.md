# Public package contents

## Functional code

- `adapter3r/core.py`: checkpoint-compatible Adapter3R; legacy and corrected
  sampling, FP32 dense grids and explicit query positions.
- `adapter3r/windows.py`: reference view selection.
- `adapter3r/manifest.py`: checkpoint utilities and adapter configuration records.
- `adapter3r/integrations/`: Pi3 and Human3R tensor integration helpers.
- Table 1 Pi3: `run_table1_pi3.sh`, `pi3_entry.py`, decoder/GPU compatibility patches.
- Table 1 CUT3R: `run_table1_cut3r.sh`, `cut3r_entry.py`, CUT3R model/loader/evaluator patches.
- Table 2 7-Scenes: `run_table2_test3r_7scenes.sh`, `table2_entry.py`; all 18 sequences.
- Table 2 NRGBD: `run_table2_test3r_nrgbd.sh`, `test3r_nrgbd_entry.py`; all nine sequences.
- Table 3 BEDLAM: `run_table3_human3r_bedlam.sh`, `table3_entry.py`; human, scene and depth metrics.
- Table 4 Oxford: `run_correspondence.sh`, `correspondence.py`, portable GT
  geometry, three-seed summary, shard merge and full-token profile tools.
- `prepare_upstream.py` and `patches/*-recipe.json`: fresh upstream worktrees,
  source hash checks and compatibility patch application.
- `train_adapter.py`, `training_entry.py`: fresh adapter training integration
  for Pi3/CUT3R/Human3R; Pi3/Human3R release presets fix the model, seed and budget.
- `configs/release/`: corrected Pi3/Human3R fixed configurations for three seeds.
- `release_config.py`: preset resolution, seed/layout/hyperparameter checks and
  checkpoint hash verification.
- `collect_results.py`: JSON/CSV metrics and source/checkpoint metadata;
  optional completed-run checks.

All entrypoints live under `experiments/`. CUT3R and Human3R patch recipes also
include the corresponding adapter training configuration and necessary training
source changes. Required backbone implementations are included in `third_party/`.
Weights and data are supplied separately. Source preparation and patches remain
optional historical tools; default execution uses bundled source directly.

## Documentation and assets

- `README.md` and `experiments/README.md`: setup, source revisions and commands.
- `requirements-pi3-cu128.txt`: Pi3 versions used for the RTX 5090 audit;
  separate from historical Torch 2.5.1 requirements.
- `results/public_results.json`: paper values and existing reruns, with their
  numerical evidence kept separate.
- `results/release_weights.json`, `results/release_3seed_results.json`:
  six independent checkpoint hashes and three-seed results.
- `WEIGHTS.md`: checkpoint formats, seeds, hashes and use.
- `results/correspondence/`: original numerical CSVs and sanitized provenance;
  fixed 204-input selection under `configs/oxford_correspondence_samples.json`.
  `port_validation.json` records exact full three-seed port comparisons.
- `RELEASE_NOTES.md`: release fixes and the separately documented effect of the
  Oxford training frame-index correction on future training.
- `tests/`: core, sampling, runner, table integration and release-scope checks.
- `results/oxford_20261105/`: fresh three-seed Oxford author-method results
  and center-224 protocol comparisons, separate from historical metrics.
- Release `project-page.zip`: academic project webpage, figures, paper, talk,
  local font assets and the supplied MP4; excluded from Git.
- `.github/workflows/pages.yml`: deploys the website from its Release asset.
- `CITATION.cff`, `pyproject.toml`, `.gitignore`, `THIRD_PARTY_NOTICES.md`,
  `LICENSE_STATUS.md`.
- `third_party/{pi3,pi3_train,cut3r,test3r,human3r}`: model implementations,
  data loaders, native extension sources, evaluation and training configs.
- `third_party/MANIFEST.json` and each `SOURCE.json`: pinned revisions,
  original and published hashes, and local portability changes.

BEDLAM's paper and current scene cutoff are both 15m. Depth `max_depth=16`
is independently recorded. Core tests and release-fix validation pass;
this does not imply a new full Tables 1–3 run through every packaged entry or
a clean installation validation of every backbone dependency. Six fresh
Oxford author-method runs and center-224 comparisons have a separate evidence
record; the CUT3R packaged scene entry preserves its original center protocol.

Excluded from this package: checkpoint files, dataset RGB/depth files, runtime
logs, cross-method comparison implementations and historical machine
experiment directories. This package targets the paper's GitHub repository.
The distribution license remains to be chosen by the copyright holder.
