# Package contents

`adapter3r` contains the shared adapter module, integration helpers and all
training, evaluation and analysis implementations. `python -m adapter3r` is
the public task interface.

- `adapter3r/runners/`: backbone runners, data paths, configuration and runtime.
- `adapter3r/analysis/`: cross-view measurements, summaries and metric collection.
- `configs/`: model settings and the fixed Oxford input selection.
- `third_party/`: pinned backbone source and retained third-party licenses.
- `results/`: original numerical records and separately identified reruns.
- `documentation/`: task setup, checkpoint inventory and release history.
- `tests/`: existing adapter, sampling, integration and runner checks.

`experiments/` contains compatibility launchers and historical patch recipes.
Its Python entries forward to package implementations; they do not contain
separate copies of the models or evaluators.

The project webpage and video are Release assets, deployed by
`.github/workflows/pages.yml`, and are excluded from ordinary Git clones.
Model checkpoints and datasets are supplied separately.
