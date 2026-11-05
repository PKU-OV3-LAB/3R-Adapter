import ast
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def test_bedlam_scene_mask_enforces_paper_15m_cutoff():
    from table3_entry import scene_mask
    depths = torch.tensor([0.0, 14.9, 15.0, 15.1, 200.0])
    valid = torch.ones(5, dtype=torch.bool)
    assert scene_mask(depths, valid).tolist() == [False, True, True, False, False]


def test_bedlam_entry_updates_actual_gt_masks_and_separate_depth_metric(tmp_path):
    from table3_entry import configure_tree, scene_mask
    source = tmp_path / "launch.py"
    source.write_text(
        "def prepare_gt(gts):\n"
        "    valid_masks = [gt['valid_mask'] for gt in gts]\n"
        "    return valid_masks\n"
        "def depth():\n"
        "    return depth_evaluation(1, 2, max_depth=10)\n"
    )
    namespace = {"scene_mask": scene_mask, "depth_evaluation": lambda *args, **kw: kw["max_depth"]}
    exec(compile(configure_tree(source), str(source), "exec"), namespace)
    gt = dict(depthmap=torch.tensor([15., 15.1]), valid_mask=torch.tensor([True, True]))
    assert namespace["prepare_gt"]([gt])[0].tolist() == [True, False]
    assert namespace["depth"]() == 16


def test_full_table_runners_and_compatibility_files_are_shipped():
    required = ["run_table1_pi3.sh", "run_table1_cut3r.sh",
                "run_table2_test3r_7scenes.sh", "run_table2_test3r_nrgbd.sh",
                "run_table3_human3r_bedlam.sh", "train_adapter.py"]
    for name in required:
        assert (ROOT / "experiments" / name).is_file()
    for name in ("cut3r", "human3r"):
        recipe = json.loads((ROOT / "experiments/patches" / f"{name}-recipe.json").read_text())
        assert len(recipe["revision"]) == 40
        assert "src/dust3r/model.py" in recipe["patched_sha256"]
        assert (ROOT / "experiments/patches" / f"{name}-tables.patch").stat().st_size > 1000


def test_table2_dataset_selection_covers_all_sequences(tmp_path):
    from table2_entry import dataset_tree
    source = tmp_path / "launch.py"
    source.write_text("def main(args):\n    resolution=(512,384)\n    datasets_all={}\n    return datasets_all\n")
    for name, count in (("7scenes", 18), ("nrgbd", 9)):
        tree, key, expected, stride = dataset_tree(source, name, tmp_path)
        class Dataset:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __len__(self):
                return count
        scope = {"SevenScenes": Dataset, "NRGBD": Dataset}
        exec(compile(tree, str(source), "exec"), scope)
        result = scope["main"](None)
        assert len(result[key]) == expected == count
        assert result[key].kwargs["kf_every"] == stride
