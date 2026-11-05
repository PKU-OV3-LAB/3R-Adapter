import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_release_has_only_public_scope_and_no_runtime_artifacts():
    blocked_parts = {
        "base" + "lines",
        "repro" + "ductions",
        "moni" + "toring",
    }
    blocked_suffixes = {".log", ".pyc", ".pyo"}
    runtime_parts = {"__py" + "cache__", ".pytest" + "_cache"}
    for path in ROOT.rglob("*"):
        relative_parts = set(path.relative_to(ROOT).parts)
        if relative_parts & runtime_parts:
            continue
        assert not relative_parts & blocked_parts, path
        assert path.suffix not in blocked_suffixes, path


def test_text_files_do_not_expose_private_paths_or_unreleased_fields():
    private_prefixes = (
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "/" + "data" + "/",
    )
    unreleased_terms = (
        "unified" + "_correspondence",
        "shared" + "-mask",
        "shared" + "_mask",
        "oracle" + "_best_of4",
    )
    text_suffixes = {".py", ".sh", ".md", ".toml", ".json", ".patch"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert not any(re.search(r"(?<![a-z0-9_.])" + re.escape(prefix.lower()), text)
                       for prefix in private_prefixes), path
        assert not any(term in text for term in unreleased_terms), path


def test_public_results_schema_and_values():
    results = json.loads(
        (ROOT / "results" / "public_results.json").read_text(encoding="utf-8")
    )
    assert results["schema_version"] == 2
    assert results["method"] == "3R-Adapter"
    table_1 = results["tables"]["table_1"]["backbones"]
    assert table_1["Pi3"]["decoder_layers"] == list(range(17, 25))
    assert table_1["Pi3"]["camera_pose"]["ate"] == 0.2584515308032284
    assert (
        table_1["Pi3"]["scene_reconstruction"]["chamfer"]
        == 0.6328260895429239
    )
    assert table_1["CUT3R"]["scene_reconstruction"]["chamfer"] == 0.581
    table_2 = results["tables"]["table_2"]["datasets"]
    assert table_2["7-Scenes"]["chamfer"] == 0.216
    assert table_2["NRGBD"]["normal_consistency"] == 0.879
    table_3 = results["tables"]["table_3"]
    assert table_3["scene_reconstruction_corrected"]["depth_cutoff_m"] == 15
    assert (
        table_3["human_mesh_reconstruction_mm"]["metric_mpjpe"] == 217.2
    )
