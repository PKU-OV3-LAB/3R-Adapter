"""Portable data paths for the bundled training configurations."""
import json
from pathlib import Path
import re


def training_data_overrides(backbone, repo, data_root):
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--data-root must be a directory")
    if backbone == "pi3":
        return [f"{split}_dataset.OxfordDayAndNight.data_root={json.dumps(str(root))}"
                for split in ("train", "test")]
    import yaml
    name, key = (("cut3r-3RAdapter", "dataset1") if backbone == "cut3r"
                 else ("train_human3r_adapter-bedlam", "dataset28"))
    config = yaml.safe_load((repo / "config" / f"{name}.yaml").read_text())
    result = []
    for field in (key, "test_dataset"):
        expression, count = re.subn(r"""ROOT\s*=\s*(['"]).*?\1""",
                                   lambda _: "ROOT=" + repr(str(root)), config[field])
        if count != 1:
            raise ValueError(f"expected one dataset ROOT in {field}")
        result.append(f"{field}={json.dumps(expression)}")
    return result
