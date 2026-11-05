import subprocess
from pathlib import Path


EXPERIMENTS = Path(__file__).parents[1] / "experiments"


def test_table1_runner_contract():
    script = (EXPERIMENTS / "run_table1_pi3.sh").read_text(encoding="utf-8")
    assert "pi3_table_entry.py" in script
    import json
    source = EXPERIMENTS.parent / "third_party/pi3"
    manifest = json.loads((source / "SOURCE.json").read_text())
    assert manifest["base_revision"].startswith("39c8162")
    assert "peft_layers=[[],[17,18,19,20,21,22,23,24]]" in (
        source / "pi3/models/pi3_Adapter3R.py").read_text()


def test_table2_runner_contract():
    script = (EXPERIMENTS / "run_table2_test3r_nrgbd.sh").read_text(
        encoding="utf-8"
    )
    assert 'table2_entry.py" --dataset nrgbd' in script
    assert (EXPERIMENTS.parent / "third_party/test3r/eval/mv_recon/launch.py").is_file()
    assert "conda" not in script


def test_pi3_gpu_patch_applies(tmp_path):
    patch = (
        EXPERIMENTS
        / "patches"
        / "pi3-eval-67de78e-respect-cuda-visible-devices.patch"
    )
    (tmp_path / "mv_recon").mkdir()
    (tmp_path / "relpose").mkdir()
    (tmp_path / "mv_recon" / "eval.py").write_text(
        "import os\n"
        "os.environ['OMP_NUM_THREADS'] = '8'\n"
        "os.environ['MKL_NUM_THREADS'] = '8'\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = '4'\n"
        "os.environ['CUDA_LAUNCH_BLOCKING'] = '1'\n",
        encoding="utf-8",
    )
    (tmp_path / "relpose" / "eval_dist.py").write_text(
        "import os\n"
        "os.environ['OMP_NUM_THREADS'] = '8'\n"
        "os.environ['MKL_NUM_THREADS'] = '8'\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = '1'\n"
        "os.environ['CUDA_LAUNCH_BLOCKING'] = '1'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "apply", "--check", str(patch)], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)
    assert "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4')" in (
        tmp_path / "mv_recon" / "eval.py"
    ).read_text(encoding="utf-8")


def test_pi3_decoder_patch_applies(tmp_path):
    patch = (
        EXPERIMENTS / "patches" / "pi3-eval-67de78e-decoder-17t24.patch"
    )
    model_dir = tmp_path / "pi3" / "models"
    model_dir.mkdir(parents=True)
    model = model_dir / "pi3_Adapter3R.py"
    model.write_text(
        "class Pi3_Adapter3R:\n"
        "    def __init__(\n"
        "            self,\n"
        "            pos_type='rope100',\n"
        "            decoder_size='large',\n"
        "            peft_layers=[[],[30,31,32,33,34,35]]\n"
        "        ):\n"
        "        super().__init__()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "apply", "--check", str(patch)], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)
    assert "peft_layers=[[],[17,18,19,20,21,22,23,24]]" in model.read_text(
        encoding="utf-8"
    )
