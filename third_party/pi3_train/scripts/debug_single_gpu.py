# debug_single_gpu.py
import os
import sys
import runpy
import torch
import torch.distributed as dist

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

if torch.cuda.is_available():
    torch.cuda.set_device(0)

if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=0,
        world_size=1,
    )

debug_overrides = [
    "train.distributed=false",
    "train.world_size=1",
    "train.rank=0",
    "train.gpu=0",
    "train.num_workers=2",
    "train.dynamic_compile=false",
    "name=train_peft_3radapter_30t35_highres_lr1en4"
]

sys.argv = ["scripts/train_pi3.py"] + debug_overrides

runpy.run_path("scripts/train_pi3.py", run_name="__main__")