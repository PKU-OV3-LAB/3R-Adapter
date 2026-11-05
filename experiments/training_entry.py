"""Install adapters inside the training process, then execute upstream training."""
import argparse
from pathlib import Path
import runpy
import sys
from runtime import paths, install_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--grid-layout", required=True)
    p.add_argument("--sampling-layout", required=True)
    p.add_argument("--region-scale", type=float, default=0.2)
    p.add_argument("--fusion-factor", type=float, default=0.1)
    p.add_argument("--residual-scale", type=float, default=0.3)
    args, command = p.parse_known_args()
    command = command[1:] if command and command[0] == "--" else command
    import os
    os.environ.update(ADAPTER3R_REGION_SCALE=str(args.region_scale),
                      ADAPTER3R_FUSION_FACTOR=str(args.fusion_factor),
                      ADAPTER3R_RESIDUAL_SCALE=str(args.residual_scale))
    if args.backbone == "pi3":
        # One-process torchrun semantics without introducing a second process
        # that would lose the adapter installation.
        import os
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
        source_index = next(i for i,value in enumerate(command) if value.endswith("train_pi3.py"))
        source = command[source_index]
        sys.path.insert(0, str(args.repo))
        from pi3_entry import main as run_pi3
        sys.argv = ["pi3_entry", "--source-entry", source,
                    "--grid-layout", args.grid_layout, "--sampling-layout", args.sampling_layout,
                    *command[source_index + 1:]]
        run_pi3()
    else:
        paths(args.repo)
        install_adapter(args.grid_layout, args.sampling_layout, args.region_scale,
                        args.fusion_factor, args.residual_scale)
        source = command[1]
        sys.argv = [source, *command[2:]]
        runpy.run_path(source, run_name="__main__")


if __name__ == "__main__":
    main()
