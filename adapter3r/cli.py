"""One command interface for adapter training, evaluation and online adaptation."""
import argparse
import importlib
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m adapter3r",
                                     description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", add_help=False,
                        help="train Pi3, CUT3R or Human3R adapters from base weights")
    evaluate = commands.add_parser("evaluate", add_help=False,
                                   help="evaluate reconstruction and camera pose")
    evaluate.add_argument("--backbone", choices=["pi3", "cut3r", "human3r"], required=True)
    commands.add_parser("adapt", add_help=False,
                        help="self-supervised Test3R adaptation on 7scenes or nrgbd")
    if argv == ["evaluate", "--help"]:
        evaluate.print_help()
        print("\nSelect --backbone pi3, cut3r or human3r, then add --help for task options.")
        return
    args, remaining = parser.parse_known_args(argv)
    if args.command == "train":
        module = "train"
    elif args.command == "adapt":
        module = "adapt_test3r"
    else:
        module = f"evaluate_{args.backbone}"
        defaults = {"--grid-layout": "pixel_center", "--sampling-layout": "reference_major"}
        if args.backbone in {"pi3", "human3r"}:
            defaults.update({"--release-config": "none",
                             "--residual-scale": "0.3" if args.backbone == "pi3" else "0.5"})
        for flag, value in defaults.items():
            if not any(item == flag or item.startswith(flag + "=") for item in remaining):
                remaining.extend([flag, value])
    previous = sys.argv
    try:
        sys.argv = [f"{parser.prog} {args.command}", *remaining]
        importlib.import_module(f"adapter3r.runners.{module}").main()
    finally:
        sys.argv = previous
