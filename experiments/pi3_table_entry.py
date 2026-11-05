"""Compatibility entry; use python -m adapter3r evaluate --backbone pi3."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter3r.runners.evaluate_pi3 import *

if __name__ == "__main__":
    main()
