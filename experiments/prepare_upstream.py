"""Compatibility entry for adapter3r.runners.prepare_upstream."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter3r.runners.prepare_upstream import *

if __name__ == "__main__":
    main()
