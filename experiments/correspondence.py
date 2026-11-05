"""Compatibility entry for adapter3r.analysis.correspondence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter3r.analysis.correspondence import *

if __name__ == "__main__":
    main()
