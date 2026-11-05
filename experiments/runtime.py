"""Compatibility import; implementation lives in adapter3r.runners."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter3r.runners.runtime import *
