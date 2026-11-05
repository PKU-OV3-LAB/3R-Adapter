"""Compatibility import for adapter3r.analysis."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter3r.analysis.correspondence_validation import *
