"""Compatibility entry for adapter3r.analysis.correspondence_summary."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy

if __name__ == "__main__":
    runpy.run_module("adapter3r.analysis.correspondence_summary", run_name="__main__")
