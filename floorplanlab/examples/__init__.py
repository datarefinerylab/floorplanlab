"""Small example data shipped with the library, for the tutorial and for trying
things out without needing the real dataset.

    from floorplanlab.examples import sample_layout, sample_log

    csv_path = sample_layout()      # one layout CSV, ready for csv_to_json()
    log_text = sample_log()         # stdout from a real sampling run
"""
from pathlib import Path

__all__ = ["sample_layout", "sample_log", "DATA_DIR"]

DATA_DIR = Path(__file__).parent


def sample_layout() -> Path:
    """Path to a single example layout CSV (6 rooms, doors, windows)."""
    return DATA_DIR / "sample_layout.csv"


def sample_log() -> str:
    """Stdout from a real sampling run, for trying the metrics parser."""
    return (DATA_DIR / "sample_sampling_log.txt").read_text()
