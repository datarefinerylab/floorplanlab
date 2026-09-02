"""CSV -> JSON conversion for the OHD-W pipeline.

The notebook kept all of this in a single 1,354-line cell. Same behaviour,
split by concern and with the tuning constants gathered into ConvertConfig:

    from floorplanlab.convert import csv_to_json, ConvertConfig

    data = csv_to_json("layout.csv", output_dir="out/", show_plot=False)
    loose = csv_to_json("layout.csv", cfg=ConvertConfig(edge_adj_dist_tol=6.0))
"""
from .batch import batch_convert_csvs
from .config import (DEFAULT, DEFAULT_COLUMNS, DOOR, ENTRANCE_DOOR, NON_ROOM,
                     ORIENT_NAMES, ROOM_ZONES, WINDOW, WINDOW_ORIENT_MAP,
                     ZONING_MAP, ContextColumns, ConvertConfig)
from .context import compute_contextual_orientations
from .orientation import angle_to_8way_label, compute_orient_rooms_and_windows
from .pipeline import csv_to_json, csv_to_json_no_windows
from .splits import make_splits, write_list_txt

__all__ = [
    "csv_to_json", "csv_to_json_no_windows", "batch_convert_csvs",
    "make_splits", "write_list_txt",
    "ConvertConfig", "ContextColumns", "DEFAULT", "DEFAULT_COLUMNS",
    "compute_contextual_orientations", "compute_orient_rooms_and_windows",
    "angle_to_8way_label",
    "ZONING_MAP", "WINDOW_ORIENT_MAP", "ORIENT_NAMES",
    "ROOM_ZONES", "NON_ROOM", "DOOR", "WINDOW", "ENTRANCE_DOOR",
]
