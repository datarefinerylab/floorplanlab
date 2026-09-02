"""Tuning parameters and vocabulary for the CSV -> JSON conversion.

In the notebook these were module-level globals, which meant a single process
could only ever run one configuration, and changing a tolerance to test its
effect meant editing the source. They are grouped here into a frozen dataclass
that is threaded through the pipeline instead.

The defaults reproduce the notebook's values exactly.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

# --- vocabulary -----------------------------------------------------------
# Component classes as stored in `zone_types`.
ROOM_ZONES = (1, 2, 3, 4)
ENTRANCE_DOOR = 15
WINDOW = 16
DOOR = 17
NON_ROOM = (ENTRANCE_DOOR, WINDOW, DOOR)

ZONING_MAP: Mapping[str, int] = MappingProxyType({
    "zone01": 1,
    "zone02": 2,
    "zone03": 3,
    "zone04": 4,
    "DOOR": DOOR,
    "WINDOW": WINDOW,
    "ENTRANCE_DOOR": ENTRANCE_DOOR,
})

# 8-way orientation, Cartesian convention (+x East, +y North).
WINDOW_ORIENT_MAP: Mapping[str, int] = MappingProxyType({
    "East": 1, "North-East": 2, "North": 3, "North-West": 4,
    "West": 5, "South-West": 6, "South": 7, "South-East": 8,
})

ORIENT_NAMES: Mapping[int, str] = MappingProxyType({
    0: "none", 1: "E", 2: "NE", 3: "N", 4: "NW",
    5: "W", 6: "SW", 7: "S", 8: "SE",
})

DROPPED_ZONINGS = ("WALL", "remaining", "COLUMN")


@dataclass(frozen=True)
class ConvertConfig:
    """Geometry and adjacency tolerances for one conversion run."""

    simplify_tol: float = 0.15              # simplification tolerance, normal zones
    simplify_tol_doorwin: float = 0.01      # tighter, for doors/windows (15/16/17)
    snap_iou_threshold: float = 0.90        # IoU above which we snap to oriented box
    edge_adj_dist_tol: float = 4.0          # max distance for adjacency, normalized coords
    parallel_tol: float = 0.50              # cosine threshold for "parallel"
    margin: int = 20                        # margin within the [0,255] frame
    edge_adj_dist_tol_entrance: float = 8.0     # looser, for entrance doors
    parallel_tol_entrance: float = 0.50         # looser, for entrance doors
    room_diagonal_min_deg: float = 30.0     # 8-way diagonal band, lower bound
    room_diagonal_max_deg: float = 60.0     # 8-way diagonal band, upper bound
    gap_buffer_dist: float = 2.5            # morphological close: dilate distance
    gap_shrink_ratio: float = 0.5           # morphological close: re-erode fraction

    def tol_for(self, zone_type: int) -> float:
        """Simplification tolerance for a component class."""
        return (self.simplify_tol_doorwin if zone_type in NON_ROOM
                else self.simplify_tol)

    def __post_init__(self):
        if not 0.0 <= self.snap_iou_threshold <= 1.0:
            raise ValueError("snap_iou_threshold must be in [0, 1]")
        if self.room_diagonal_min_deg >= self.room_diagonal_max_deg:
            raise ValueError("room_diagonal_min_deg must be < room_diagonal_max_deg")
        if self.margin < 0 or self.margin * 2 >= 255:
            raise ValueError("margin must leave room inside the [0, 255] frame")


DEFAULT = ConvertConfig()


@dataclass(frozen=True)
class ContextColumns:
    """CSV column names for the environmental metrics.

    AIF is accepted under several spellings, as in the notebook.
    """

    view_landscape: str = "view_layer_landscape"
    noise_night: str = "noise_night"
    window_perimeter: str = "layout_window_perimeter"
    aif_candidates: Tuple[str, ...] = (
        "afternoon_illuminance_factor",
        "Afternoon Illuminance Factor",
        "AIF",
        "aif",
    )


DEFAULT_COLUMNS = ContextColumns()
