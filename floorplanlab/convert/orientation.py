"""8-way orientation for rooms and windows.

Rooms take the direction from the room-only plan centroid to their own
centroid. Windows keep the orientation supplied by the CSV. Doors get 0.
"""
import math

import numpy as np

from .config import DEFAULT, NON_ROOM

__all__ = ["angle_to_8way_label", "compute_orient_rooms_and_windows"]


def _angle_to_8way_label(angle_deg,
                         diagonal_min=DEFAULT.room_diagonal_min_deg,
                         diagonal_max=DEFAULT.room_diagonal_max_deg):
    """
    Convert Cartesian angle to O-ESD orientation:
      1 E, 2 NE, 3 N, 4 NW, 5 W, 6 SW, 7 S, 8 SE.

    Diagonal precision uses the 30°-60° bands discussed previously.
    """
    a = ((float(angle_deg) + 180.0) % 360.0) - 180.0

    if -diagonal_min < a < diagonal_min:
        return 1
    if diagonal_min <= a <= diagonal_max:
        return 2
    if diagonal_max < a < 180.0 - diagonal_max:
        return 3
    if 180.0 - diagonal_max <= a <= 180.0 - diagonal_min:
        return 4
    if a > 180.0 - diagonal_min or a < -180.0 + diagonal_min:
        return 5
    if -180.0 + diagonal_min <= a <= -180.0 + diagonal_max:
        return 6
    if -180.0 + diagonal_max < a < -diagonal_max:
        return 7
    if -diagonal_max <= a <= -diagonal_min:
        return 8

    raise ValueError(f"Could not classify angle {a}")


def compute_orient_rooms_and_windows(polygons, zone_types, win_orient=None):
    """
    Rooms use the vector from the room-only plan centroid to each room centroid.
    Windows retain the CSV-provided 8-way orientation.
    Doors receive 0.

    No bounding boxes are produced.
    """
    centroids = np.array(
        [p.centroid.coords[0] for p in polygons],
        dtype=float
    )

    room_idxs = [
        i for i, zt in enumerate(zone_types)
        if zt not in NON_ROOM
    ]

    overall_c = (
        centroids[room_idxs].mean(axis=0)
        if room_idxs
        else centroids.mean(axis=0)
    )

    orient_values = [0] * len(polygons)

    for i, zt in enumerate(zone_types):
        if zt in [3, 15, 17]:
            orient_values[i] = 0

        elif zt == 16:
            orient_values[i] = (
                int(win_orient[i]) if win_orient is not None else 0
            )

        else:
            dx = float(centroids[i][0] - overall_c[0])
            dy = float(centroids[i][1] - overall_c[1])

            if math.hypot(dx, dy) < 1e-8:
                # Rare exact-centroid case: use dominant polygon axis fallback.
                xmin, ymin, xmax, ymax = polygons[i].bounds
                w = xmax - xmin
                h = ymax - ymin
                if w >= h:
                    orient_values[i] = 3 if dy >= 0 else 7
                else:
                    orient_values[i] = 1 if dx >= 0 else 5
            else:
                angle_deg = math.degrees(math.atan2(dy, dx))
                orient_values[i] = _angle_to_8way_label(angle_deg)

    return orient_values




# public alias for the notebook's private helper
angle_to_8way_label = _angle_to_8way_label
