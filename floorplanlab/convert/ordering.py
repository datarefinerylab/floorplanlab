"""Canonical component ordering, and pruning of orphan interior doors.

Order is: rooms, then interior doors, then windows, with the entrance door
always last. Edges and `ed_rm` must be recomputed after reordering so indices
stay consistent -- the pipeline does this.
"""
import numpy as np
from shapely.geometry import LineString

from .adjacency import assign_ed_rm
from .config import DEFAULT, NON_ROOM
from .geometry import extract_edges

__all__ = [
    "canonical_sort_indices", "apply_reorder",
    "filter_internal_doors_without_two_sided_room_adjacency",
]


def canonical_sort_indices(polygons, zone_types, win_orient=None, include_windows=True):
    """
    Returns a list of indices that sorts zones into a consistent order:
      1) rooms (not in 15/16/17)
      2) interior doors (17)
      3) windows (16)  [if include_windows]
      4) entrance door (15) ALWAYS LAST
    Secondary: sort by centroid y (desc), then x (asc) for stability.
    """
    centroids = np.array([p.centroid.coords[0] for p in polygons])
    # Priority groups
    def prio(zt):
        if zt == 15:
            return 3
        if include_windows and zt == 16:
            return 2
        if zt == 17:
            return 1
        if zt in [16] and not include_windows:
            # shouldn't happen if you filtered windows out
            return 2
        return 0  # rooms

    idxs = list(range(len(polygons)))
    idxs.sort(key=lambda i: (prio(zone_types[i]),
                             -centroids[i][1],   # north first (bigger y)
                             centroids[i][0]))   # west first (smaller x)
    return idxs

def apply_reorder(polygons, zone_types, win_orient, idxs):
    polygons2 = [polygons[i] for i in idxs]
    zone_types2 = [zone_types[i] for i in idxs]
    win_orient2 = [win_orient[i] for i in idxs] if win_orient is not None else None
    return polygons2, zone_types2, win_orient2

def filter_internal_doors_without_two_sided_room_adjacency(
    polygons, zone_types, win_orient=None,
    dist_tol=DEFAULT.edge_adj_dist_tol, parallel_tol=DEFAULT.parallel_tol,
    return_indices=False
):
    """
    Remove internal doors (zt==17) that do NOT have room adjacency on BOTH long edges.
    i.e. keep door if each of its two longest edges is matched to some ROOM by assign_ed_rm.
    """
    if not polygons:
        if return_indices:
            return polygons, zone_types, win_orient, []
        return polygons, zone_types, win_orient

    edges_full = extract_edges(polygons)
    ed_rm = assign_ed_rm(zone_types, edges_full, dist_tol=dist_tol, parallel_tol=parallel_tol)
    edge_lines = [LineString([(e[0], e[1]), (e[2], e[3])]) for e in edges_full]

    door_idxs = [i for i, zt in enumerate(zone_types) if zt == 17]
    room_set = set(i for i, zt in enumerate(zone_types) if zt not in NON_ROOM)

    keep = [True] * len(polygons)

    for di in door_idxs:
        door_edge_ids = [ei for ei, e in enumerate(edges_full) if e[4] == di]
        if len(door_edge_ids) < 4:
            keep[di] = False
            continue

        lengths = [edge_lines[ei].length for ei in door_edge_ids]
        long_local = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)[:2]
        long_edges = [door_edge_ids[k] for k in long_local]

        # count how many long edges got a ROOM match: ed_rm[ei] == [door_idx, room_idx]
        hits = 0
        for ei in long_edges:
            rm = ed_rm[ei]
            if len(rm) == 2 and rm[1] in room_set:
                hits += 1

        # your requested rule:
        # remove only if it DOES NOT have rooms on both long edges simultaneously
        if hits < 2:
            keep[di] = False

    idxs = [i for i, ok in enumerate(keep) if ok]
    polygons2 = [polygons[i] for i in idxs]
    zone_types2 = [zone_types[i] for i in idxs]
    win_orient2 = [win_orient[i] for i in idxs] if win_orient is not None else None
    if return_indices:
        return polygons2, zone_types2, win_orient2, idxs
    return polygons2, zone_types2, win_orient2

