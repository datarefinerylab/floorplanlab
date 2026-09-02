"""Polygon geometry: simplification, normalization, edges, gap closing.

Extracted verbatim from the OHD-W notebook's converter cell; the only changes
are that tuning constants now arrive via ConvertConfig instead of module
globals, and normalize_polygons rejects a degenerate layout explicitly rather
than dividing by zero.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .config import ConvertConfig, DEFAULT, NON_ROOM

__all__ = [
    "polygon_iou", "simplify_and_snap", "normalize_polygons", "extract_edges",
    "canonicalize_polygon", "close_room_gaps_morphological",
]


def polygon_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union else 0.0


def simplify_and_snap(poly, zone_type, cfg: ConvertConfig = DEFAULT):
    """
    Simplify polygon (doors/windows get smaller tol) and optionally snap to
    minimum rotated rectangle if IoU is high enough.
    """
    tol = cfg.tol_for(zone_type)
    simp = poly.simplify(tol, preserve_topology=True)
    oobb = simp.minimum_rotated_rectangle
    iou = polygon_iou(simp, oobb)
    return oobb if iou > cfg.snap_iou_threshold else simp

def normalize_polygons(polygons, cfg: ConvertConfig = DEFAULT):
    """Scale + translate all polygons to fit [0,255] with margin."""
    all_coords = np.vstack([np.array(p.exterior.coords) for p in polygons])
    minx, miny = all_coords.min(axis=0)
    maxx, maxy = all_coords.max(axis=0)
    span = max(maxx - minx, maxy - miny)
    if span <= 0:
        raise ValueError('All polygons collapse to a single point; '
                         'cannot normalize this layout.')
    scale = (255 - 2 * cfg.margin) / span

    normed = []
    for p in polygons:
        coords = [((x - minx) * scale + cfg.margin, (y - miny) * scale + cfg.margin)
                  for x, y in p.exterior.coords]
        normed.append(Polygon(coords))
    return normed

def extract_edges(polygons):

    edges = []
    for idx, poly in enumerate(polygons):
        coords = list(poly.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append([round(x1, 1), round(y1, 1),
                          round(x2, 1), round(y2, 1), idx])
    return edges

def _rotate_ring_to_canonical_start(coords):
    ring = coords[:-1]
    start_i = min(
        range(len(ring)),
        key=lambda i: (round(ring[i][1], 1), round(ring[i][0], 1))
    )
    rotated = ring[start_i:] + ring[:start_i]
    return rotated + [rotated[0]]


def canonicalize_polygon(poly, clockwise=True):
    # CW if sign=-1, CCW if sign=+1
    poly2 = orient(poly, sign=-1.0 if clockwise else 1.0)
    ext = _rotate_ring_to_canonical_start(list(poly2.exterior.coords))
    return Polygon(ext)



def close_room_gaps_morphological(
    polygons,
    zone_types,
    buffer_dist=2.5,
    shrink_ratio=0.5,
):
    """
    Morphologically expand and partially shrink architectural spaces
    to close small gaps between room polygons.

    Doors and windows (15/16/17) are left unchanged.
    """
    fixed = []

    for poly, zt in zip(polygons, zone_types):
        if zt not in NON_ROOM:
            p = poly.buffer(
                buffer_dist,
                join_style=2,
                mitre_limit=2.0
            ).buffer(
                -buffer_dist * shrink_ratio,
                join_style=2,
                mitre_limit=2.0
            )

            if p.geom_type == "MultiPolygon":
                p = max(list(p.geoms), key=lambda g: g.area)

            if not p.is_valid:
                p = p.buffer(0)

            fixed.append(p)
        else:
            fixed.append(poly)

    return fixed

