"""CSV -> JSON conversion for one layout.

Pipeline order matters and is load-bearing:

    read -> drop non-spaces -> simplify -> normalize -> close room gaps
         -> drop orphan interior doors -> canonical reorder
         -> orientations -> environmental labels
         -> canonicalize rings -> edges + ed_rm -> validate

Edges and `ed_rm` are computed only after the reorder, so component indices in
`ed_rm` refer to the same positions as `zone_types`.
"""
import json
import logging
import os

import pandas as pd
from shapely import wkt

from .adjacency import assign_ed_rm, repair_room_room_reciprocity, validate_ed_rm
from .config import (DEFAULT, DEFAULT_COLUMNS, DROPPED_ZONINGS, ConvertConfig,
                     ContextColumns, WINDOW_ORIENT_MAP, ZONING_MAP)
from .context import compute_contextual_orientations, resolve_column
from .geometry import (canonicalize_polygon, close_room_gaps_morphological,
                       extract_edges, normalize_polygons, simplify_and_snap)
from .ordering import (apply_reorder, canonical_sort_indices,
                       filter_internal_doors_without_two_sided_room_adjacency)
from .orientation import compute_orient_rooms_and_windows
from .plotting import plot_with_polygons

log = logging.getLogger(__name__)

__all__ = ["csv_to_json", "csv_to_json_no_windows"]


def csv_to_json(csv_path, show_plot=True, output_dir=None, custom_filename=None,
                plot_path=None, cfg: ConvertConfig = DEFAULT,
                columns: ContextColumns = DEFAULT_COLUMNS):
    df = pd.read_csv(csv_path)
    df = df[~df["zoning"].isin(DROPPED_ZONINGS)].reset_index(drop=True)

    polygons = []
    zone_types = []
    win_orient = []
    context_records = []

    view_col = resolve_column(df, columns.view_landscape)
    noise_col = resolve_column(df, columns.noise_night)
    perimeter_col = resolve_column(df, columns.window_perimeter)
    aif_col = resolve_column(df, columns.aif_candidates)

    for _, row in df.iterrows():
        geom = wkt.loads(row["geometry"])
        if geom.geom_type == "MultiPolygon":
            geom = max(list(geom.geoms), key=lambda g: g.area)

        zname = row["zoning"]
        zt = ZONING_MAP.get(zname, 0)

        polygons.append(geom)
        zone_types.append(zt)

        if zname == "WINDOW":
            wo = WINDOW_ORIENT_MAP.get(str(row.get("window_orientation", "")), 0)
        else:
            wo = 0
        win_orient.append(wo)

        context_records.append({
            "view_layer_landscape": row.get(view_col),
            "noise_night": row.get(noise_col),
            "layout_window_perimeter": row.get(perimeter_col),
            "aif": row.get(aif_col),
        })

    # simplify first (still in original coords)
    polygons = [simplify_and_snap(p, zt, cfg) for p, zt in zip(polygons, zone_types)]

    # normalize
    polygons = normalize_polygons(polygons, cfg)
    polygons = close_room_gaps_morphological(
        polygons,
        zone_types,
        buffer_dist=cfg.gap_buffer_dist,
        shrink_ratio=cfg.gap_shrink_ratio,
    )

    # filter orphan internal doors (needs normalized coords)
    before = sum(1 for z in zone_types if z == 17)
    polygons, zone_types, win_orient, kept_idxs = filter_internal_doors_without_two_sided_room_adjacency(
        polygons, zone_types, win_orient=win_orient,
        dist_tol=cfg.edge_adj_dist_tol, parallel_tol=cfg.parallel_tol,
        return_indices=True
    )
    context_records = [context_records[i] for i in kept_idxs]
    after = sum(1 for z in zone_types if z == 17)
    log.debug("removed %d internal doors lacking two-sided room adjacency",
              before - after)

    # CANONICAL REORDER (rooms -> door17 -> win16 -> entrance15 last)
    idxs = canonical_sort_indices(polygons, zone_types, win_orient=win_orient, include_windows=True)
    context_records = [context_records[i] for i in idxs]
    polygons, zone_types, win_orient = apply_reorder(polygons, zone_types, win_orient, idxs)

    # 8-way component orientations
    orient = compute_orient_rooms_and_windows(polygons, zone_types, win_orient)

    # Layout-level contextual orientation labels
    contextual = compute_contextual_orientations(
        zone_types,
        orient,
        context_records,
    )

    # edges + ed_rm AFTER reorder (indices consistent)
    polygons = [canonicalize_polygon(p, clockwise=True) for p in polygons]
    edges_full = extract_edges(polygons)

    ed_rm = assign_ed_rm(zone_types, edges_full,
                         dist_tol=cfg.edge_adj_dist_tol,
                         parallel_tol=cfg.parallel_tol)
    ed_rm, repaired_pairs = repair_room_room_reciprocity(zone_types, edges_full, ed_rm)
    if repaired_pairs:
        log.debug("repaired %d reciprocal room-room ed_rm mappings",
                  len(repaired_pairs))
    validate_ed_rm(zone_types, edges_full, ed_rm)
    edges = [[e[0], e[1], e[2], e[3]] for e in edges_full]

    data = {
        "zone_types": zone_types,
        "edges": edges,
        "ed_rm": ed_rm,
        "orient": orient,
        "AIF": contextual["AIF"],
        "VL": contextual["VL"],
        "NN": contextual["NN"],
    }

    if custom_filename:
        base_name = custom_filename
    else:
        base_name = os.path.splitext(os.path.basename(csv_path))[0] + ".json"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, base_name)
    else:
        out_path = os.path.splitext(csv_path)[0] + ".json"

    with open(out_path, "w") as f:
        json.dump(data, f, separators=(", ", ": "))
    log.info("wrote %s", out_path)

    if show_plot or plot_path is not None:
        plot_with_polygons(
            polygons,
            edges_full,
            ed_rm,
            zone_types,
            orient,
            win_orient=win_orient,
            save_path=plot_path,
            show=show_plot,
        )

    return data

# ============================================================
# --- CSV → JSON (NO WINDOWS VERSION) ---
# ============================================================

def csv_to_json_no_windows(
    csv_path,
    show_plot=True,
    output_dir=None,
    custom_filename=None,
    plot_path=None,
    cfg: ConvertConfig = DEFAULT,
    columns: ContextColumns = DEFAULT_COLUMNS,
):
    df = pd.read_csv(csv_path)
    df = df[~df["zoning"].isin(DROPPED_ZONINGS)].reset_index(drop=True)

    df_nowin = df[df["zoning"] != "WINDOW"].reset_index(drop=True)

    polygons = []
    zone_types = []
    context_records = []

    view_col = resolve_column(df_nowin, columns.view_landscape)
    noise_col = resolve_column(df_nowin, columns.noise_night)
    perimeter_col = resolve_column(df_nowin, columns.window_perimeter)
    aif_col = resolve_column(df_nowin, columns.aif_candidates)

    for _, row in df_nowin.iterrows():
        geom = wkt.loads(row["geometry"])
        if geom.geom_type == "MultiPolygon":
            geom = max(list(geom.geoms), key=lambda g: g.area)

        zname = row["zoning"]
        zt = ZONING_MAP.get(zname, 0)

        polygons.append(geom)
        zone_types.append(zt)

        context_records.append({
            "view_layer_landscape": row.get(view_col),
            "noise_night": row.get(noise_col),
            "layout_window_perimeter": row.get(perimeter_col),
            "aif": row.get(aif_col),
        })

    polygons = [simplify_and_snap(p, zt, cfg) for p, zt in zip(polygons, zone_types)]
    polygons = normalize_polygons(polygons, cfg)
    polygons = close_room_gaps_morphological(
        polygons,
        zone_types,
        buffer_dist=cfg.gap_buffer_dist,
        shrink_ratio=cfg.gap_shrink_ratio,
    )
    before = sum(1 for z in zone_types if z == 17)
    polygons, zone_types, _, kept_idxs = filter_internal_doors_without_two_sided_room_adjacency(
        polygons, zone_types, win_orient=None,
        dist_tol=cfg.edge_adj_dist_tol, parallel_tol=cfg.parallel_tol,
        return_indices=True
    )
    context_records = [context_records[i] for i in kept_idxs]

    after = sum(1 for z in zone_types if z == 17)
    log.debug("removed %d internal doors lacking two-sided room adjacency",
              before - after)


    # CANONICAL REORDER (rooms -> door17 -> entrance15 last)
    idxs = canonical_sort_indices(polygons, zone_types, win_orient=None, include_windows=False)
    context_records = [context_records[i] for i in idxs]
    polygons, zone_types, _ = apply_reorder(polygons, zone_types, None, idxs)

    orient = compute_orient_rooms_and_windows(polygons, zone_types, win_orient=None)

    contextual = compute_contextual_orientations(
        zone_types,
        orient,
        context_records,
    )

    polygons = [canonicalize_polygon(p, clockwise=True) for p in polygons]
    edges_full = extract_edges(polygons)
    ed_rm = assign_ed_rm(zone_types, edges_full,
                         dist_tol=cfg.edge_adj_dist_tol,
                         parallel_tol=cfg.parallel_tol)
    ed_rm, repaired_pairs = repair_room_room_reciprocity(zone_types, edges_full, ed_rm)
    if repaired_pairs:
        log.debug("repaired %d reciprocal room-room ed_rm mappings",
                  len(repaired_pairs))
    validate_ed_rm(zone_types, edges_full, ed_rm)
    edges = [[e[0], e[1], e[2], e[3]] for e in edges_full]

    data = {
        "zone_types": zone_types,
        "edges": edges,
        "ed_rm": ed_rm,
        "orient": orient,
        "AIF": contextual["AIF"],
        "VL": contextual["VL"],
        "NN": contextual["NN"],
    }

    if custom_filename:
        base_name = custom_filename
    else:
        base_name = os.path.splitext(os.path.basename(csv_path))[0] + "_no_windows.json"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, base_name)
    else:
        out_path = os.path.splitext(csv_path)[0] + "_no_windows.json"

    with open(out_path, "w") as f:
        json.dump(data, f, separators=(", ", ": "))

    log.info("wrote %s", out_path)

    if show_plot or plot_path is not None:
        plot_with_polygons(
            polygons,
            edges_full,
            ed_rm,
            zone_types,
            orient,
            save_path=plot_path,
            show=show_plot,
        )

    return data
