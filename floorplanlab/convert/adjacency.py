"""Door-based adjacency: building, repairing and validating `ed_rm`.

`ed_rm` maps each polygon edge to the components it joins. Extracted verbatim
from the notebook; constants now come from ConvertConfig defaults.
"""
import math

from shapely.geometry import LineString

from .config import DEFAULT, NON_ROOM

__all__ = ["assign_ed_rm", "repair_room_room_reciprocity", "validate_ed_rm"]


def _dedupe_mapping(mapping):
    """Remove duplicate component ids inside one ed_rm entry, preserving order."""
    return list(dict.fromkeys(int(x) for x in mapping))


def assign_ed_rm(zone_types, edges,
                 dist_tol=DEFAULT.edge_adj_dist_tol,
                 parallel_tol=DEFAULT.parallel_tol):
    """
    Build ed_rm without duplicate ids.

    Important ownership rule:
      ed_rm[i][0] is always the owner of edges[i].

    Reciprocity policy:
      - room <-> room adjacency is enforced reciprocally;
      - internal-door long edges remain [door, room];
      - window / entrance long edges remain [component, room];
      - room edges are NOT overwritten merely to force [room, window] or
        [room, entrance], because that can destroy a genuine room-room
        shared-boundary mapping.

    This protects the geometry used for training while preserving the
    connectivity information required by HouseDiffusion.
    """
    ed_rm = [[int(e[4])] for e in edges]
    edge_lines = [
        LineString([(e[0], e[1]), (e[2], e[3])])
        for e in edges
    ]

    component_idxs = [
        i for i, zt in enumerate(zone_types)
        if zt in NON_ROOM
    ]
    room_idxs = [
        i for i, zt in enumerate(zone_types)
        if zt not in NON_ROOM
    ]

    # Defer room-room reciprocity until ALL doors/windows have been processed,
    # so later components cannot overwrite it.
    reciprocal_room_pairs = []

    for di in component_idxs:
        comp_edges = [ei for ei, e in enumerate(edges) if int(e[4]) == di]
        if len(comp_edges) < 2:
            continue

        lengths = [edge_lines[ei].length for ei in comp_edges]
        long_local = sorted(
            range(len(lengths)),
            key=lambda k: lengths[k],
            reverse=True
        )[:2]
        long_edges = [comp_edges[k] for k in long_local]

        connected_zones = []  # (zone_id, zone_edge_idx, component_edge_idx)

        for ei in long_edges:
            comp_line = edge_lines[ei]
            dx_comp = edges[ei][2] - edges[ei][0]
            dy_comp = edges[ei][3] - edges[ei][1]

            closest_zone = None
            min_dist = float("inf")

            for zei, e2 in enumerate(edges):
                zj = int(e2[4])
                if zj not in room_idxs:
                    continue

                if zone_types[di] in [15, 16]:
                    dist_tol_use = DEFAULT.edge_adj_dist_tol_entrance
                    parallel_tol_use = DEFAULT.parallel_tol_entrance
                else:
                    dist_tol_use = dist_tol
                    parallel_tol_use = parallel_tol

                dx2 = e2[2] - e2[0]
                dy2 = e2[3] - e2[1]
                norm1 = math.hypot(dx_comp, dy_comp)
                norm2 = math.hypot(dx2, dy2)

                if norm1 == 0 or norm2 == 0:
                    continue

                cos_angle = abs(
                    (dx_comp * dx2 + dy_comp * dy2) / (norm1 * norm2)
                )
                if cos_angle < parallel_tol_use:
                    continue

                d = comp_line.distance(edge_lines[zei])
                if d < min_dist and d < dist_tol_use:
                    min_dist = d
                    closest_zone = (zj, zei)

            if closest_zone is None:
                continue

            zone_id, zone_edge_idx = closest_zone

            # The component owns this edge.
            ed_rm[ei] = [di, zone_id]
            connected_zones.append((zone_id, zone_edge_idx, ei))

        # Internal door: its two room matches imply reciprocal room-room
        # adjacency on the corresponding room-owned edges.
        if zone_types[di] == 17:
            unique = []
            seen = set()
            for item in connected_zones:
                if item[0] not in seen:
                    unique.append(item)
                    seen.add(item[0])

            if len(unique) >= 2:
                (z1, e1, _), (z2, e2, _) = unique[:2]
                if z1 != z2:
                    reciprocal_room_pairs.append((z1, e1, z2, e2))

    # Apply room-room reciprocal mappings LAST so they cannot be overwritten
    # by later windows / entrance doors.
    for z1, e1, z2, e2 in reciprocal_room_pairs:
        ed_rm[e1] = [z1, z2]
        ed_rm[e2] = [z2, z1]

    return [_dedupe_mapping(m) for m in ed_rm]



def repair_room_room_reciprocity(zone_types, edges, ed_rm):
    """
    Ensure every room-room pair [a, b] has a reciprocal [b, a].

    Strategy:
      1. Find all room-room pairs currently encoded in ed_rm.
      2. For each missing reverse pair, look for an edge owned by room b.
      3. Prefer the edge geometrically closest and most parallel to the
         source edge owned by room a.
      4. Rewrite only a single-value owner entry [b] when possible, so we
         do not destroy an existing two-value adjacency.
      5. If no clean [b] edge exists, use the best b-owned edge as fallback.

    This preserves edge ownership: ed_rm[i][0] always equals edges[i][4].
    """
    room_set = {
        i for i, zt in enumerate(zone_types)
        if zt not in NON_ROOM
    }

    edge_lines = [
        LineString([(e[0], e[1]), (e[2], e[3])])
        for e in edges
    ]

    def edge_vec(e):
        return (e[2] - e[0], e[3] - e[1])

    def parallel_score(e1, e2):
        dx1, dy1 = edge_vec(e1)
        dx2, dy2 = edge_vec(e2)
        n1 = math.hypot(dx1, dy1)
        n2 = math.hypot(dx2, dy2)
        if n1 == 0 or n2 == 0:
            return -1.0
        return abs((dx1 * dx2 + dy1 * dy2) / (n1 * n2))

    def current_pairs():
        return {
            (int(m[0]), int(m[1]))
            for m in ed_rm
            if len(m) == 2
            and int(m[0]) in room_set
            and int(m[1]) in room_set
        }

    pairs = current_pairs()

    # Work from a snapshot because we will add reverse mappings.
    missing = [
        (a, b)
        for (a, b) in sorted(pairs)
        if (b, a) not in pairs
    ]

    repaired = []

    for a, b in missing:
        # Source edge(s) that encode [a, b].
        src_edge_ids = [
            i for i, m in enumerate(ed_rm)
            if len(m) == 2 and int(m[0]) == a and int(m[1]) == b
        ]
        if not src_edge_ids:
            continue

        # Candidate edges owned by b.
        b_edge_ids = [
            i for i, e in enumerate(edges)
            if int(e[4]) == b
        ]
        if not b_edge_ids:
            continue

        # Prefer untouched owner-only edges [b].
        clean_candidates = [
            i for i in b_edge_ids
            if len(ed_rm[i]) == 1 and int(ed_rm[i][0]) == b
        ]
        candidates = clean_candidates if clean_candidates else b_edge_ids

        best = None
        best_key = None

        for src_i in src_edge_ids:
            for cand_i in candidates:
                dist = edge_lines[src_i].distance(edge_lines[cand_i])
                par = parallel_score(edges[src_i], edges[cand_i])

                # Lower distance is better; higher parallel score is better.
                key = (dist, -par)

                if best_key is None or key < best_key:
                    best_key = key
                    best = cand_i

        if best is not None:
            ed_rm[best] = [b, a]
            repaired.append((a, b, best))

    return [_dedupe_mapping(m) for m in ed_rm], repaired


def validate_ed_rm(zone_types, edges, ed_rm):
    """
    Validate the invariants that matter for training geometry:

      1. one ed_rm entry per geometric edge;
      2. first id equals the edge owner;
      3. no duplicate ids such as [3, 3];
      4. every ROOM-ROOM pair is reciprocal.

    Door/window reciprocity is intentionally NOT required because ed_rm is an
    edge-ownership map. Requiring [room, window] for every [window, room] can
    overwrite a room's genuine shared-boundary adjacency.
    """
    if len(edges) != len(ed_rm):
        raise ValueError(
            f"edges and ed_rm length mismatch: {len(edges)} != {len(ed_rm)}"
        )

    for ei, (e, m) in enumerate(zip(edges, ed_rm)):
        if not m:
            raise ValueError(f"Empty ed_rm at edge {ei}")

        owner = int(e[4])
        if int(m[0]) != owner:
            raise ValueError(
                f"Owner mismatch at edge {ei}: edge owner={owner}, ed_rm={m}"
            )

        if len(m) != len(set(m)):
            raise ValueError(f"Duplicate ids remain in ed_rm[{ei}]={m}")

    room_set = {
        i for i, zt in enumerate(zone_types)
        if zt not in NON_ROOM
    }

    room_pairs = {
        (int(m[0]), int(m[1]))
        for m in ed_rm
        if len(m) == 2
        and int(m[0]) in room_set
        and int(m[1]) in room_set
    }

    missing = [
        (a, b) for (a, b) in sorted(room_pairs)
        if (b, a) not in room_pairs
    ]

    if missing:
        raise ValueError(
            "Missing reciprocal ROOM-ROOM ed_rm pairs: "
            + ", ".join(
                f"[{a},{b}] missing [{b},{a}]"
                for a, b in missing[:20]
            )
        )


