"""Layout-level environmental labels: AIF, VL, NN.

These are what make OHD-W "context-aware". Each is the orientation of the
single best space in the layout:

    VL   orientation of the space with the largest landscape view
    NN   orientation of the windowed space with the least night noise
    AIF  orientation of the windowed space with the highest afternoon
         illuminance factor

Each is 0 when no space qualifies. NN and AIF consider only spaces that
actually have window perimeter; VL does not require one.
"""
import numpy as np

from .config import NON_ROOM

__all__ = ["compute_contextual_orientations", "resolve_column"]


def _resolve_column(df, candidates, required=True):
    """Return the first matching column name."""
    if isinstance(candidates, str):
        candidates = (candidates,)
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Missing required column. Tried {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def compute_contextual_orientations(
    zone_types,
    orient_values,
    context_records,
):
    """
    Compute layout-level orientation labels:

      VL:
        orientation of the SPACE with maximum view_layer_landscape.

      NN:
        among SPACES with layout_window_perimeter > 0,
        orientation of the space with minimum noise_night.

      AIF:
        among SPACES with layout_window_perimeter > 0,
        orientation of the space with maximum Afternoon Illuminance Factor.

    context_records must be aligned 1:1 with zone_types/orient_values.
    Doors/windows are excluded from candidate spaces.

    Returns 0 if no valid eligible space exists.
    """
    spaces = []
    for i, (zt, ori, rec) in enumerate(
        zip(zone_types, orient_values, context_records)
    ):
        if zt in NON_ROOM:
            continue
        spaces.append((i, int(ori), rec))

    result = {"VL": 0, "NN": 0, "AIF": 0}

    # ---------------- VL: maximum landscape view ----------------
    vl_candidates = []
    for i, ori, rec in spaces:
        v = _safe_float(rec.get("view_layer_landscape"))
        if v is not None:
            vl_candidates.append((v, -i, ori))

    if vl_candidates:
        _, _, ori = max(vl_candidates)
        result["VL"] = int(ori)

    # ---------- NN / AIF: require non-zero window perimeter -----
    windowed_spaces = []
    for i, ori, rec in spaces:
        wp = _safe_float(rec.get("layout_window_perimeter"))
        if wp is not None and wp > 0:
            windowed_spaces.append((i, ori, rec))

    nn_candidates = []
    for i, ori, rec in windowed_spaces:
        v = _safe_float(rec.get("noise_night"))
        if v is not None:
            nn_candidates.append((v, i, ori))

    if nn_candidates:
        _, _, ori = min(nn_candidates)
        result["NN"] = int(ori)

    aif_candidates = []
    for i, ori, rec in windowed_spaces:
        v = _safe_float(rec.get("aif"))
        if v is not None:
            aif_candidates.append((v, -i, ori))

    if aif_candidates:
        _, _, ori = max(aif_candidates)
        result["AIF"] = int(ori)

    return result



# public alias for the notebook's private helper
resolve_column = _resolve_column
