"""Tests for the CSV -> JSON converter.

The golden files in tests/golden/ were produced by the ORIGINAL notebook cell,
so they pin the extracted package to the behaviour it was extracted from. If a
change here is deliberate, regenerate them and say so in the commit.
"""
import json
from pathlib import Path

import pytest

from floorplanlab.convert import (ConvertConfig, angle_to_8way_label,
                                  compute_contextual_orientations, csv_to_json)
from floorplanlab.convert.config import DEFAULT

GOLDEN = Path(__file__).parent / "golden"
CASES = sorted(p.stem for p in GOLDEN.glob("*.csv"))


@pytest.mark.parametrize("case", CASES)
def test_matches_notebook_output(case, tmp_path):
    """Extracted converter reproduces the original cell exactly."""
    expected = json.loads((GOLDEN / f"{case}.json").read_text())
    actual = csv_to_json(str(GOLDEN / f"{case}.csv"), show_plot=False,
                         output_dir=str(tmp_path))
    assert actual == expected


@pytest.mark.parametrize("case", CASES)
def test_output_is_internally_consistent(case, tmp_path):
    """Structural invariants the downstream model relies on."""
    d = csv_to_json(str(GOLDEN / f"{case}.csv"), show_plot=False,
                    output_dir=str(tmp_path))
    n = len(d["zone_types"])

    assert len(d["orient"]) == n, "one orientation per component"
    assert len(d["edges"]) == len(d["ed_rm"]), "one ed_rm entry per edge"

    # every ed_rm index must address a real component
    for mapping in d["ed_rm"]:
        for idx in mapping:
            assert -1 <= idx < n

    # doors carry no orientation; rooms and windows do
    for zt, ori in zip(d["zone_types"], d["orient"]):
        if zt in (15, 17):
            assert ori == 0, "doors must have orient 0"
        assert 0 <= ori <= 8

    # canonical order: entrance door last, rooms before non-rooms
    if 15 in d["zone_types"]:
        assert d["zone_types"][-1] == 15, "entrance door must be last"
    rooms = [i for i, z in enumerate(d["zone_types"]) if z not in (15, 16, 17)]
    others = [i for i, z in enumerate(d["zone_types"]) if z in (15, 16, 17)]
    if rooms and others:
        assert max(rooms) < min(others), "rooms come before doors/windows"

    for key in ("AIF", "VL", "NN"):
        assert 0 <= d[key] <= 8


def test_config_is_honoured(tmp_path):
    """A non-default config must actually change the result."""
    csv = str(GOLDEN / "6rooms_windows.csv")
    base = csv_to_json(csv, show_plot=False, output_dir=str(tmp_path))
    wide = csv_to_json(csv, show_plot=False, output_dir=str(tmp_path),
                       cfg=ConvertConfig(margin=60))
    assert base["edges"] != wide["edges"], "margin change must move geometry"
    assert base["zone_types"] == wide["zone_types"], "but not change components"


# --- orientation ---------------------------------------------------------

@pytest.mark.parametrize("angle,expected", [
    (0, 1), (45, 2), (90, 3), (135, 4),
    (180, 5), (-180, 5), (-135, 6), (-90, 7), (-45, 8),
])
def test_angle_to_8way_cardinals_and_diagonals(angle, expected):
    assert angle_to_8way_label(angle) == expected


def test_angle_wraps_around():
    assert angle_to_8way_label(360) == angle_to_8way_label(0)
    assert angle_to_8way_label(-720 + 90) == angle_to_8way_label(90)


def test_every_angle_classifies():
    """No angle may fall through the band logic."""
    for deg in range(-360, 361):
        assert 1 <= angle_to_8way_label(float(deg)) <= 8


# --- environmental context (AIF / VL / NN) -------------------------------

def _records(*rows):
    keys = ("view_layer_landscape", "noise_night", "layout_window_perimeter", "aif")
    return [dict(zip(keys, r)) for r in rows]


def test_context_picks_best_space():
    zone_types = [1, 2, 3]
    orient = [1, 5, 7]
    recs = _records(
        (0.1, 50.0, 2.0, 0.2),   # dim, noisy, poor view
        (0.9, 30.0, 2.0, 0.8),   # best view, quietest, brightest
        (0.5, 40.0, 2.0, 0.5),
    )
    got = compute_contextual_orientations(zone_types, orient, recs)
    assert got == {"VL": 5, "NN": 5, "AIF": 5}


def test_context_requires_windows_for_aif_and_nn():
    """A space with no window perimeter cannot win AIF or NN, but can win VL."""
    zone_types = [1, 2]
    orient = [3, 6]
    recs = _records(
        (0.9, 10.0, 0.0, 0.99),  # best on everything but has NO window
        (0.1, 50.0, 4.0, 0.10),
    )
    got = compute_contextual_orientations(zone_types, orient, recs)
    assert got["VL"] == 3, "VL does not require a window"
    assert got["NN"] == 6, "NN must ignore the window-less space"
    assert got["AIF"] == 6, "AIF must ignore the window-less space"


def test_context_ignores_doors_and_windows():
    zone_types = [1, 17, 16, 15]
    orient = [2, 0, 4, 0]
    recs = _records(
        (0.2, 40.0, 3.0, 0.2),
        (0.99, 1.0, 9.0, 0.99),   # a door with absurdly good numbers
        (0.99, 1.0, 9.0, 0.99),   # a window likewise
        (0.99, 1.0, 9.0, 0.99),
    )
    got = compute_contextual_orientations(zone_types, orient, recs)
    assert got == {"VL": 2, "NN": 2, "AIF": 2}


def test_context_returns_zero_when_nothing_eligible():
    got = compute_contextual_orientations([1], [3], _records((None, None, 0.0, None)))
    assert got == {"VL": 0, "NN": 0, "AIF": 0}


def test_context_survives_missing_and_nan_values():
    recs = _records(
        (float("nan"), None, 2.0, "not a number"),
        (0.5, 20.0, 2.0, 0.7),
    )
    got = compute_contextual_orientations([1, 2], [1, 5], recs)
    assert got == {"VL": 5, "NN": 5, "AIF": 5}


# --- config validation ---------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(snap_iou_threshold=1.5),
    dict(snap_iou_threshold=-0.1),
    dict(room_diagonal_min_deg=70.0),
    dict(margin=200),
    dict(margin=-1),
])
def test_config_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        ConvertConfig(**kwargs)


def test_config_tolerance_depends_on_component_class():
    assert DEFAULT.tol_for(16) == DEFAULT.simplify_tol_doorwin
    assert DEFAULT.tol_for(1) == DEFAULT.simplify_tol
