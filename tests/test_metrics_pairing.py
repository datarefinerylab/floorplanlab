"""Tests for metric parsing and SVG pairing.

The metrics cases use stdout captured from the real T5 notebook run, so the
parser is pinned to output the actual sampling script produced. That stdout
lives in fixtures/ so this file does not need the 19.5 MB notebook; when the
notebook *is* present, test_fixtures_match_the_notebook checks it has not
drifted from the source it was copied out of.
"""
import json
from pathlib import Path

import pytest

from floorplanlab import metrics, pairing

FIXTURES = Path(__file__).parent / "fixtures"
NOTEBOOK = (Path(__file__).parents[1] /
            "Copy_of_T5_Oriented_HouseDiffusion_for_floor_layout_generation_O_ESD.ipynb")

# (fixture, source cell index, expected GED, expected OA) -- as published in the notebook
REAL_RUNS = [("t5_target8_stdout.txt", 29, 6.6875, 0.9204799107142858),
             ("t5_target6_stdout.txt", 44, 3.834375, 0.8709374999999999)]


def _cell_stdout(index):
    nb = json.loads(NOTEBOOK.read_text())
    outs = nb["cells"][index].get("outputs", [])
    return "".join("".join(o.get("text", [])) for o in outs)


@pytest.mark.parametrize("fixture,index,ged,oa", REAL_RUNS)
def test_parses_real_notebook_output(fixture, index, ged, oa):
    m = metrics.parse((FIXTURES / fixture).read_text())
    assert m.ged == pytest.approx(ged)
    assert m.oa == pytest.approx(oa)
    assert m.rounds_match()


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="source notebook not present")
@pytest.mark.parametrize("fixture,index,ged,oa", REAL_RUNS)
def test_fixtures_match_the_notebook(fixture, index, ged, oa):
    """The fixtures are copies -- fail if they no longer match the notebook."""
    assert (FIXTURES / fixture).read_text() == _cell_stdout(index)


def test_falls_back_to_round_averages_when_interrupted():
    """An interrupted run has no summary line but still has usable rounds."""
    partial = ("sampling complete\nCompatibility: 4.0\nOA: 0.9\n"
               "sampling complete\nCompatibility: 4.4\nOA: 0.8\n")
    m = metrics.parse(partial)
    assert m.ged == pytest.approx(4.2)
    assert m.oa == pytest.approx(0.85)
    assert len(m.ged_rounds) == 2


def test_absent_metrics_are_reported_not_invented():
    m = metrics.parse("training finished, nothing to score")
    assert not m.complete
    assert m.ged is None and m.oa is None


def test_compatibility_is_an_alias_for_ged():
    """The script prints 'Compatibility'; the tutorials say 'GED'."""
    m = metrics.parse("Compatibility mean: 3.0 \t Compatibility std: 0.1\n"
                      "OA mean: 0.9 \t OA std: 0.01\n")
    assert m.compatibility == m.ged == 3.0


# --- pairing --------------------------------------------------------------

def _dir(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir()
    for f in files:
        (d / f).write_text("<svg/>")
    return d


def test_t4_naming_convention(tmp_path):
    left = _dir(tmp_path, "gt", ["1c_0_gt.svg", "2c_0_gt.svg"])
    right = _dir(tmp_path, "pred", ["1c_0_pred.svg", "2c_0_pred.svg"])
    r = pairing.build_pairs(left, right_dir=right)
    assert [p.key for p in r.pairs] == ["1c_0", "2c_0"]
    assert r.strategy == "full stem"


def test_t5_naming_convention(tmp_path):
    left = _dir(tmp_path, "graphs_gt", ["20.svg", "21.svg"])
    right = _dir(tmp_path, "pred", ["20c_0_pred.svg", "21c_0_pred.svg"])
    r = pairing.build_pairs(left, right_dir=right)
    assert [p.key for p in r.pairs] == ["20", "21"]
    assert r.strategy == "leading number"


def test_ambiguous_matches_are_refused_not_guessed(tmp_path):
    """Two runs merged into one folder must not silently pair the wrong file."""
    left = _dir(tmp_path, "graphs_gt", ["20.svg"])
    right = _dir(tmp_path, "pred", ["20c_0_pred.svg", "20c_1_pred.svg"])
    r = pairing.build_pairs(left, right_dir=right)
    assert r.pairs == []
    assert "20" in r.collisions
    assert "more than one run" in r.summary()


def test_partial_overlap_is_reported(tmp_path):
    left = _dir(tmp_path, "graphs_gt", ["20.svg", "21.svg", "99.svg"])
    right = _dir(tmp_path, "pred", ["20c_0_pred.svg"])
    r = pairing.build_pairs(left, right_dir=right)
    assert len(r.pairs) == 1
    assert sorted(r.left_only) == ["21", "99"]


def test_missing_directory_is_an_error(tmp_path):
    left = _dir(tmp_path, "gt", ["1_gt.svg"])
    with pytest.raises(FileNotFoundError):
        pairing.build_pairs(left, right_dir=tmp_path / "nope")
