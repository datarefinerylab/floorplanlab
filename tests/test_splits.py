"""Tests for train/test/deploy splitting."""
import json
from pathlib import Path

import pytest

from floorplanlab.convert import make_splits
from floorplanlab.convert.splits import DEPLOY_KEYS, FULL_KEYS


def _record(i):
    return {
        "zone_types": [1, 2, 17, 15], "edges": [[0, 0, 1, 1]] * 4,
        "ed_rm": [[0], [1], [2, 0], [3]], "orient": [1, 5, 0, 0],
        "AIF": (i % 8) + 1, "VL": (i % 8) + 1, "NN": (i % 8) + 1,
    }


def _populate(directory, n, start=0):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        (directory / f"layout_{i:04d}.json").write_text(json.dumps(_record(i)))
    return directory


def test_split_counts_and_disjointness(tmp_path):
    src = _populate(tmp_path / "in", 100)
    res = make_splits(src, tmp_path / "out", test_frac=0.10, deploy_frac=0.05)

    assert res.counts == {"train": 85, "test": 10, "deploy": 5}

    names = {k: {p.name for p in (tmp_path / "out" / k).glob("*.json")}
             for k in ("train", "test", "deploy")}
    assert not names["train"] & names["test"]
    assert not names["train"] & names["deploy"]
    assert not names["test"] & names["deploy"]
    assert sum(len(v) for v in names.values()) == 100


def test_deploy_records_drop_geometry(tmp_path):
    """Deploy inputs must contain only what is known before generation."""
    src = _populate(tmp_path / "in", 40)
    make_splits(src, tmp_path / "out")

    for p in (tmp_path / "out" / "deploy").glob("*.json"):
        keys = set(json.loads(p.read_text()))
        assert keys == set(DEPLOY_KEYS)
        assert "edges" not in keys, "deploy must not leak ground-truth geometry"

    for p in (tmp_path / "out" / "train").glob("*.json"):
        assert set(json.loads(p.read_text())) == set(FULL_KEYS)


def test_split_is_reproducible(tmp_path):
    src = _populate(tmp_path / "in", 60)
    a = make_splits(src, tmp_path / "a", seed=7)
    b = make_splits(src, tmp_path / "b", seed=7)
    c = make_splits(src, tmp_path / "c", seed=8)

    def names(res, bucket):
        return sorted(p.name for p in (res.root / bucket).glob("*.json"))

    assert names(a, "test") == names(b, "test"), "same seed -> same split"
    assert names(a, "test") != names(c, "test"), "different seed -> different split"


def test_list_txt_written_for_every_split(tmp_path):
    """The sampling scripts fail confusingly without list.txt."""
    src = _populate(tmp_path / "in", 30)
    make_splits(src, tmp_path / "out")

    for bucket in ("train", "test", "deploy"):
        lst = tmp_path / "out" / bucket / "list.txt"
        assert lst.exists(), f"{bucket}/list.txt missing"
        listed = [l for l in lst.read_text().splitlines() if l]
        actual = sorted(p.name for p in (tmp_path / "out" / bucket).glob("*.json"))
        assert listed == actual
        assert "list.txt" not in listed


def test_multiple_target_sizes_are_stratified(tmp_path):
    """Each target size must appear in every split, in proportion."""
    inputs = {}
    for size, n in ((5, 100), (6, 100), (7, 100), (8, 100)):
        inputs[size] = _populate(tmp_path / f"t{size}", n, start=size * 1000)

    res = make_splits(inputs, tmp_path / "out", test_frac=0.10, deploy_frac=0.05)
    assert res.counts == {"train": 340, "test": 40, "deploy": 20}

    manifest = json.loads(res.manifest_path.read_text())
    for size in ("5", "6", "7", "8"):
        assert manifest["per_target"][size] == {"train": 85, "test": 10, "deploy": 5}


def test_manifest_records_provenance(tmp_path):
    src = _populate(tmp_path / "in", 20)
    res = make_splits(src, tmp_path / "out", seed=99)
    m = json.loads(res.manifest_path.read_text())
    assert m["seed"] == 99
    assert set(m["files"]) == {"train", "test", "deploy"}
    assert sum(len(v) for v in m["files"].values()) == 20


def test_missing_required_keys_are_rejected(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    (src / "bad.json").write_text(json.dumps({"zone_types": [1]}))  # no AIF/VL/NN
    with pytest.raises(KeyError, match="missing required keys"):
        make_splits(src, tmp_path / "out", test_frac=0.0, deploy_frac=0.0)


def test_empty_input_is_rejected(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(FileNotFoundError):
        make_splits(tmp_path / "in", tmp_path / "out")


@pytest.mark.parametrize("kw", [
    dict(test_frac=1.0), dict(deploy_frac=-0.1), dict(test_frac=0.6, deploy_frac=0.5),
])
def test_nonsense_fractions_rejected(tmp_path, kw):
    src = _populate(tmp_path / "in", 10)
    with pytest.raises(ValueError):
        make_splits(src, tmp_path / "out", **kw)
