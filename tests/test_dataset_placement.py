"""Tests for installing a prepared split into datasets/<dataset>.

`datasets/<dataset>` holds one split at a time. The failure this guards against
is a test split landing on top of a train split, so a run scores against layouts
the model was trained on.
"""
import json

import pytest

from floorplanlab import env, models


def _split(directory, names, key="a"):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(json.dumps({"zone_types": [1], "tag": key}))
    return directory


@pytest.fixture
def spec():
    return models.get("ohdw-oesd")


def test_split_lands_in_the_dataset_dir(fake_repo, spec, tmp_path):
    src = _split(tmp_path / "train", ["0.json", "1.json", "2.json"])
    count = env.place_split(spec, fake_repo, src)
    dst = fake_repo / "datasets" / spec.dataset
    assert count == 3
    assert sorted(p.name for p in dst.glob("*.json")) == ["0.json", "1.json", "2.json"]


def test_list_txt_is_written_when_the_split_lacks_one(fake_repo, spec, tmp_path):
    """Sampling fails confusingly without list.txt, so never leave it missing."""
    src = _split(tmp_path / "test", ["0.json", "1.json"])
    env.place_split(spec, fake_repo, src)
    listing = (fake_repo / "datasets" / spec.dataset / "list.txt").read_text()
    assert sorted(listing.split()) == ["0.json", "1.json"]


def test_a_second_split_replaces_the_first(fake_repo, spec, tmp_path):
    train = _split(tmp_path / "train", ["0.json", "1.json", "2.json"], key="train")
    test = _split(tmp_path / "test", ["9.json"], key="test")
    env.place_split(spec, fake_repo, train)
    env.place_split(spec, fake_repo, test)
    dst = fake_repo / "datasets" / spec.dataset
    assert [p.name for p in dst.glob("*.json")] == ["9.json"]
    assert json.loads((dst / "9.json").read_text())["tag"] == "test"


def test_merging_is_possible_but_not_the_default(fake_repo, spec, tmp_path):
    env.place_split(spec, fake_repo, _split(tmp_path / "a", ["0.json"]))
    env.place_split(spec, fake_repo, _split(tmp_path / "b", ["1.json"]), replace=False)
    dst = fake_repo / "datasets" / spec.dataset
    assert sorted(p.name for p in dst.glob("*.json")) == ["0.json", "1.json"]


def test_an_empty_split_is_refused(fake_repo, spec, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="prepare"):
        env.place_split(spec, fake_repo, empty)


def test_a_missing_split_is_refused(fake_repo, spec, tmp_path):
    with pytest.raises(FileNotFoundError):
        env.place_split(spec, fake_repo, tmp_path / "nope")


# --- the stale-cache trap the notebook warns about in prose ---------------

def test_eval_cache_is_cleared(fake_repo, spec):
    cache = fake_repo / "scripts" / spec.processed_dir
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("oesd_eval_6.npz", "oesd_eval_6_syn.npz",
                 "oesd_train_all_cndist.npz"):
        (cache / name).write_text("cached")
    assert env.clear_eval_cache(spec, fake_repo) == 2
    assert [p.name for p in cache.glob("*.npz")] == ["oesd_train_all_cndist.npz"]


def test_clearing_a_cache_that_does_not_exist_is_fine(fake_repo, spec):
    assert env.clear_eval_cache(spec, fake_repo) == 0
