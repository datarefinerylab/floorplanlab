"""Train / test / deploy splits for converted layouts.

Consolidates three notebook cells (per-target split, all-targets split, and the
separate `list.txt` cell) into one function.

The key asymmetry, kept from the notebook: train and test hold the full
geometric JSON, while **deploy holds only the conditions known before a layout
exists** -- no boxes, no edges. Deploy is generation from a brief; test is
reconstruction of a known plan.

`list.txt` is written automatically. In the notebook it lived in its own cell
that had to be run once per split directory, and sampling fails confusingly
when it is missing.
"""
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

log = logging.getLogger(__name__)

__all__ = ["SplitResult", "make_splits", "write_list_txt"]

# Keys the full (train/test) records must carry.
FULL_KEYS = ("zone_types", "edges", "ed_rm", "orient", "AIF", "VL", "NN")
# Keys a deploy record keeps: conditions available before generation.
DEPLOY_KEYS = ("zone_types", "orient", "ed_rm", "AIF", "VL", "NN")


@dataclass
class SplitResult:
    root: Path
    counts: Dict[str, int]
    manifest_path: Path

    def __str__(self) -> str:
        parts = ", ".join(f"{k} {v}" for k, v in self.counts.items())
        return f"{parts}  ->  {self.root}"


def write_list_txt(directory) -> Path:
    """Write the list.txt manifest the sampling scripts require."""
    directory = Path(directory)
    names = sorted(p.name for p in directory.glob("*.json") if p.name != "list.txt")
    out = directory / "list.txt"
    out.write_text("".join(n + "\n" for n in names))
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_full(paths: Sequence[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        data = _load(p)
        missing = [k for k in FULL_KEYS if k not in data]
        if missing:
            raise KeyError(f"{p.name}: missing required keys {missing}")
        with open(dst / p.name, "w") as f:
            json.dump({k: data[k] for k in FULL_KEYS}, f, separators=(", ", ": "))


def _write_deploy(paths: Sequence[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        data = _load(p)
        record = {
            "zone_types": data["zone_types"],
            "orient": data["orient"],
            "ed_rm": data["ed_rm"],
            "AIF": int(data.get("AIF", 0)),
            "VL": int(data.get("VL", 0)),
            "NN": int(data.get("NN", 0)),
        }
        with open(dst / p.name, "w") as f:
            json.dump(record, f, separators=(", ", ": "))


def make_splits(
    inputs,
    out_root,
    test_frac: float = 0.10,
    deploy_frac: float = 0.05,
    seed: int = 123,
) -> SplitResult:
    """Split converted JSONs into train/test/deploy.

    `inputs` is either one directory, or a mapping of target size -> directory
    to pool several target sizes into one dataset. Splitting is done *within*
    each target size so every split keeps the same size mix.

    `deploy_frac` is a fraction of the whole, not of the remainder -- the
    notebook's per-category cell took it from the remainder while its
    all-targets cell took it from the whole, which made the two disagree.
    """
    if not 0 <= test_frac < 1 or not 0 <= deploy_frac < 1:
        raise ValueError("test_frac and deploy_frac must each be in [0, 1)")
    if test_frac + deploy_frac >= 1:
        raise ValueError("test_frac + deploy_frac must leave something for train")

    if not isinstance(inputs, Mapping):
        inputs = {None: inputs}

    out_root = Path(out_root)
    dirs = {name: out_root / name for name in ("train", "test", "deploy")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Path]] = {"train": [], "test": [], "deploy": []}
    per_target: Dict[str, Dict[str, int]] = {}

    for target, directory in sorted(inputs.items(), key=lambda kv: str(kv[0])):
        files = sorted(Path(directory).glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No .json files in {directory}")
        rng = random.Random(seed)
        rng.shuffle(files)

        n_test = int(round(len(files) * test_frac))
        n_deploy = int(round(len(files) * deploy_frac))
        test, deploy = files[:n_test], files[n_test:n_test + n_deploy]
        train = files[n_test + n_deploy:]

        buckets["test"] += test
        buckets["deploy"] += deploy
        buckets["train"] += train
        per_target[str(target)] = {
            "train": len(train), "test": len(test), "deploy": len(deploy)
        }

    _write_full(buckets["train"], dirs["train"])
    _write_full(buckets["test"], dirs["test"])
    _write_deploy(buckets["deploy"], dirs["deploy"])

    for d in dirs.values():
        write_list_txt(d)

    manifest = {
        "seed": seed,
        "test_frac": test_frac,
        "deploy_frac": deploy_frac,
        "inputs": {str(k): str(v) for k, v in inputs.items()},
        "per_target": per_target,
        "files": {k: sorted(p.name for p in v) for k, v in buckets.items()},
    }
    manifest_path = out_root / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    counts = {k: len(v) for k, v in buckets.items()}
    log.info("splits written: %s", counts)
    return SplitResult(out_root, counts, manifest_path)
