"""Set up the Colab environment for a given model.

Everything here is idempotent: cells get re-run constantly in a classroom, so no
step may fail because it already succeeded.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from . import models

REPO_URL = "https://github.com/aminshabani/house_diffusion"
GUIDED_URL = "https://github.com/openai/guided-diffusion"

APT_PACKAGES = ["graphviz", "graphviz-dev"]
PIP_PACKAGES = ["drawsvg", "cairosvg", "mpi4py", "pytorch_fid", "pygraphviz", "pillow"]


def _run(cmd, **kw):
    """Run a shell command, streaming output, raising on failure."""
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def clone(root: Path, quiet: bool = True) -> Path:
    """Clone house_diffusion with guided-diffusion nested inside it."""
    root = Path(root)
    flag = "--quiet" if quiet else ""
    if not root.exists():
        _run(f"git clone {flag} {REPO_URL} {root}")
    else:
        print(f"  repo already present at {root}")

    guided = root / "guided-diffusion"
    if not guided.exists():
        tmp = root.parent / "guided-diffusion"
        if not tmp.exists():
            _run(f"git clone {flag} {GUIDED_URL} {tmp}")
        shutil.move(str(tmp), str(guided))
    else:
        print("  guided-diffusion already in place")
    return root


def install_dependencies(root: Path, quiet: bool = True) -> None:
    """Install system and Python dependencies, then the repo itself."""
    q = "-q" if quiet else ""
    if shutil.which("apt-get"):
        _run("apt-get update -qq")
        _run(f"apt-get install -y -qq {' '.join(APT_PACKAGES)}")
    _run(f"{sys.executable} -m pip install {q} {' '.join(PIP_PACKAGES)}")
    _run(f"{sys.executable} -m pip install {q} -e .", cwd=str(root))


def place_files(spec: models.ModelSpec, root: Path, drive_dir: Path) -> None:
    """Copy patched sources, checkpoint and preprocessed data into the repo.

    Missing files stop the run immediately -- a missing checkpoint is not
    something to warn about and continue past.
    """
    root, drive_dir = Path(root), Path(drive_dir)
    source = drive_dir / spec.drive_subdir
    if not source.is_dir():
        raise FileNotFoundError(
            f"Model files not found: {source}\n"
            f"Check that you made a Drive shortcut named {spec.drive_subdir!r} "
            f"inside {drive_dir}."
        )

    targets = {
        root / "house_diffusion": spec.pkg_files,
        root / "scripts": spec.scripts,
        root / "scripts" / "ckpts" / "exp": [spec.checkpoint],
        root / "scripts" / spec.processed_dir: spec.npz_files,
    }
    (root / "datasets" / spec.dataset).mkdir(parents=True, exist_ok=True)

    missing = [
        f for files in targets.values() for f in files if not (source / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"These files are missing from {source}:\n  " + "\n  ".join(missing)
        )

    for dest, files in targets.items():
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copy2(source / name, dest / name)
        print(f"  {len(files):>2} file(s) -> {dest.relative_to(root)}")


def clear_eval_cache(spec: models.ModelSpec, root: Path) -> int:
    """Delete the cached eval/syn .npz files for this dataset.

    The sampling scripts cache the encoded eval set next to the model. Swap the
    dataset without clearing it and the next run silently samples the *previous*
    dataset. The notebook says "do not forget to delete the cache"; this does it.
    """
    cache = Path(root) / "scripts" / spec.processed_dir
    if not cache.is_dir():
        return 0
    stale = [p for p in cache.glob("*.npz")
             if "_eval_" in p.name or p.name.endswith("_syn.npz")]
    for p in stale:
        p.unlink()
    return len(stale)


def place_split(spec: models.ModelSpec, root: Path, split_dir,
                replace: bool = True) -> int:
    """Copy a prepared split into datasets/<dataset> and return its file count.

    `datasets/<dataset>` holds whichever split is in use -- train for training,
    test for sampling -- so by default this replaces the contents rather than
    merging into them. Merging a test split on top of a train split is how a
    run ends up scoring against data it was trained on.
    """
    src = Path(split_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Split directory not found: {src}")
    if not any(src.glob("*.json")):
        raise FileNotFoundError(f"No JSON layouts in {src}. Run prepare() first.")

    dst = Path(root) / "datasets" / spec.dataset
    if replace and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

    if not (dst / "list.txt").exists():
        from .convert.splits import write_list_txt
        write_list_txt(dst)
    return len(list(dst.glob("*.json")))


def place_testset(spec: models.ModelSpec, root: Path, drive_dir: Path) -> int:
    """Copy the JSON test set in and return how many files are present."""
    src = Path(drive_dir) / spec.testset_subdir
    dst = Path(root) / "datasets" / spec.dataset
    if not src.is_dir():
        raise FileNotFoundError(
            f"Test set not found: {src}\n"
            f"Expected a folder named {spec.testset_subdir!r} inside {drive_dir}."
        )
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return len(list(dst.iterdir()))
