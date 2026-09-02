"""Match SVG files across two output folders.

The tutorials use two different naming conventions:

    T4   gt/1c_0_gt.svg        pred/1c_0_pred.svg
    T5   graphs_gt/20.svg      pred/20c_0_pred.svg

Rather than hardcode one regex per notebook (T4 and T5 both define a function
called `build_pairs` with incompatible rules), we strip the known role suffixes
and match on the remaining stem. If that finds nothing, we retry on the leading
number, which bridges the T5 case.

Collisions are reported, never silently resolved. The tutorials' `setdefault`
approach keeps whichever file the filesystem happened to list first, which can
silently pair a target-6 plot with a leftover target-8 layout.
"""
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

_ROLE_SUFFIX = re.compile(r"_(gt|pred)$")
_LEADING_NUM = re.compile(r"^(\d+)")


class Pair(NamedTuple):
    key: str
    left: Path
    right: Path


class PairResult(NamedTuple):
    pairs: List[Pair]
    left_only: List[str]
    right_only: List[str]
    collisions: Dict[str, List[str]]
    strategy: str

    def summary(self) -> str:
        lines = [f"Matched {len(self.pairs)} pairs (matched on {self.strategy})."]
        if self.left_only:
            lines.append(f"  {len(self.left_only)} left-only, e.g. {self.left_only[:5]}")
        if self.right_only:
            lines.append(f"  {len(self.right_only)} right-only, e.g. {self.right_only[:5]}")
        if self.collisions:
            lines.append(
                f"  WARNING: {len(self.collisions)} key(s) matched multiple files; "
                "these were skipped rather than guessed. "
                f"e.g. {list(self.collisions.items())[:2]}"
            )
            lines.append(
                "  This usually means an output folder holds results from more "
                "than one run. Use a fresh run directory."
            )
        return "\n".join(lines)


def _stem_key(path: Path) -> str:
    return _ROLE_SUFFIX.sub("", path.stem)


def _number_key(path: Path) -> str:
    m = _LEADING_NUM.match(path.stem)
    return m.group(1) if m else ""


def _index(paths, keyfn) -> Tuple[Dict[str, Path], Dict[str, List[str]]]:
    index: Dict[str, Path] = {}
    dupes: Dict[str, List[str]] = {}
    for p in paths:
        k = keyfn(p)
        if not k:
            continue
        if k in index:
            dupes.setdefault(k, [index[k].name]).append(p.name)
        else:
            index[k] = p
    for k in dupes:
        index.pop(k, None)          # ambiguous -> drop, do not guess
    return index, dupes


def build_pairs(left_dir, left_glob="*.svg", right_dir=None, right_glob="*.svg") -> PairResult:
    """Pair SVGs in two directories, trying stem matching then number matching."""
    left_dir, right_dir = Path(left_dir), Path(right_dir)
    for d in (left_dir, right_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Not a directory: {d}")

    left_files = sorted(left_dir.glob(left_glob))
    right_files = sorted(right_dir.glob(right_glob))

    last_collisions: Dict[str, List[str]] = {}
    for strategy, keyfn in (("full stem", _stem_key), ("leading number", _number_key)):
        lmap, ldupes = _index(left_files, keyfn)
        rmap, rdupes = _index(right_files, keyfn)
        last_collisions = {**ldupes, **rdupes}
        shared = sorted(set(lmap) & set(rmap), key=lambda k: (len(k), k))
        if shared:
            return PairResult(
                pairs=[Pair(k, lmap[k], rmap[k]) for k in shared],
                left_only=sorted(set(lmap) - set(rmap)),
                right_only=sorted(set(rmap) - set(lmap)),
                collisions=last_collisions,
                strategy=strategy,
            )

    # Nothing matched under either rule; keep the collisions so the caller can
    # see whether ambiguity (rather than a naming mismatch) was the cause.
    return PairResult([], sorted(p.name for p in left_files),
                      sorted(p.name for p in right_files), last_collisions,
                      "no match")
