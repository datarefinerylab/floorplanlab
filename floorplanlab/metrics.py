"""Parse GED / OA out of the sampling script's stdout.

The script prints per-round lines and a final summary:

    sampling complete
    Compatibility: 6.5625
    OA: 0.9098772321428572
    ...
    Compatibility mean: 6.6875 <TAB> Compatibility std: 0.10597390598633231
    OA mean: 0.9204799107142858 <TAB> OA std: 0.008244965875777126

Note the script says "Compatibility" where the tutorials say "GED"; we expose it
as `ged` and keep `compatibility` as an alias so both vocabularies work.
"""
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import List, Optional

_ROUND_GED = re.compile(r"^Compatibility:\s*([\d.eE+-]+)\s*$", re.M)
_ROUND_OA = re.compile(r"^OA:\s*([\d.eE+-]+)\s*$", re.M)
_SUMMARY_GED = re.compile(
    r"Compatibility mean:\s*([\d.eE+-]+)\s+Compatibility std:\s*([\d.eE+-]+)"
)
_SUMMARY_OA = re.compile(r"OA mean:\s*([\d.eE+-]+)\s+OA std:\s*([\d.eE+-]+)")


@dataclass
class Metrics:
    """Graph edit distance and orientation alignment for one sampling run."""

    ged: Optional[float] = None
    ged_std: Optional[float] = None
    oa: Optional[float] = None
    oa_std: Optional[float] = None
    ged_rounds: List[float] = field(default_factory=list)
    oa_rounds: List[float] = field(default_factory=list)

    # the sampling script's own name for GED
    @property
    def compatibility(self) -> Optional[float]:
        return self.ged

    def rounds_match(self) -> bool:
        """True when a GED round was recorded alongside every OA round."""
        return len(self.ged_rounds) == len(self.oa_rounds)

    @property
    def complete(self) -> bool:
        return self.ged is not None and self.oa is not None

    def as_dict(self) -> dict:
        return {
            "ged": self.ged,
            "ged_std": self.ged_std,
            "oa": self.oa,
            "oa_std": self.oa_std,
            "rounds": len(self.ged_rounds),
        }

    def __str__(self) -> str:
        if not self.complete:
            return "Metrics(not found in output)"
        return (
            f"GED {self.ged:.3f} +/- {self.ged_std:.3f}   "
            f"OA {self.oa:.4f} +/- {self.oa_std:.4f}   "
            f"({len(self.ged_rounds)} rounds)"
        )


def parse(stdout: str) -> Metrics:
    """Extract metrics from a sampling run's stdout.

    Falls back to averaging the per-round lines when the summary is absent, so an
    interrupted run still yields usable numbers.
    """
    m = Metrics()
    m.ged_rounds = [float(x) for x in _ROUND_GED.findall(stdout)]
    m.oa_rounds = [float(x) for x in _ROUND_OA.findall(stdout)]

    summary_ged = _SUMMARY_GED.search(stdout)
    if summary_ged:
        m.ged, m.ged_std = float(summary_ged.group(1)), float(summary_ged.group(2))
    elif m.ged_rounds:
        m.ged, m.ged_std = mean(m.ged_rounds), pstdev(m.ged_rounds)

    summary_oa = _SUMMARY_OA.search(stdout)
    if summary_oa:
        m.oa, m.oa_std = float(summary_oa.group(1)), float(summary_oa.group(2))
    elif m.oa_rounds:
        m.oa, m.oa_std = mean(m.oa_rounds), pstdev(m.oa_rounds)

    return m
