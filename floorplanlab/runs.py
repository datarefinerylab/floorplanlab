"""Run the sampling script and hold onto its results.

Each call gets its own output directory. The tutorials reuse a single
`scripts/outputs/` folder across runs, which lets a later run's plots pick up an
earlier run's SVGs.
"""
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import metrics as _metrics
from . import models, pairing, view
from .stages import Stage, StageSpec


@dataclass
class Run:
    """One stage run: where its files are, and how it scored."""

    spec: models.ModelSpec
    stage: Stage
    target_set: int
    num_samples: int
    out_dir: Path
    metrics: _metrics.Metrics
    log_path: Path
    seconds: float

    @property
    def label(self) -> str:
        suffix = "" if self.stage is Stage.TEST else f" [{self.stage}]"
        return f"{self.spec.label} / target {self.target_set}{suffix}"

    def dirs(self) -> List[str]:
        """Which SVG folders this run actually produced."""
        return sorted(d.name for d in self.out_dir.iterdir() if d.is_dir())

    def pairs(self, view_name: Optional[str] = None):
        """Match SVGs for one of the model's named views."""
        view_name = view_name or self.spec.default_view
        if view_name not in self.spec.views:
            raise KeyError(
                f"{self.spec.key} has no view {view_name!r}. "
                f"Available: {', '.join(self.spec.views)}"
            )
        left, right = self.spec.views[view_name]
        return pairing.build_pairs(self.out_dir / left, right_dir=self.out_dir / right)

    def show(self, n: int = 6, view_name: Optional[str] = None, **kw):
        """Plot the first n matched pairs."""
        view_name = view_name or self.spec.default_view
        result = self.pairs(view_name)
        print(result.summary())
        left, right = self.spec.views[view_name]
        return view.plot_pairs(
            result.pairs[:n], titles=(left, right), suptitle=self.label, **kw
        )

    def save_to(self, destination, archive: bool = True) -> Path:
        """Copy results to Drive. Zips by default -- copying thousands of small
        files onto mounted Drive is slow enough that it gets interrupted."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"{self.spec.key}_{self.stage}_target{self.target_set}"
        if archive:
            out = shutil.make_archive(str(destination / stem), "zip", str(self.out_dir))
            print(f"Saved {out}")
            return Path(out)
        target = destination / stem
        shutil.copytree(self.out_dir, target, dirs_exist_ok=True)
        print(f"Saved {target}")
        return target

    def __str__(self) -> str:
        return f"{self.label}: {self.metrics}"


def run_stage(spec, root, stage: Stage, target_set, runs_dir=None,
              echo=True, **params) -> Run:
    """Run one stage of one model.

    Parameters are validated against what that stage's script accepts, so
    passing `num_samples` to training fails with a clear message rather than
    being silently ignored by argparse.
    """
    stage = Stage(stage)
    stage_spec: StageSpec = spec.stage(stage)

    root = Path(root)
    scripts = root / "scripts"
    script_path = scripts / stage_spec.script
    if not script_path.exists():
        raise FileNotFoundError(
            f"Script missing: {script_path}. Run setup first.")

    if target_set not in spec.target_sets:
        print(f"  note: target_set {target_set} is outside the sets prepared for "
              f"{spec.key} ({spec.target_sets}); this may fail to load data.")

    runs_dir = Path(runs_dir) if runs_dir else root / "runs"
    stamp = datetime.now().strftime("%H%M%S")
    # Never reuse a directory that already holds results: mixing two runs' SVGs
    # is what lets a plot silently show layouts from the wrong run.
    base = runs_dir / f"{spec.key}_{stage}_target{target_set}_{stamp}"
    out_dir, n = base, 2
    while out_dir.exists() and any(out_dir.iterdir()):
        out_dir, n = base.with_name(f"{base.name}_{n}"), n + 1
    out_dir.mkdir(parents=True, exist_ok=True)

    # The script always writes to scripts/outputs; clear it so this run cannot
    # inherit files from a previous one, then move the results out afterwards.
    scratch = scripts / "outputs"
    if scratch.exists():
        shutil.rmtree(scratch)

    params.setdefault("target_set", target_set)
    if "model_path" in stage_spec.accepts:
        params.setdefault("model_path", f"ckpts/exp/{spec.checkpoint}")
    cmd = [sys.executable, stage_spec.script] + stage_spec.build_args(
        spec.dataset, **params)

    print(f"Running {stage} for {spec.label}, target set {target_set}...")
    print("  this takes several minutes; progress appears below\n")

    started = datetime.now()
    captured: List[str] = []
    process = subprocess.Popen(
        cmd, cwd=str(scripts), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        captured.append(line)
        if echo:
            sys.stdout.write(line)
    process.wait()
    seconds = (datetime.now() - started).total_seconds()

    stdout = "".join(captured)
    log_path = out_dir / "sampling_log.txt"
    log_path.write_text(stdout)

    if scratch.exists():
        for item in scratch.iterdir():
            shutil.move(str(item), str(out_dir / item.name))
        shutil.rmtree(scratch, ignore_errors=True)

    if process.returncode != 0:
        raise RuntimeError(
            f"Sampling failed (exit {process.returncode}). Full log: {log_path}\n"
            + "".join(captured[-15:])
        )

    m = _metrics.parse(stdout)
    if stage_spec.reports_metrics and not m.complete:
        print("\n  warning: could not find GED/OA in the output; see", log_path)

    run = Run(spec, stage, target_set, int(params.get("num_samples") or 0),
              out_dir, m, log_path, seconds)
    print(f"\nDone in {seconds/60:.1f} min -> {out_dir}")
    if stage_spec.reports_metrics:
        print(" ", m)
    return run


def sample(spec, root, target_set, num_samples=64, batch_size=64,
           runs_dir=None, echo=True) -> Run:
    """Back-compatible shorthand for the TEST stage."""
    return run_stage(spec, root, Stage.TEST, target_set, runs_dir=runs_dir,
                     echo=echo, num_samples=num_samples, batch_size=batch_size)
