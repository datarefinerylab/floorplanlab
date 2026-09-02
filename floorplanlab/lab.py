"""The object students interact with."""
from pathlib import Path
from typing import List, Optional, Sequence

from . import env, models, runs, view
from .stages import Stage


class Lab:
    """A workspace for generating floor layouts.

    Typical use in a notebook:

        lab = Lab(drive_folder="T5")
        lab.setup("ohd-oesd")
        run8 = lab.generate(target_set=8)
        run8.show(6)
    """

    def __init__(self, drive_folder: str = "T5", root: str = "/content/house_diffusion",
                 mount: bool = True, drive_root: str = "/content/drive/MyDrive"):
        self.root = Path(root)
        self.drive_root = Path(drive_root)
        self.drive_dir = self.drive_root / drive_folder
        self.spec: Optional[models.ModelSpec] = None
        self.runs: List[runs.Run] = []
        if mount:
            self._mount()

    def _mount(self) -> None:
        try:
            from google.colab import drive as _drive
        except ImportError:
            print("Not running in Colab; skipping Drive mount.")
            return
        _drive.mount(str(self.drive_root.parent), force_remount=False)

    # ----- setup -------------------------------------------------------
    def setup(self, model: str, install: bool = True, quiet: bool = True) -> "Lab":
        """Clone the repo, install dependencies, and place this model's files.

        Safe to call again -- switching models re-copies only the files that
        differ between them.
        """
        self.spec = spec = models.get(model)
        print(f"Setting up {spec.label}\n" + "=" * 60)

        if not self.drive_dir.is_dir():
            raise FileNotFoundError(
                f"Drive folder not found: {self.drive_dir}\n"
                f"Create it in My Drive and place the shared shortcuts inside, or "
                f"pass Lab(drive_folder='<your folder name>')."
            )

        print("1/4 cloning repositories")
        env.clone(self.root, quiet=quiet)

        if install:
            print("2/4 installing dependencies (a few minutes the first time)")
            env.install_dependencies(self.root, quiet=quiet)
        else:
            print("2/4 skipping dependency install")

        print("3/4 placing model files")
        env.place_files(spec, self.root, self.drive_dir)

        if spec.builds_own_dataset:
            print("4/4 dataset comes from prepare(); nothing to copy from Drive")
        else:
            print("4/4 placing test set")
            count = env.place_testset(spec, self.root, self.drive_dir)
            print(f"  {count} test files in datasets/{spec.dataset}")

        print(f"\nReady. Target sets available: {spec.target_sets}")
        if spec.notes:
            print(f"Note: {spec.notes}")
        return self

    # ----- stages ------------------------------------------------------
    def _require(self, stage: Stage) -> models.ModelSpec:
        if self.spec is None:
            raise RuntimeError("Call lab.setup('<model>') first.")
        if not self.spec.supports(stage):
            supported = ", ".join(str(s) for s in Stage if self.spec.supports(s))
            raise NotImplementedError(
                f"{self.spec.label} does not support the {stage} stage. "
                f"It supports: {supported}."
            )
        return self.spec

    def can(self, stage) -> bool:
        """Whether the current model supports a stage."""
        return self.spec is not None and self.spec.supports(Stage(stage))

    def generate(self, target_set: Optional[int] = None, num_samples: int = 64,
                 batch_size: int = 64, echo: bool = True) -> runs.Run:
        """Generate layouts for a held-out set, with metrics. (TEST stage)"""
        spec = self._require(Stage.TEST)
        if target_set is None:
            target_set = spec.target_sets[0]
        run = runs.run_stage(spec, self.root, Stage.TEST, target_set, echo=echo,
                             num_samples=num_samples, batch_size=batch_size)
        self.runs.append(run)
        return run

    def train(self, target_set: Optional[int] = None, batch_size: int = 128,
              save_interval: int = 50000, lr_anneal_steps: int = 0,
              echo: bool = True) -> runs.Run:
        """Train from scratch. Checkpoints are written every save_interval steps."""
        spec = self._require(Stage.TRAIN)
        if target_set is None:
            target_set = spec.target_sets[0]
        run = runs.run_stage(spec, self.root, Stage.TRAIN, target_set, echo=echo,
                             batch_size=batch_size, save_interval=save_interval,
                             lr_anneal_steps=lr_anneal_steps)
        self.runs.append(run)
        return run

    def deploy(self, deployment_dir, target_set: Optional[int] = None,
               num_samples: int = 30, batch_size: int = 1,
               corner_dist_path: Optional[str] = None, checkpoint: Optional[str] = None,
               save_svg: bool = True, draw_graph: bool = True,
               echo: bool = True) -> runs.Run:
        """Generate from conditions only -- no ground-truth geometry.

        This is design-from-a-brief rather than reconstruction, so no GED/OA
        are produced: there is nothing to score against.
        """
        spec = self._require(Stage.DEPLOY)
        if target_set is None:
            target_set = spec.target_sets[0]
        if corner_dist_path is None:
            corner_dist_path = f"{spec.processed_dir}/oesd_train_{target_set}_cndist.npz"
        params = dict(deployment_dir=str(deployment_dir),
                      corner_dist_path=corner_dist_path,
                      num_samples=num_samples, batch_size=batch_size,
                      save_svg=save_svg, draw_graph=draw_graph)
        if checkpoint:
            params["model_path"] = f"ckpts/exp/{checkpoint}"
        run = runs.run_stage(spec, self.root, Stage.DEPLOY, target_set,
                             echo=echo, **params)
        self.runs.append(run)
        return run

    def prepare(self, csv_dir, out_root, target_sets=None, include_windows: bool = True,
                test_frac: float = 0.10, deploy_frac: float = 0.05, seed: int = 123,
                make_plots: bool = False, max_files: Optional[int] = None):
        """Build this model's dataset from layout CSVs. (PREPARE stage)

        Converts CSV -> JSON, then splits into train/test/deploy with the
        `list.txt` manifests the sampling scripts require.
        """
        self._require(Stage.PREPARE)
        from pathlib import Path as _Path

        from .convert import batch_convert_csvs, make_splits

        out_root = _Path(out_root)
        json_root = out_root / "json"
        targets = target_sets or self.spec.target_sets
        inputs = {}
        for t in targets:
            src = _Path(csv_dir) / f"target_{t}"
            if not src.is_dir():
                raise FileNotFoundError(f"No CSV folder for target set {t}: {src}")
            win_dir = json_root / f"target_{t}_context"
            batch_convert_csvs(
                csv_dir=str(src),
                no_windows_output_dir=str(json_root / f"target_{t}_nowin"),
                windows_output_dir=str(win_dir),
                include_windows=include_windows,
                exclude_windows=not include_windows,
                make_plots=make_plots, show_plots=False,
                max_files=max_files, selection_seed=seed,
            )
            inputs[t] = win_dir
        result = make_splits(inputs, out_root / "splits",
                             test_frac=test_frac, deploy_frac=deploy_frac, seed=seed)
        print(f"Dataset ready: {result}")
        return result

    def use_dataset(self, split_dir, replace: bool = True) -> int:
        """Install a prepared split into datasets/<dataset> for the next stage.

        `datasets/<dataset>` holds one split at a time: point this at the train
        split before train(), at the test split before generate(). The cached
        eval/syn .npz files are cleared too, so a run cannot sample the split
        that was in place before it.
        """
        if self.spec is None:
            raise RuntimeError("Call lab.setup('<model>') first.")
        count = env.place_split(self.spec, self.root, split_dir, replace=replace)
        cleared = env.clear_eval_cache(self.spec, self.root)
        print(f"{count} layouts in datasets/{self.spec.dataset}"
              + (f"; cleared {cleared} cached .npz" if cleared else ""))
        return count

    # ----- reporting ---------------------------------------------------
    def table(self, selected: Optional[Sequence[runs.Run]] = None) -> None:
        """Print a metrics table for the runs done so far."""
        chosen = list(selected or self.runs)
        if not chosen:
            print("No runs yet.")
            return
        width = max(len(r.label) for r in chosen) + 2
        print(f"{'run':<{width}}{'GED':>18}{'OA':>20}")
        print("-" * (width + 38))
        for r in chosen:
            g = "-" if r.metrics.ged is None else f"{r.metrics.ged:.3f} +/- {r.metrics.ged_std:.3f}"
            o = "-" if r.metrics.oa is None else f"{r.metrics.oa:.4f} +/- {r.metrics.oa_std:.4f}"
            print(f"{r.label:<{width}}{g:>18}{o:>20}")
        print("\nGED: lower is better. OA: higher is better (0-1).")
        if len({r.target_set for r in chosen}) > 1:
            print("Careful: GED is not normalised by graph size, so it rises with "
                  "target set. Compare GED across models at equal target size.")

    def report(self, out_path: str = "comparison.pdf",
               selected: Optional[Sequence[runs.Run]] = None,
               view_name: Optional[str] = None, max_rows: Optional[int] = 12) -> Path:
        """Write a PDF comparing runs, opening with the metrics table."""
        chosen = list(selected or self.runs)
        if not chosen:
            raise RuntimeError("No runs to report on.")
        return view.comparison_pdf(chosen, out_path, view_name=view_name,
                                   max_rows=max_rows)


def compare(*selected_runs, out_path: Optional[str] = None):
    """Compare runs from anywhere, including runs made by different Labs."""
    chosen = list(selected_runs)
    width = max(len(r.label) for r in chosen) + 2
    print(f"{'run':<{width}}{'GED':>18}{'OA':>20}")
    print("-" * (width + 38))
    for r in chosen:
        g = "-" if r.metrics.ged is None else f"{r.metrics.ged:.3f} +/- {r.metrics.ged_std:.3f}"
        o = "-" if r.metrics.oa is None else f"{r.metrics.oa:.4f} +/- {r.metrics.oa_std:.4f}"
        print(f"{r.label:<{width}}{g:>18}{o:>20}")
    if out_path:
        return view.comparison_pdf(chosen, out_path)
    return None
