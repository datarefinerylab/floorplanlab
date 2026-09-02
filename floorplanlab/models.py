"""Model registry: everything that differs between tutorials lives here as data.

Adding a new model means adding one ModelSpec below -- no new code paths.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .stages import (Stage, StageSpec, deployment_stage, inference_stage,
                     training_stage)


@dataclass(frozen=True)
class ModelSpec:
    """One (model, dataset) combination that can generate floor layouts."""

    key: str                      # short name students type, e.g. "ohd-oesd"
    label: str                    # human-readable, used in plot titles and reports
    dataset: str                  # --dataset flag, also names datasets/<dataset>
    checkpoint: str               # default .pt file inside ckpts/exp/
    drive_subdir: str             # folder in the student's Drive holding the files
    pkg_files: List[str]          # patched .py files -> house_diffusion/
    npz_files: List[str]          # preprocessed data -> scripts/processed_<dataset>/
    testset_subdir: str = "test_set"
    target_sets: Tuple[int, ...] = (6,)
    # view name -> (left svg dir, right svg dir) inside a run's output folder
    views: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    default_view: str = "gt_vs_pred"
    notes: str = ""
    #: what this model can do; keyed by Stage
    stages: Dict[Stage, StageSpec] = field(default_factory=dict)
    #: True when the model's dataset is built by floorplanlab.convert
    builds_own_dataset: bool = False

    @property
    def processed_dir(self) -> str:
        return f"processed_{self.dataset}"

    @property
    def scripts(self) -> List[str]:
        """Every sampling/training script this model needs copied in."""
        return sorted({st.script for st in self.stages.values()})

    def supports(self, stage: Stage) -> bool:
        # PREPARE runs in Python (floorplanlab.convert), not via a script,
        # so it is declared by builds_own_dataset rather than by a StageSpec.
        if stage is Stage.PREPARE:
            return self.builds_own_dataset
        return stage in self.stages

    def stage(self, stage: Stage) -> StageSpec:
        try:
            return self.stages[stage]
        except KeyError:
            raise KeyError(
                f"{self.key} does not support the {stage} stage. "
                f"It supports: {', '.join(str(s) for s in sorted(self.stages))}"
            ) from None


_SHARED_PKG = ["script_util.py", "transformer.py", "train_util.py"]

_RPLAN_NPZ = [
    "rplan_train_6_cndist.npz",
    "rplan_train_6.npz",
    "rplan_eval_6.npz",
    "rplan_eval_6_syn.npz",
]

_OESD_NPZ = [
    "oesd_train_all_cndist.npz",
    "oesd_eval_6.npz",
    "oesd_eval_6_syn.npz",
    "oesd_eval_8.npz",
    "oesd_eval_8_syn.npz",
]

MODELS: Dict[str, ModelSpec] = {
    "hd-rplan": ModelSpec(
        key="hd-rplan",
        label="HouseDiffusion (RPLAN)",
        dataset="rplan",
        checkpoint="model250000.pt",
        stages={Stage.TEST: inference_stage("image_sample_rplan.py")},
        drive_subdir="HouseDiffusion_files",
        pkg_files=["rplanhg_datasets.py"] + _SHARED_PKG,
        npz_files=_RPLAN_NPZ,
        target_sets=(6,),
        views={"gt_vs_pred": ("gt", "pred")},
        notes="Baseline. Orientation is NOT an input condition; OA is reported "
              "only so it can be compared against the oriented model.",
    ),
    "ohd-rplan": ModelSpec(
        key="ohd-rplan",
        label="Oriented-HouseDiffusion (O-RPLAN)",
        dataset="rplan",
        checkpoint="model250000.pt",
        stages={Stage.TEST: inference_stage("image_sample_rplan.py")},
        drive_subdir="Oriented-HouseDiffusion_files",
        pkg_files=["rplanhg_datasets.py"] + _SHARED_PKG,
        npz_files=_RPLAN_NPZ,
        target_sets=(6,),
        views={"gt_vs_pred": ("gt", "pred")},
        notes="Same repo and checkpoint filename as hd-rplan; the difference is "
              "the patched source files and weights copied in from Drive.",
    ),
    "ohd-oesd": ModelSpec(
        key="ohd-oesd",
        label="Oriented-HouseDiffusion (O-ESD)",
        dataset="oesd",
        checkpoint="model310000.pt",
        stages={Stage.TEST: inference_stage("image_sample_oesd.py")},
        drive_subdir="sampling_files",
        pkg_files=["oesdhg_datasets.py"] + _SHARED_PKG,
        npz_files=_OESD_NPZ,
        target_sets=(5, 6, 7, 8),
        views={
            "gt_vs_pred": ("gt", "pred"),
            "graph_vs_pred": ("graphs_gt", "pred"),
            "predgraph_vs_pred": ("graphs_pred", "pred"),
        },
        default_view="graph_vs_pred",
        notes="Swiss Dwellings layouts. Input graphs show orientation as node ring "
              "colour: orange=West, pink=South, yellow=East, blue=North.",
    ),
    "ohdw-oesd": ModelSpec(
        key="ohdw-oesd",
        label="Oriented-HouseDiffusion with Windows (O-ESD)",
        dataset="oesd",
        checkpoint="ema_0.9999_300000.pt",
        drive_subdir="E4_paper_files",
        pkg_files=["oesdhg_datasets.py", "gaussian_diffusion.py"] + _SHARED_PKG,
        npz_files=[],          # the corner-distance file is built during prepare
        target_sets=(5, 6, 7, 8),
        stages={
            Stage.TRAIN: training_stage("image_train.py"),
            Stage.TEST: inference_stage("image_sample_wins.py"),
            Stage.DEPLOY: deployment_stage("image_sample_wins_deployment.py"),
        },
        builds_own_dataset=True,
        views={
            "gt_vs_pred": ("gt", "pred"),
            "graph_vs_pred": ("graphs_gt", "pred"),
            "predgraph_vs_pred": ("graphs_pred", "pred"),
        },
        default_view="graph_vs_pred",
        notes="Adds windows as geometry plus three layout-level environmental "
              "conditions (AIF, VL, NN). Builds its own dataset from CSV, and "
              "supports training and deployment as well as testing.",
    ),
}


def get(model_key: str) -> ModelSpec:
    try:
        return MODELS[model_key]
    except KeyError:
        raise KeyError(
            f"Unknown model {model_key!r}. Available: {', '.join(sorted(MODELS))}"
        ) from None


def available() -> None:
    """Print the model menu -- the first thing a student runs."""
    print("Available models\n" + "-" * 60)
    for spec in MODELS.values():
        sizes = ", ".join(str(t) for t in spec.target_sets)
        can = sorted(spec.stages)
        if spec.builds_own_dataset:
            can = [Stage.PREPARE] + can
        stages = ", ".join(str(s) for s in can)
        print(f"  {spec.key:<12} {spec.label}")
        print(f"  {'':<12} target sets: {sizes}")
        print(f"  {'':<12} can: {stages}")
        if spec.notes:
            print(f"  {'':<12} {spec.notes}")
        print()
