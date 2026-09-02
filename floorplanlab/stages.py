"""What a model can *do*, not just what data it uses.

T4 and T5 only ever sample from a pre-trained checkpoint. The OHD-W pipeline
also prepares its own dataset, trains, and deploys -- and deploy takes a
different script with different arguments and a different input schema.

Modelling that as a fourth registry row would not work, because the stages are
a second axis. Each model instead declares which stages it supports and how
each one is invoked, so `lab.train(...)` simply does not exist for a model that
ships as inference-only.
"""
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Tuple


class Stage(str, Enum):
    """A thing a model can be asked to do."""

    PREPARE = "prepare"   # CSV -> JSON -> splits (no sampling script)
    TRAIN = "train"
    TEST = "test"         # sample against a held-out set, with metrics
    DEPLOY = "deploy"     # generate from conditions only, no ground truth

    def __str__(self) -> str:
        return self.value


# Parameters the runner knows how to turn into CLI flags.
KNOWN_PARAMS: FrozenSet[str] = frozenset({
    "target_set", "num_samples", "batch_size", "model_path",
    "save_interval", "lr_anneal_steps",
    "deployment_dir", "corner_dist_path", "save_svg", "draw_graph",
})


@dataclass(frozen=True)
class StageSpec:
    """How to invoke one stage of one model."""

    script: str
    set_name: str
    #: flags always passed for this stage, as (flag, value) pairs
    fixed_args: Tuple[Tuple[str, str], ...] = ()
    #: parameters the caller may supply
    accepts: FrozenSet[str] = frozenset({"target_set", "num_samples",
                                         "batch_size", "model_path"})
    #: parameters the caller MUST supply
    requires: FrozenSet[str] = frozenset()
    #: whether this stage produces GED/OA worth parsing
    reports_metrics: bool = True
    #: whether outputs are SVG layouts to collect
    produces_outputs: bool = True
    description: str = ""

    def __post_init__(self):
        unknown = (self.accepts | self.requires) - KNOWN_PARAMS
        if unknown:
            raise ValueError(f"{self.script}: unknown parameters {sorted(unknown)}")
        if not self.requires <= self.accepts:
            raise ValueError(
                f"{self.script}: required params must also be accepted: "
                f"{sorted(self.requires - self.accepts)}"
            )

    def build_args(self, dataset: str, **params) -> list:
        """Turn parameters into a CLI argument list, rejecting unsupported ones."""
        supplied = {k: v for k, v in params.items() if v is not None}

        unsupported = set(supplied) - self.accepts
        if unsupported:
            raise TypeError(
                f"{self.script} does not accept {sorted(unsupported)}. "
                f"It accepts: {sorted(self.accepts)}"
            )
        missing = self.requires - set(supplied)
        if missing:
            raise TypeError(f"{self.script} requires {sorted(missing)}")

        args = ["--dataset", dataset, "--set_name", self.set_name]
        for flag, value in self.fixed_args:
            args += [f"--{flag}", str(value)]
        for key in sorted(supplied):
            args += [f"--{key}", str(supplied[key])]
        return args


# --- the stage shapes the three sampling scripts actually have ------------

def inference_stage(script: str, set_name: str = "eval") -> StageSpec:
    """Sampling against a held-out set: the T4/T5 shape."""
    return StageSpec(
        script=script,
        set_name=set_name,
        accepts=frozenset({"target_set", "num_samples", "batch_size", "model_path"}),
        requires=frozenset({"target_set", "model_path"}),
        description="Sample layouts for a held-out set and score them.",
    )


def training_stage(script: str = "image_train.py") -> StageSpec:
    return StageSpec(
        script=script,
        set_name="train",
        accepts=frozenset({"target_set", "batch_size", "save_interval",
                           "lr_anneal_steps"}),
        requires=frozenset({"target_set"}),
        reports_metrics=False,
        produces_outputs=False,
        description="Train from scratch; checkpoints are written periodically.",
    )


def deployment_stage(script: str) -> StageSpec:
    return StageSpec(
        script=script,
        set_name="deploy",
        accepts=frozenset({"target_set", "num_samples", "batch_size", "model_path",
                           "deployment_dir", "corner_dist_path",
                           "save_svg", "draw_graph"}),
        requires=frozenset({"target_set", "model_path", "deployment_dir"}),
        reports_metrics=False,   # no ground truth to score against
        description="Generate from conditions only -- no ground-truth geometry.",
    )
