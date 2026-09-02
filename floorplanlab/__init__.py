"""floorplanlab - generate floor layouts by switching one model name.

    import floorplanlab as fpl

    fpl.available()                     # see the models
    lab = fpl.Lab(drive_folder="T5")
    lab.setup("ohd-oesd")               # switch models by changing this string
    run = lab.generate(target_set=8)
    run.show(6)
    lab.table()
"""
from .lab import Lab, compare
from .metrics import Metrics, parse as parse_metrics
from .models import MODELS, ModelSpec, available, get
from .view import comparison_pdf, plot_pairs, svg_to_image

__version__ = "0.1.0"
__all__ = [
    "Lab", "compare", "available", "get", "MODELS", "ModelSpec",
    "Metrics", "parse_metrics", "plot_pairs", "comparison_pdf", "svg_to_image",
]
