# floorplanlab

**Generate architectural floor plans with diffusion models**

`floorplanlab` is a small Python library that wraps
[HouseDiffusion](https://github.com/aminshabani/house_diffusion) and its orientation-aware
variants so you can go from "which model do I want?" to a set of generated floor layouts and
comparable metrics in about five lines of code.

[![Open the tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datarefinerylab/floorplanlab/blob/main/floorplanlab/tutorial.ipynb)

**New here?** [**tutorial.ipynb**](tutorial.ipynb) is a 10-minute tour of everything below. Most
of it runs with no GPU and no model files, so you can try it right now.

---

## The problem

Running these models means cloning repositories, nesting one inside the other, installing a pile
of dependencies, copying patched sources and pre-trained weights into precisely the right
directories, invoking a CLI script with six flags, then hand-parsing its log for the metrics.

Doing it for a *second* model means doing all of it again — the models ship files with identical
names and different contents.

**Before** — a dozen hardcoded paths:

```python
!git clone https://github.com/aminshabani/house_diffusion
!git clone https://github.com/openai/guided-diffusion
!pip install drawsvg cairosvg mpi4py pytorch_fid pygraphviz
shutil.move("/content/guided-diffusion", "/content/house_diffusion")
os.chdir("/content/house_diffusion"); !pip install -e .

SOURCE_DIR = Path("/content/drive/MyDrive/my-floorplan-files/HouseDiffusion_files")
for d in [ROOT/"datasets"/"rplan", ROOT/"scripts"/"processed_rplan", ...]:
    d.mkdir(parents=True, exist_ok=True)
copy_files(["rplanhg_datasets.py", "script_util.py", ...], SOURCE_DIR, dest_pkg)
copy_files(["model250000.pt"], SOURCE_DIR, dest_ckpts)
# ... and so on, then:
!python image_sample_rplan.py --dataset rplan --batch_size 64 --set_name eval \
    --target_set 6 --model_path ckpts/exp/model250000.pt --num_samples 64
# now scroll back through 13 minutes of log output to find the GED and OA values
```

**After:**

```python
import floorplanlab as fpl

lab = fpl.Lab(drive_folder="my-floorplan-files")
lab.setup("hd-rplan")
run = lab.generate(target_set=6, num_samples=64)

run.show(6)          # ground truth beside generated layouts
run.metrics          # GED 3.378 +/- 0.100   OA 0.6181 +/- 0.0100  (5 rounds)
```

Switching to a different model is one string.

---

## Installation

```bash
pip install git+https://github.com/datarefinerylab/floorplanlab.git
```

In a Colab notebook:

```python
!pip install -q git+https://github.com/datarefinerylab/floorplanlab.git
```

`floorplanlab` orchestrates the upstream models; it does not vendor them. `lab.setup()` clones
the repositories and installs their dependencies on first use.

---

## Quick start

`drive_folder` is the folder in your Google Drive holding the pre-trained weights, patched model
sources and datasets — see [Requirements](#requirements) for how to obtain them. Name it
whatever you like and pass that name.

```python
import floorplanlab as fpl

fpl.available()                                    # print the model menu

lab = fpl.Lab(drive_folder="my-floorplan-files")
lab.setup("hd-rplan")                              # clone, install, place weights and data

run = lab.generate(target_set=6, num_samples=64)
run.show(6)                                        # plot the first 6 results
run.save_to("results/")
```

For a guided version of this — plus dataset conversion, splits and the guardrails — open
[tutorial.ipynb](tutorial.ipynb).

---

## Models

`fpl.available()` lists what you can generate with. All four share the HouseDiffusion
architecture; the Oriented variants are research models that add orientation and environmental
conditioning. Weights come from their authors — see [Credits](#credits).

| Model | Dataset | Conditioned on | Can |
|---|---|---|---|
| `hd-rplan` | RPLAN | room adjacency | test |
| `ohd-rplan` | O-RPLAN | adjacency + orientation | test |
| `ohd-oesd` | O-ESD (Swiss Dwellings) | adjacency + orientation | test |
| `ohdw-oesd` | O-ESD | adjacency, orientation, windows, daylight / view / noise | prepare, train, test, deploy |

`target set` is the number of spaces in a layout: `hd-rplan` and `ohd-rplan` are trained for 6,
the O-ESD models for 5–8.

---

## Features

- **Switch models with one string.** Every difference between models — patched sources,
  checkpoint, dataset, sampling script — is configuration, not code.
- **Metrics as data.** GED and orientation alignment are parsed out of the sampling log into
  `run.metrics`, so you can tabulate and compare them instead of scrolling.
- **Comparison built in.** `lab.table()` for the numbers, `lab.report()` for a multi-page PDF
  whose first page is the metrics summary.
- **Runs never contaminate each other.** Every run gets its own output directory, and the SVG
  matcher refuses to guess when filenames are ambiguous rather than silently pairing the wrong
  files.
- **Survives bad input.** A single unreadable SVG is skipped with a warning instead of
  destroying a figure that took ten minutes to render.
- **Idempotent setup.** Re-running any cell is safe — a common need in notebooks.
- **Full lifecycle where the model supports it.** Dataset preparation, training, testing and
  deployment, each declared per model so unsupported calls fail with a clear message.
- **Dataset tooling.** Convert layout CSVs to the model's JSON schema and build reproducible
  train/test/deploy splits with manifests.

---

## Examples

### Compare two models on the same data

```python
lab = fpl.Lab(drive_folder="my-floorplan-files")

lab.setup("hd-rplan")
baseline = lab.generate(target_set=6)

lab.setup("ohd-rplan", install=False)     # swap sources; no runtime restart needed
oriented = lab.generate(target_set=6)

lab.table()
lab.report("comparison.pdf")
```

```
run                                              GED                   OA
--------------------------------------------------------------------------
HouseDiffusion (RPLAN) / target 6      3.378 +/- 0.100    0.6181 +/- 0.0100
Oriented-HouseDiffusion (O-RPLAN) / t  2.100 +/- 0.100    0.9227 +/- 0.0100
```

### Compare layout sizes for one model

```python
lab.setup("ohd-oesd")

big   = lab.generate(target_set=8)
small = lab.generate(target_set=6)

big.show(6, view_name="graph_vs_pred")    # input bubble diagram beside the result
lab.table()
```

> **Note.** GED is not normalised by graph size — layouts with more rooms have more edges and
> tend to score higher. Compare GED across *models at equal target size*, not across sizes.
> `lab.table()` prints this warning when you mix sizes.

### Build a dataset, train, and deploy

```python
lab.setup("ohdw-oesd")

splits = lab.prepare(csv_dir="layouts_csv/", out_root="dataset/").root   # CSV -> JSON -> splits

lab.use_dataset(splits / "train")     # datasets/<dataset> holds one split at a time
lab.train(target_set=7)

lab.use_dataset(splits / "test")
run = lab.generate(target_set=7)

lab.deploy(deployment_dir=splits / "deploy", target_set=7)
```

`use_dataset()` also clears the cached `eval`/`syn` `.npz` files, which are keyed by dataset name
rather than contents — leave a stale one in place and the next run quietly samples the previous
dataset.

`deploy` generates from *conditions only* — room types, adjacencies, orientations and
environmental targets, with no ground-truth geometry. That is design from a brief, as opposed to
reconstructing a plan the model has the answer to.

Calling a stage a model does not support tells you so:

```python
lab.setup("hd-rplan")
lab.train(target_set=6)
# NotImplementedError: HouseDiffusion (RPLAN) does not support the train stage.
#                      It supports: test.
```

### Convert layout CSVs on their own

```python
from floorplanlab.convert import csv_to_json, make_splits, ConvertConfig

data = csv_to_json("layout.csv", output_dir="json/")
data["AIF"], data["VL"], data["NN"]     # daylight / view / noise orientation targets

# tolerances are configurable rather than hardcoded
loose = csv_to_json("layout.csv", cfg=ConvertConfig(edge_adj_dist_tol=6.0))

make_splits({6: "json/target_6", 8: "json/target_8"}, "dataset/",
            test_frac=0.10, deploy_frac=0.05, seed=123)
```

---

## Requirements

- **Python 3.9+**
- **A CUDA GPU** for generation. Designed for Google Colab (T4 or better); a 64-sample run takes
  roughly 13–16 minutes on a T4.
- **Model files in Google Drive** — pre-trained weights, patched model sources and datasets.
  These are not redistributed here; obtain them from the model authors or your course materials,
  put them in a Drive folder, and pass its name as `Lab(drive_folder=...)`.
- `numpy`, `pandas`, `matplotlib`, `shapely`, `pillow`, `cairosvg`.

The upstream models additionally need `drawsvg`, `mpi4py`, `pytorch_fid`, `pygraphviz` and
system Graphviz; `lab.setup()` installs these for you.

---

## How it works

Two orthogonal axes, both expressed as data:

- **`ModelSpec`** — which model and dataset.
- **`StageSpec`** — what that model can do (prepare, train, test, deploy), including which CLI
  arguments each stage accepts.

Adding a model means adding one record to the registry, not a new code path. The design rests on
one observation: all four models are the *same* cloned repository, frequently with the same
script and checkpoint *filenames*, differing only in the patched files copied in. Because
sampling runs as a subprocess, swapping those files is enough to switch models in place.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module-by-module breakdown, or
[tutorial.ipynb](tutorial.ipynb) to watch it work.

---

## Development

```bash
git clone https://github.com/datarefinerylab/floorplanlab.git
cd floorplanlab
pip install -e ".[dev]"

MPLBACKEND=Agg pytest tests/ -q          # 80 tests, ~2 seconds
```

Tests cover metric parsing, SVG pairing, run isolation, stage argument validation, error
tolerance, the CSV converter and dataset splits. Sampling is exercised through stub scripts that
imitate the real ones' file layout and output, so the suite runs without a GPU.

The converter's tests assert byte-identical agreement with golden files produced by the original
research notebook — regenerate them only for a deliberate behaviour change.

---

## Status

The pure-Python surface is well covered by tests; the parts that touch Colab, Google Drive and a
GPU are exercised against stubs rather than the real environment, so treat a first run in a new
setup as something to watch.

Contributions welcome — especially additional model definitions. Adding one should mean adding a
single `ModelSpec`; if it doesn't, that's a design bug worth reporting.

---

## Credits

`floorplanlab` is a wrapper. The models and datasets are the work of their authors:

- Shabani, M. A., Hosseini, S., & Furukawa, Y. (2023). *HouseDiffusion: Vector Floorplan
  Generation via a Diffusion Model with Discrete and Continuous Denoising.* CVPR.
  [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf)
  · [code](https://github.com/aminshabani/house_diffusion)
- Mostafavi, F., Khademi, S., & Vrachliotis, G. *Oriented-HouseDiffusion: orientation-aware
  floor layout generation using diffusion models.*
  [paper](https://conf.dap.tuwien.ac.at/preprints/ecaade2025/ecaade2025_349.pdf)
- Wu, W., Fu, X.-M., Tang, R., Wang, Y., Qi, Y.-H., & Liu, L. (2019). *Data-driven Interior Plan
  Generation for Residential Buildings.* SIGGRAPH Asia. (RPLAN dataset)
  [paper](https://dl.acm.org/doi/pdf/10.1145/3355089.3356556)
- [guided-diffusion](https://github.com/openai/guided-diffusion), OpenAI.

## License

Not yet chosen. Note that any license here applies only to `floorplanlab` itself — the upstream
models, weights and datasets carry their own terms.
