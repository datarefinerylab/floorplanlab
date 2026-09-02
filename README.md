# floorplanlab

Generate architectural floor plans with diffusion models by naming a model, not by wiring paths.

**📖 Full documentation: [floorplanlab/README.md](floorplanlab/README.md)**
· [Architecture](floorplanlab/ARCHITECTURE.md)
· [Tutorial notebook](floorplanlab/tutorial.ipynb)

```python
import floorplanlab as fpl

lab = fpl.Lab(drive_folder="my-floorplan-files")
lab.setup("hd-rplan")                      # switching models is this one string
run = lab.generate(target_set=6, num_samples=64)

run.show(6)                                # ground truth beside generated layouts
run.metrics                                # GED 3.378 +/- 0.100   OA 0.6181 +/- 0.0100
lab.report("comparison.pdf")
```

## Install

```bash
pip install git+https://github.com/datarefinerylab/floorplanlab.git
```

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

MPLBACKEND=Agg pytest tests/ -q -p no:warnings                    # 80 tests, ~2s
pyflakes floorplanlab/*.py floorplanlab/convert/*.py tests/*.py   # must be clean
```

`MPLBACKEND=Agg` is required — the tests render figures and PDFs headlessly.

## Scope

This repository is the library. The Colab teaching notebooks that use it are kept separately;
`floorplanlab/tutorial.ipynb` here is a self-contained tour, most of which runs with no GPU and
no model files.
