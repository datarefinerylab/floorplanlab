# Architecture

How `floorplanlab` is put together and why. For what it does and how to use it,
see [README.md](README.md).

```python
import floorplanlab as fpl

fpl.available()                       # list the models
lab = fpl.Lab(drive_folder="T5")
lab.setup("ohd-oesd")                 # <- switching models is this one string
run  = lab.generate(target_set=8)
run.show(6)
lab.table()
lab.report("comparison.pdf")
```

## Design

Two axes, both data.

**Which model** -- a `ModelSpec` in `models.py`:

| model | dataset | checkpoint | can |
|---|---|---|---|
| `hd-rplan`  | rplan | model250000.pt | test |
| `ohd-rplan` | rplan | model250000.pt | test |
| `ohd-oesd`  | oesd  | model310000.pt | test |
| `ohdw-oesd` | oesd  | ema_0.9999_300000.pt | prepare, train, test, deploy |

**Which stage** -- a `StageSpec` in `stages.py`. T4/T5 only ever sample from a
pre-trained checkpoint; OHD-W also builds its dataset, trains, and deploys, and
deploy takes a different script with different arguments. Stages are declared
per model, so `lab.train(...)` raises a clear `NotImplementedError` on a model
that ships inference-only, and passing `num_samples` to training is rejected
rather than silently ignored by argparse.

Adding a model means adding one `ModelSpec`, not a new code path.

```python
lab.setup("ohdw-oesd")
splits = lab.prepare(csv_dir=".../Layouts_csv", out_root=".../dataset").root
lab.use_dataset(splits / "train"); lab.train(target_set=7)
lab.use_dataset(splits / "test");  run = lab.generate(target_set=7)
lab.deploy(deployment_dir=splits / "deploy", target_set=7)
```

`datasets/<dataset>` holds one split at a time, so `use_dataset()` replaces rather than merges --
a test split landing on top of a train split is how a run ends up scored against data the model
saw in training. It clears the eval/syn `.npz` cache too: that cache is keyed by dataset name, not
contents, so a stale one makes the next run sample the *previous* dataset. The OHD-W notebook
carried this as a prose warning ("do not forget to delete all the cache") in a cell you had to
remember to run.

## What it fixes

- **Run isolation.** The tutorials reuse `scripts/outputs/` across runs, so a later run's plots
  can pick up an earlier run's SVGs. Each `generate()` gets its own directory.
- **No silent mispairing.** The tutorials' `build_pairs` keeps "the first one found" on a key
  collision. This refuses to guess and says why.
- **Metrics as data.** GED/OA are parsed from the sampling log into `run.metrics` instead of
  living only in scrollback. (The script prints "Compatibility" where the tutorials say "GED".)
- **Idempotent setup.** Re-running any cell is safe.
- **Fault-tolerant rendering.** One unreadable SVG skips with a warning instead of destroying a
  figure that took minutes to build. A missing *system library* still fails loudly, once.
- **Fast saving.** Results zip to Drive rather than copying thousands of small files.
- **No stale eval cache.** `use_dataset()` clears the cached eval/syn `.npz` when the dataset
  changes, so a run cannot sample the split that was in place before it.

## Layout

| file | role |
|---|---|
| `models.py`  | the registry -- the only file you edit to add a model |
| `stages.py`  | what a model can *do*; validates each stage's CLI arguments |
| `env.py`     | clone, install, place files and splits, clear caches (idempotent) |
| `runs.py`    | run the sampler, isolate outputs, return a `Run` |
| `metrics.py` | parse GED/OA from stdout |
| `pairing.py` | match SVGs across folders under either naming convention |
| `view.py`    | plots and the PDF report |
| `lab.py`     | the `Lab` facade students use |
| `convert/`   | CSV -> JSON conversion and dataset splits (see below) |

## `convert/` -- the OHD-W data pipeline

Extracted from the OHD-W notebook's single 1,354-line cell. Same behaviour,
split by concern, with the tuning constants gathered into `ConvertConfig`
instead of module globals (so two configurations can coexist in one process,
and a tolerance can be varied without editing source).

| file | role |
|---|---|
| `config.py`      | tolerances, zone map, orientation map; validated on construction |
| `geometry.py`    | simplify, normalize, edges, morphological gap closing |
| `adjacency.py`   | build, repair and validate `ed_rm` |
| `ordering.py`    | canonical component order; prune orphan interior doors |
| `orientation.py` | 8-way orientation for rooms and windows |
| `context.py`     | the AIF / VL / NN environmental labels |
| `pipeline.py`    | one CSV -> one JSON |
| `batch.py`       | a directory of CSVs |
| `splits.py`      | train/test/deploy splits, manifest, and `list.txt` |

`make_splits` consolidates three notebook cells and writes `list.txt`
automatically -- sampling fails confusingly without it. Deploy records keep only
`zone_types, orient, ed_rm, AIF, VL, NN`: the conditions known *before* a layout
exists.

## Tests

```
pytest tests/        # 80 tests
```

`tests/golden/` holds outputs produced by the **original notebook cell**. The
converter tests assert byte-identical agreement with them, so the extraction is
pinned to the behaviour it was extracted from. Regenerate them only for a
deliberate change, and say so.

## Status

Tested: metrics parsing (against the real stdout saved in T5, which it
reproduces exactly, kept in `tests/fixtures/`), SVG pairing under both naming
conventions, run isolation, dataset placement and cache clearing, stage argument
validation, corrupt-file tolerance, PDF reporting, the whole converter, and
splits.

Not tested: anything touching Colab, Drive or a GPU. Sampling is exercised
through stub scripts that imitate the real ones' file layout and stdout. Do one
real Colab run before teaching with it.
