"""Tests for stage execution, run isolation and reporting."""
import matplotlib
import pytest

matplotlib.use("Agg")

from floorplanlab import models, runs, view
from floorplanlab.stages import Stage


def _sample(spec, root, target_set=6, **kw):
    kw.setdefault("num_samples", 8)
    return runs.run_stage(spec, root, Stage.TEST, target_set, echo=False, **kw)


# --- run isolation: the bug this library exists to prevent ----------------

def test_each_run_gets_its_own_directory(fake_repo):
    spec = models.get("ohd-oesd")
    a = _sample(spec, fake_repo, 8)
    b = _sample(spec, fake_repo, 6)
    assert a.out_dir != b.out_dir


def test_repeated_runs_never_share_a_directory(fake_repo):
    """Same model, same target, same second must not collide."""
    spec = models.get("ohd-oesd")
    dirs = [_sample(spec, fake_repo, 6).out_dir for _ in range(3)]
    assert len({d.name for d in dirs}) == 3


def test_scratch_outputs_is_cleaned_between_runs(fake_repo):
    """The shared scripts/outputs folder must not survive a run."""
    spec = models.get("ohd-oesd")
    _sample(spec, fake_repo, 6)
    assert not (fake_repo / "scripts" / "outputs").exists()


def test_a_run_cannot_inherit_a_previous_runs_files(fake_repo, tag_model):
    tag_model("first")
    a = _sample(models.get("ohd-oesd"), fake_repo, 6)
    tag_model("second")
    b = _sample(models.get("ohd-oesd"), fake_repo, 6)

    a_text = sorted((a.out_dir / "pred").glob("*.svg"))[0].read_text()
    b_text = sorted((b.out_dir / "pred").glob("*.svg"))[0].read_text()
    assert "first" in a_text and "second" in b_text


# --- switching models -----------------------------------------------------

def test_switching_models_changes_the_outputs(fake_repo, tag_model):
    """T4's flow: HD and OHD share a script name but must produce different runs."""
    hd, ohd = models.get("hd-rplan"), models.get("ohd-rplan")
    assert hd.scripts == ohd.scripts, "same script filename, different contents"

    tag_model("hd")
    r_hd = _sample(hd, fake_repo, 6)
    tag_model("ohd")
    r_ohd = _sample(ohd, fake_repo, 6)

    assert "hd" in sorted((r_hd.out_dir / "pred").glob("*.svg"))[0].read_text()
    assert "ohd" in sorted((r_ohd.out_dir / "pred").glob("*.svg"))[0].read_text()
    assert r_hd.out_dir.name.startswith("hd-rplan")
    assert r_ohd.out_dir.name.startswith("ohd-rplan")


# --- stages ---------------------------------------------------------------

def test_unsupported_stage_is_refused(fake_repo):
    with pytest.raises(KeyError, match="does not support"):
        runs.run_stage(models.get("hd-rplan"), fake_repo, Stage.TRAIN, 6)


def test_training_runs_and_reports_no_metrics(fake_repo):
    run = runs.run_stage(models.get("ohdw-oesd"), fake_repo, Stage.TRAIN, 7,
                         echo=False, batch_size=128, save_interval=50000,
                         lr_anneal_steps=0)
    assert run.stage is Stage.TRAIN
    assert not run.metrics.complete, "training has no GED/OA to report"


def test_deploy_requires_a_deployment_dir(fake_repo):
    with pytest.raises(TypeError, match="requires"):
        runs.run_stage(models.get("ohdw-oesd"), fake_repo, Stage.DEPLOY, 6,
                       echo=False)


def test_deploy_runs_with_its_own_arguments(fake_repo, tmp_path):
    d = tmp_path / "deploy_in"; d.mkdir()
    run = runs.run_stage(models.get("ohdw-oesd"), fake_repo, Stage.DEPLOY, 6,
                         echo=False, deployment_dir=str(d),
                         corner_dist_path="processed_oesd/x.npz",
                         num_samples=4, batch_size=1)
    assert run.stage is Stage.DEPLOY
    assert "--deployment_dir" in run.log_path.read_text() or run.out_dir.exists()


def test_wrong_parameter_for_a_stage_is_rejected(fake_repo):
    """Passing num_samples to training must fail loudly, not be ignored."""
    with pytest.raises(TypeError, match="does not accept"):
        runs.run_stage(models.get("ohdw-oesd"), fake_repo, Stage.TRAIN, 7,
                       echo=False, num_samples=64)


# --- metrics + pairing ----------------------------------------------------

def test_metrics_are_captured_as_data(fake_repo):
    run = _sample(models.get("ohd-oesd"), fake_repo, 8)
    d = run.metrics.as_dict()
    assert d["ged"] is not None and d["oa"] is not None
    assert d["rounds"] == 5


def test_all_declared_views_pair_up(fake_repo):
    spec = models.get("ohd-oesd")
    run = _sample(spec, fake_repo, 6)
    for view_name in spec.views:
        result = run.pairs(view_name)
        assert len(result.pairs) == 8, view_name
        assert not result.collisions


def test_unknown_view_is_refused(fake_repo):
    run = _sample(models.get("ohd-oesd"), fake_repo, 6)
    with pytest.raises(KeyError, match="no view"):
        run.pairs("nonsense")


# --- resilience -----------------------------------------------------------

def test_a_corrupt_svg_does_not_destroy_the_figure(fake_repo, capsys, renderer):
    """The failure that killed T5 cell 38 after 10 pages of output."""
    run = _sample(models.get("ohd-oesd"), fake_repo, 6)
    victim = sorted((run.out_dir / "pred").glob("*.svg"))[2]
    victim.write_text("")                      # truncated, as an interrupted copy leaves it

    run.show(6, view_name="graph_vs_pred")     # must not raise
    assert "skipped" in capsys.readouterr().out


def test_report_includes_metrics_page(fake_repo, tmp_path, renderer):
    spec = models.get("ohd-oesd")
    a, b = _sample(spec, fake_repo, 8), _sample(spec, fake_repo, 6)
    pdf = view.comparison_pdf([a, b], tmp_path / "cmp.pdf", view_name="graph_vs_pred")
    assert pdf.exists() and pdf.stat().st_size > 5000


def test_results_save_as_an_archive(fake_repo, tmp_path):
    run = _sample(models.get("ohd-oesd"), fake_repo, 6)
    out = run.save_to(tmp_path / "drive")
    assert out.exists() and out.suffix == ".zip"
