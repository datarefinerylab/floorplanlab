"""Shared fixtures: a fake house_diffusion checkout with stub scripts.

Lets the runner, pairing and reporting be tested without a GPU, Drive or the
real model -- the stubs imitate the real scripts' file layout and stdout.
"""
import io
import sys
import textwrap
from pathlib import Path
from xml.etree import ElementTree

import pytest

from floorplanlab import models


def _cairosvg_available() -> bool:
    try:
        import cairosvg
    except Exception:
        return False
    return cairosvg is not None


class _StubCairoSVG:
    """Stand-in used only where the native cairo library is unavailable.

    It reproduces the one behaviour these tests depend on: raising ParseError
    on a truncated SVG, exactly as the real cairosvg does. Tests that need
    genuine rendering are skipped instead of stubbed.
    """

    @staticmethod
    def svg2png(url=None, dpi=96):
        from PIL import Image
        ElementTree.fromstring(Path(url).read_bytes())   # ParseError if truncated
        buf = io.BytesIO()
        Image.new("RGBA", (60, 60), (90, 120, 180, 255)).save(buf, "PNG")
        return buf.getvalue()


@pytest.fixture
def renderer(monkeypatch):
    """Guarantee svg rendering works, via the real library or the stub."""
    if not _cairosvg_available():
        monkeypatch.setitem(sys.modules, "cairosvg", _StubCairoSVG)
    return None


requires_real_cairo = pytest.mark.skipif(
    not _cairosvg_available(), reason="native cairo library not installed"
)

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">{body}</svg>'

# Writes the four SVG folders the real scripts produce, then prints metrics in
# the exact format image_sample_*.py uses.
FAKE_SAMPLER = textwrap.dedent('''
    import argparse, random
    from pathlib import Path
    ap = argparse.ArgumentParser()
    for f in ("dataset", "set_name", "model_path", "deployment_dir",
              "corner_dist_path", "save_svg", "draw_graph"):
        ap.add_argument("--" + f)
    for f in ("batch_size", "target_set", "num_samples", "save_interval",
              "lr_anneal_steps"):
        ap.add_argument("--" + f, type=int)
    a = ap.parse_args()
    tag_file = Path("MODEL_TAG")
    tag = tag_file.read_text().strip() if tag_file.exists() else "m"
    SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">{b}</svg>'
    n_out = a.num_samples or 8
    for sub, tmpl in (("graphs_gt", "{n}.svg"), ("graphs_pred", "{n}.svg"),
                      ("gt", "{n}c_0_gt.svg"), ("pred", "{n}c_0_pred.svg")):
        d = Path("outputs") / sub
        d.mkdir(parents=True, exist_ok=True)
        for n in range(min(n_out, 8)):
            d.joinpath(tmpl.format(n=n)).write_text(
                SVG.format(b='<text y="30">%s-t%s</text>' % (tag, a.target_set)))
    if a.set_name == "eval":
        random.seed(a.target_set or 0)
        gs, os_ = [], []
        for _ in range(5):
            g = (a.target_set or 6) * 0.55 + random.uniform(-.2, .2)
            o = 0.92 - (a.target_set or 6) * 0.006 + random.uniform(-.02, .02)
            gs.append(g); os_.append(o)
            print("sampling complete"); print("Compatibility: %s" % g); print("OA: %s" % o)
        print("Compatibility mean: %s \\t Compatibility std: 0.1" % (sum(gs)/5))
        print("OA mean: %s \\t OA std: 0.01" % (sum(os_)/5))
    else:
        print("stage %s complete" % a.set_name)
''')


@pytest.fixture
def fake_repo(tmp_path):
    """A house_diffusion checkout with every registry script stubbed out."""
    root = tmp_path / "house_diffusion"
    (root / "scripts" / "ckpts" / "exp").mkdir(parents=True)
    seen = set()
    for spec in models.MODELS.values():
        (root / "scripts" / "ckpts" / "exp" / spec.checkpoint).write_text("weights")
        for script in spec.scripts:
            if script not in seen:
                (root / "scripts" / script).write_text(FAKE_SAMPLER)
                seen.add(script)
    return root


@pytest.fixture
def tag_model(fake_repo):
    """Stand in for place_files() swapping one model's patched sources."""
    def _tag(name):
        (fake_repo / "scripts" / "MODEL_TAG").write_text(name)
    return _tag
