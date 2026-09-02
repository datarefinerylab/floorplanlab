"""Render SVG outputs as plots and PDF reports.

A single unreadable SVG must not destroy a figure the student already waited
minutes for, so rendering failures degrade to a placeholder cell and a warning.
"""
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def _renderer():
    """Import cairosvg, turning its native-library failure into a clear message.

    A missing system library affects every file and needs fixing once, so it is
    raised loudly rather than reported per file.
    """
    try:
        import cairosvg
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not load cairosvg, which renders the SVG outputs.\n"
            "In Colab: !apt-get install -y libcairo2 && !pip install cairosvg\n"
            f"Original error: {exc}"
        ) from exc
    return cairosvg


def svg_to_image(svg_path, dpi: int = 200):
    """Rasterise one SVG. Returns None (with a warning) if that file is unreadable."""
    from PIL import Image

    cairosvg = _renderer()
    try:
        png = cairosvg.svg2png(url=str(svg_path), dpi=dpi)
        return Image.open(BytesIO(png)).convert("RGBA")
    except Exception as exc:                       # noqa: BLE001 - one bad file
        print(f"  skipped {Path(svg_path).name}: {type(exc).__name__} ({exc})")
        return None


def _draw(ax, path, dpi):
    img = svg_to_image(path, dpi=dpi)
    if img is None:
        ax.text(0.5, 0.5, "could not\nrender", ha="center", va="center",
                fontsize=9, color="crimson", transform=ax.transAxes)
    else:
        ax.imshow(img)
    ax.axis("off")


def plot_pairs(pairs, titles: Tuple[str, str] = ("input", "generated"),
               rows_per_figure: int = 6, dpi: int = 220,
               cell_size: Tuple[float, float] = (4.0, 4.0),
               suptitle: Optional[str] = None):
    """Plot matched pairs, one pair per row, in figures of rows_per_figure."""
    if not pairs:
        print("Nothing to plot.")
        return

    for start in range(0, len(pairs), rows_per_figure):
        chunk = pairs[start:start + rows_per_figure]
        fig, axes = plt.subplots(
            len(chunk), 2,
            figsize=(2 * cell_size[0], len(chunk) * cell_size[1]),
            squeeze=False,
        )
        for r, pair in enumerate(chunk):
            _draw(axes[r][0], pair.left, dpi)
            _draw(axes[r][1], pair.right, dpi)
            axes[r][0].set_title(f"{pair.key} - {titles[0]}", fontsize=10)
            axes[r][1].set_title(f"{pair.key} - {titles[1]}", fontsize=10)
        if suptitle:
            fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout()
        plt.show()


def comparison_pdf(runs: Sequence, out_path, view_name: Optional[str] = None,
                   max_rows: Optional[int] = None, rows_per_page: int = 5,
                   dpi: int = 220, cell_size: Tuple[float, float] = (3.6, 3.6)) -> Path:
    """Build a multi-page PDF: one column per run, one row per shared sample.

    The first page is a metrics summary, so the report carries the numbers the
    discussion questions ask about -- not just pictures.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = []
    for run in runs:
        result = run.pairs(view_name)
        columns.append({p.key: p for p in result.pairs})

    shared = set(columns[0])
    for col in columns[1:]:
        shared &= set(col)
    keys = sorted(shared, key=lambda k: (len(k), k))
    if max_rows:
        keys = keys[:max_rows]
    if not keys:
        raise ValueError(
            "No sample keys are common to all runs, so there is nothing to "
            "compare row by row. Runs must use the same test set."
        )

    with PdfPages(out_path) as pdf:
        pdf.savefig(_summary_page(runs))
        plt.close("all")

        # first column shows the shared input, then one generated layout per run
        ncols = len(runs) + 1
        for start in range(0, len(keys), rows_per_page):
            chunk = keys[start:start + rows_per_page]
            fig, axes = plt.subplots(
                len(chunk), ncols,
                figsize=(ncols * cell_size[0], len(chunk) * cell_size[1]),
                squeeze=False,
            )
            for r, key in enumerate(chunk):
                _draw(axes[r][0], columns[0][key].left, dpi)
                if r == 0:
                    axes[r][0].set_title("input", fontsize=11)
                for c, (run, col) in enumerate(zip(runs, columns), start=1):
                    _draw(axes[r][c], col[key].right, dpi)
                    if r == 0:
                        axes[r][c].set_title(run.label, fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out_path} ({len(keys)} samples, {len(runs)} runs)")
    return out_path


def _summary_page(runs):
    """A metrics table as the report's opening page."""
    fig, ax = plt.subplots(figsize=(11, 2 + 0.5 * len(runs)))
    ax.axis("off")
    ax.set_title("Generated floor layouts - run comparison", fontsize=15, pad=18)
    table = ax.table(
        cellText=[
            [
                r.label,
                f"{r.num_samples}",
                "-" if r.metrics.ged is None else f"{r.metrics.ged:.3f} +/- {r.metrics.ged_std:.3f}",
                "-" if r.metrics.oa is None else f"{r.metrics.oa:.4f} +/- {r.metrics.oa_std:.4f}",
            ]
            for r in runs
        ],
        colLabels=["run", "samples", "GED (lower better)", "OA (higher better)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    fig.text(0.5, 0.06,
             "GED is not normalised by graph size: layouts with more rooms have more "
             "edges, so GED tends to rise with target set.\nCompare GED across models "
             "at equal target size, not across target sizes.",
             ha="center", fontsize=8.5, style="italic")
    return fig
