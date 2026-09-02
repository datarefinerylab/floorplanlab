"""Debug plot of a converted layout: polygons, orientations, adjacency.

Used by the batch converter to write a PNG next to each JSON so conversions can
be eyeballed. Not part of the data path.
"""
from pathlib import Path

import matplotlib.pyplot as plt


__all__ = ["plot_with_polygons"]


def plot_with_polygons(
    polygons,
    edges,
    ed_rm,
    zone_types,
    orient,
    win_orient=None,
    save_path=None,
    show=True,
):
    """
    Plot the processed layout and optionally save it.

    Raw O-ESD component colors:
      1  Zone 1          purple
      2  Zone 2          coral
      3  Zone 3          yellow
      4  Zone 4/balcony  gray
      15 Entrance door   dark gray
      16 Window          sky blue
      17 Internal door   warm brown
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Zone Polygons, Orientation & Adjacencies (Canonical Order)")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.grid(alpha=0.2)

    zone_color_map = {
        1: "#9C8ADE",   # Zone 1
        2: "#E89074",   # Zone 2
        3: "#E6C66A",   # Zone 3
        4: "#B8B8B8",   # Zone 4 / balcony
        15: "#4A4A4A",  # Entrance door
        16: "#66BBD6",  # Window
        17: "#A07863",  # Internal door
        0: "#e0e0e0",
    }

    orient_labels = {
        0: "0",
        1: "E", 2: "NE", 3: "N", 4: "NW",
        5: "W", 6: "SW", 7: "S", 8: "SE",
    }

    for i, poly in enumerate(polygons):
        zt = zone_types[i]
        ori = orient[i]
        x, y = poly.exterior.xy

        ax.fill(
            x, y,
            color=zone_color_map.get(zt, "#e0e0e0"),
            alpha=0.5,
            edgecolor="black",
            linewidth=0.8,
        )

        cx, cy = poly.centroid.coords[0]
        extra = ""
        if win_orient is not None and zt == 16:
            extra = f"\nwin:{win_orient[i]}"

        ax.text(
            cx, cy,
            f"{i}\nzt:{zt}\nori:{orient_labels.get(ori, '?')}{extra}",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.25"),
        )

    for e, rm in zip(edges, ed_rm):
        x1, y1, x2, y2 = e[:4]

        if len(rm) == 2:
            # Component-room mappings involving doors are red;
            # other two-component relationships are green.
            color = "red" if any(zone_types[r] in [15, 17] for r in rm) else "green"
            width = 2.3
        else:
            color = "#bbbbbb"
            width = 1.0

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.9)

        if len(rm) == 2:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx, my, str(rm),
                fontsize=7,
                color="black",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.25"),
            )

    total_corners = sum(len(p.exterior.coords) - 1 for p in polygons)
    ax.text(
        5, 250,
        f"Total corners: {total_corners}",
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.9),
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    # Essential for batch runs with hundreds/thousands of files.
    plt.close(fig)

