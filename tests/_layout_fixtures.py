"""Synthesise layout CSVs shaped like the O-ESD exports the converter expects."""
import random

def rect(x0, y0, x1, y1):
    return (f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, "
            f"{x0} {y1}, {x0} {y0}))")

def build(seed=0, n_rooms=4, with_windows=True):
    rng = random.Random(seed)
    rows = []
    def add(zoning, geom, wo="", vl=None, nn=None, wp=None, aif=None):
        rows.append({
            "zoning": zoning, "geometry": geom, "window_orientation": wo,
            "view_layer_landscape": vl, "noise_night": nn,
            "layout_window_perimeter": wp, "afternoon_illuminance_factor": aif,
        })

    # rooms on a grid, slightly jittered so geometry is not degenerate
    cells = [(0, 0), (12, 0), (0, 12), (12, 12), (24, 0), (24, 12), (0, 24), (12, 24)]
    for i in range(n_rooms):
        cx, cy = cells[i % len(cells)]
        j = rng.uniform(0, 0.4)
        add(f"zone0{(i % 4) + 1}", rect(cx, cy, cx + 10 + j, cy + 10 + j),
            vl=rng.uniform(0, 1), nn=rng.uniform(20, 60),
            wp=rng.choice([0, rng.uniform(1, 6)]), aif=rng.uniform(0, 1))

    # interior doors straddling shared walls
    add("DOOR", rect(9.6, 3, 12.4, 5), vl=None, nn=None, wp=0, aif=None)
    if n_rooms > 2:
        add("DOOR", rect(3, 9.6, 5, 12.4), vl=None, nn=None, wp=0, aif=None)

    if with_windows:
        for wo, geom in (("North", rect(2, 21.6, 6, 22.4)),
                         ("East",  rect(21.6, 2, 22.4, 6)),
                         ("South", rect(2, -0.4, 6, 0.4))):
            add("WINDOW", geom, wo=wo, vl=None, nn=None, wp=None, aif=None)

    add("ENTRANCE_DOOR", rect(-0.4, 3, 0.4, 5), vl=None, nn=None, wp=0, aif=None)
    # rows the converter must drop
    add("WALL", rect(0, 0, 30, 0.2))
    add("COLUMN", rect(11.8, 11.8, 12.2, 12.2))
    add("remaining", rect(25, 25, 26, 26))
    return rows

def write(path, **kw):
    import csv
    rows = build(**kw)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})
    return path
