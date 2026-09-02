"""Batch conversion of a directory of layout CSVs."""
import logging
import os
import random
import re
from collections import defaultdict

from .config import DEFAULT, DEFAULT_COLUMNS, ContextColumns, ConvertConfig
from .pipeline import csv_to_json, csv_to_json_no_windows

log = logging.getLogger(__name__)

__all__ = ["batch_convert_csvs"]


def select_diverse_csvs(csv_files, max_files=None, seed=123):
    """
    Select CSVs while maximizing unit_id diversity.

    Strategy:
      1. Group files by unit_id.
      2. Shuffle unit_ids reproducibly using `seed`.
      3. Take one file from each unit before taking a second file
         from any unit (round-robin).
      4. Continue until `max_files` is reached.
    """
    csv_files = sorted(csv_files)
    if max_files is None or max_files >= len(csv_files):
        return csv_files
    if not isinstance(max_files, int) or max_files <= 0:
        raise ValueError("max_files must be a positive integer or None.")
    rng = random.Random(seed)
    by_unit = defaultdict(list)
    for fname in csv_files:
        try:
            unit_id, _ = extract_unit_and_floor_from_filename(fname)
        except Exception:
            unit_id = f"__unparsed__::{fname}"
        by_unit[unit_id].append(fname)
    unit_ids = sorted(by_unit.keys())
    for unit_id in unit_ids:
        by_unit[unit_id] = sorted(by_unit[unit_id])
        rng.shuffle(by_unit[unit_id])
    rng.shuffle(unit_ids)
    selected = []
    level = 0
    while len(selected) < max_files:
        added_this_round = False
        for unit_id in unit_ids:
            files = by_unit[unit_id]
            if level < len(files):
                selected.append(files[level])
                added_this_round = True
                if len(selected) >= max_files:
                    break
        if not added_this_round:
            break
        level += 1
    return selected


# ============================================================
# --- BATCH PROCESSOR ---
# ============================================================

def extract_unit_and_floor_from_filename(fname):
    """
    Extracts unit_id and floor from filenames like:
      unit_5073.0_floor5.csv
      unit_5130_floor2.csv
    Returns: (unit_id:str, floor:int)
    """
    m = re.search(r"unit_(\d+)(?:\.0)?_floor(\d+)", fname.lower())
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname}")

    unit_id = m.group(1)          # '5073'
    floor = int(m.group(2))       # 5
    return unit_id, floor


def batch_convert_csvs(
    csv_dir,
    no_windows_output_dir,
    windows_output_dir,
    windows_plot_dir=None,
    no_windows_plot_dir=None,
    make_plots=True,
    show_plots=False,
    include_windows=True,
    exclude_windows=True,
    max_files=None,
    selection_seed=123,
    cfg: ConvertConfig = DEFAULT,
    columns: ContextColumns = DEFAULT_COLUMNS,
):
    """
    Batch-convert CSV files.

    Parameters
    ----------
    max_files : int or None
        Maximum number of CSV files to process. For example,
        max_files=3000 processes the first 3000 files after deterministic
        filename sorting. None processes all CSV files.

    make_plots : bool
        If True, save PNG plots to the supplied plot directories.

    show_plots : bool
        If True, also display plots inline. Keep False for large batch runs.
    """
    os.makedirs(no_windows_output_dir, exist_ok=True)
    os.makedirs(windows_output_dir, exist_ok=True)

    if make_plots:
        if include_windows:
            if windows_plot_dir is None:
                windows_plot_dir = os.path.join(windows_output_dir, "plots")
            os.makedirs(windows_plot_dir, exist_ok=True)

        if exclude_windows:
            if no_windows_plot_dir is None:
                no_windows_plot_dir = os.path.join(no_windows_output_dir, "plots")
            os.makedirs(no_windows_plot_dir, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv")
        and os.path.isfile(os.path.join(csv_dir, f))
    )

    if not csv_files:
        log.info("❌ No CSV files found in directory.")
        return

    total_available = len(csv_files)

    csv_files = select_diverse_csvs(
        csv_files,
        max_files=max_files,
        seed=selection_seed,
    )

    selected_unit_ids = set()
    for fname in csv_files:
        try:
            unit_id, _ = extract_unit_and_floor_from_filename(fname)
            selected_unit_ids.add(unit_id)
        except Exception:
            pass

    print(
        f"📂 Found {total_available} CSV files; "
        f"processing {len(csv_files)}."
    )
    log.info(f"🏠 Unique unit IDs in selection: {len(selected_unit_ids)}")
    log.info(f"🎲 Selection seed: {selection_seed}")

    success_windows = 0
    success_nowin = 0
    failed_windows = 0
    failed_nowin = 0

    for n, fname in enumerate(csv_files, start=1):
        csv_path = os.path.join(csv_dir, fname)
        log.info(f"\n🔄 [{n}/{len(csv_files)}] Processing {fname} ...")

        try:
            unit_id, floor = extract_unit_and_floor_from_filename(fname)
        except Exception as e:
            log.info(f"  ⚠ Bad filename format ({fname}): {e}")
            continue

        stem = f"{unit_id}_{floor}"

        if include_windows:
            try:
                plot_path = (
                    os.path.join(windows_plot_dir, f"{stem}.png")
                    if make_plots else None
                )

                csv_to_json(
                    csv_path,
                    show_plot=show_plots,
                    output_dir=windows_output_dir,
                    custom_filename=f"{stem}.json",
                    plot_path=plot_path,
                )
                success_windows += 1
                log.info("  ✔ Full JSON (with windows)")

            except Exception as e:
                failed_windows += 1
                log.info(f"  ⚠ Error in full JSON conversion: {e}")

        if exclude_windows:
            try:
                plot_path = (
                    os.path.join(no_windows_plot_dir, f"{stem}.png")
                    if make_plots else None
                )

                csv_to_json_no_windows(
                    csv_path,
                    show_plot=show_plots,
                    output_dir=no_windows_output_dir,
                    custom_filename=f"{stem}.json",
                    plot_path=plot_path,
                )
                success_nowin += 1
                log.info("  ✔ Window-free JSON")

            except Exception as e:
                failed_nowin += 1
                log.info(f"  ⚠ Error in no-windows conversion: {e}")

    log.info("\n🎉 Batch processing complete.")
    log.info(f"  Requested/processed CSVs: {len(csv_files)}")
    if include_windows:
        log.info(f"  With-windows JSONs:       {success_windows} success, {failed_windows} failed")
    if exclude_windows:
        log.info(f"  No-windows JSONs:         {success_nowin} success, {failed_nowin} failed")
