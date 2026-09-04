"""design.py -- Synthetic corpus generator.

Generates 14 designs x 6 scenarios, writing the exact CSV files from the data
contract (docs/DATA_SCHEMA.md).  Two fidelity levels per design:

  Ground truth   96x96 fine grid, AS-BUILT strap map, per-instance current
                 with sub-tile concentration.            -> irmap.csv, LABEL
  Early estimate 24x24 coarse grid, PLANNED strap map, tile-averaged current.
                 Rebuilt at feature time from published CSVs only.

The GAP between the two is the only thing the model is asked to learn.

Mesh constants
--------------
There is no per-design calibration.  Each design's conductances are derived
from its PUBLISHED PDN geometry (strap_width_um / strap_pitch_um) through two
global technology constants (k_sheet, k_bump) calibrated once over the corpus.
That matters for two reasons: feature extraction can reproduce the conductances
without ever touching the fine solve, and on real data the same two constants
are what S8 fits against the signoff map.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from prism.io_csv import Design, load_config, write_design, validate_design, load_design
from prism.solver import PDNSolver, coarse_solver_from_design, design_conductances


# ---------------------------------------------------------------------------
# Module class templates for a NAND flash controller
# ---------------------------------------------------------------------------

_MODULE_CLASSES = [
    # (name, clock_domain, power_domain, rated power factor)
    ("host_if",    "clk_host",  "vdd_core", 1.0),
    ("dma_fabric", "clk_core",  "vdd_core", 0.8),
    ("ecc_engine", "clk_core",  "vdd_core", 1.2),
    ("ch_ctrl",    "clk_nand",  "vdd_io",   0.9),
    ("sram_ctl",   "clk_core",  "vdd_core", 0.7),
    ("seq_core",   "clk_nand",  "vdd_io",   0.6),
]

_CLOCK_DOMAINS = ["clk_host", "clk_core", "clk_nand"]

# Activity multipliers per scenario, per module class (spec table, S2)
_ACTIVITY_TABLE = {
    #                host_if  dma_fabric  ecc_engine  ch_ctrl  sram_ctl  seq_core
    "idle":         [0.05,    0.02,       0.01,       0.02,    0.05,     0.10],
    "seq_read":     [0.85,    0.80,       0.55,       0.70,    0.60,     0.35],
    "seq_write":    [0.90,    0.90,       0.95,       0.85,    0.75,     0.40],
    "rand_read_4k": [0.60,    0.55,       0.45,       0.50,    0.90,     0.85],
    "gc_compact":   [0.05,    0.85,       0.90,       0.90,    0.70,     0.60],
    "ecc_recover":  [0.10,    0.20,       1.00,       0.15,    0.30,     0.25],
}

_SCENARIOS = list(_ACTIVITY_TABLE.keys())

_SCENARIO_WEIGHTS = {
    "idle":         0.30,
    "seq_read":     0.25,
    "seq_write":    0.15,
    "rand_read_4k": 0.15,
    "gc_compact":   0.10,
    "ecc_recover":  0.05,
}

_REFERENCE_SCENARIO = "seq_read"

# ---------------------------------------------------------------------------
# Divergence mechanisms.  Values are taken from the physical ranges quoted in
# the spec and fixed BEFORE the bias was measured; they are not tuned to hit a
# target number.
# ---------------------------------------------------------------------------

_DIVERGENCE_PARAMS = {
    "congestion_thinning_max": 0.25,   # spec range 10-40%
    "macro_edge_degradation": 0.22,    # spec range 15-30%
    "macro_edge_radius_tiles": 2,
    "lognormal_sigma": 0.6,            # sub-tile current concentration
    "justification": (
        "congestion_thinning_max=0.25: mid-range of the 10-40% spec band, "
        "reflects routing congestion thinning straps under high local "
        "placement utilisation. "
        "macro_edge_degradation=0.22: mid-range of the 15-30% spec band, "
        "models via-array degradation within 2 fine tiles of a hard macro "
        "boundary. "
        "lognormal_sigma=0.6: per-instance current spread within a tile. The "
        "draw is mean-normalised so total power is UNCHANGED -- this "
        "redistributes current, it does not add any."
    ),
}

# Achievable PDN geometries (design_stats.csv columns).  The strap ratio
# r = width/pitch sets the mesh conductance.
_STRAP_WIDTHS = [0.4, 0.8, 1.6]
_STRAP_PITCHES = [8.0, 12.0, 16.0, 24.0]
_STRAP_RATIOS = sorted({w / p for w in _STRAP_WIDTHS for p in _STRAP_PITCHES})

# Median strap ratio the corpus is calibrated to sit at (geometric mid of the
# achievable range), so designs spread either side of it.
_R_TARGET_MEDIAN = float(np.sqrt(min(_STRAP_RATIOS) * max(_STRAP_RATIOS)))

# k_bump is held at this multiple of k_sheet, so scaling the PDN scales the
# mesh and the bump connection together and U stays exactly proportional to
# 1/(k*r) -- the solver's closed-form scaling law.
_BUMP_TO_SHEET = 10.0

# Area per cell fitted from role A's stats.csv:
#   cfg_small 63008/26793 = 2.351, cfg_mid 258403/107473 = 2.404,
#   cfg_large 1657077/695524 = 2.382   ->  mean 2.38 um^2/cell
_AREA_PER_CELL = 2.38

# No subsampling cap.  instances.csv carries ONE ROW PER PLACED CELL, exactly as
# role A's real export does.  A cap would make top_cells / top_util / top_capden
# scale with true_cells/sampled_cells, which is a per-design constant and
# therefore a design fingerprint (R1/R2).  Cost is disk, not correctness.


# ---------------------------------------------------------------------------
# Floorplanning
# ---------------------------------------------------------------------------

def _slice_floorplan(
    x0: float, y0: float, x1: float, y1: float,
    weights: np.ndarray, rng: np.random.RandomState,
) -> List[Tuple[float, float, float, float]]:
    """Recursive slicing floorplan: partition a rectangle into len(weights)
    non-overlapping sub-rectangles with areas proportional to `weights`.

    Cuts always run across the longer side, which keeps aspect ratios sane.
    """
    n = len(weights)
    if n == 1:
        return [(x0, y0, x1, y1)]

    # Split the weight list near the halfway point of total weight
    cum = np.cumsum(weights)
    k = int(np.searchsorted(cum, cum[-1] / 2.0)) + 1
    k = max(1, min(n - 1, k))
    frac = float(cum[k - 1] / cum[-1])
    frac = float(np.clip(frac + rng.uniform(-0.04, 0.04), 0.15, 0.85))

    if (x1 - x0) >= (y1 - y0):
        xm = x0 + (x1 - x0) * frac
        left = _slice_floorplan(x0, y0, xm, y1, weights[:k], rng)
        right = _slice_floorplan(xm, y0, x1, y1, weights[k:], rng)
    else:
        ym = y0 + (y1 - y0) * frac
        left = _slice_floorplan(x0, y0, x1, ym, weights[:k], rng)
        right = _slice_floorplan(x0, ym, x1, y1, weights[k:], rng)
    return left + right


def _place_in_rect(
    rng: np.random.RandomState, n: int,
    rect: Tuple[float, float, float, float],
    macro_rects: List[Tuple[float, float, float, float]],
    tries: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform placement inside `rect`, rejecting points that land under a hard
    macro (no standard cells sit under a macro in a real placement)."""
    x0, y0, x1, y1 = rect
    xs = rng.uniform(x0, x1, size=n)
    ys = rng.uniform(y0, y1, size=n)
    for _ in range(tries):
        bad = np.zeros(n, dtype=bool)
        for (mx0, my0, mx1, my1) in macro_rects:
            bad |= (xs >= mx0) & (xs <= mx1) & (ys >= my0) & (ys <= my1)
        k = int(bad.sum())
        if k == 0:
            break
        xs[bad] = rng.uniform(x0, x1, size=k)
        ys[bad] = rng.uniform(y0, y1, size=k)
    return xs, ys


# ---------------------------------------------------------------------------
# Layout: everything that does not depend on the mesh constants
# ---------------------------------------------------------------------------

def _build_layout(seed: int, cfg: dict) -> Dict:
    """Build the physical design (die, modules, macros, placement, bumps,
    planned and as-built strap maps).  Deterministic in `seed` and independent
    of the mesh calibration, so it can be built once and reused across the two
    calibration passes."""
    rng = np.random.RandomState(seed)

    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = ny_f // ny_c
    vdd = cfg["electrical"]["vdd"]

    # --- Design-level scalars, scaled off role A's real numbers ---
    cells = int(10 ** rng.uniform(np.log10(2.5e4), np.log10(7e5)))
    chip_area = cells * _AREA_PER_CELL
    seq_pct = 58.0 + rng.uniform(-2, 2)
    flops = int(cells * seq_pct / 100.0)

    core_util = rng.uniform(0.55, 0.85)
    aspect = rng.uniform(0.8, 1.25)
    total_area = chip_area / core_util
    die_h = float(np.sqrt(total_area / aspect))
    die_w = float(total_area / die_h)

    bump_pitch = float(rng.choice([150.0, 200.0, 300.0]))
    clock_period = float(rng.uniform(1.0, 5.0))
    design_id = f"syn_{seed:03d}"

    tile_w, tile_h = die_w / nx_f, die_h / ny_f

    # --- Modules ---
    n_modules = rng.randint(6, 11)
    mod_names, mod_class_idx, mod_rated_mw = [], [], []
    for i, (name, _cd, _pd, pf) in enumerate(_MODULE_CLASSES):
        mod_names.append(f"{name}_{i}")
        mod_class_idx.append(i)
        mod_rated_mw.append(cells * pf * rng.uniform(0.8, 1.2) * 1e-4)
    for i in range(n_modules - len(_MODULE_CLASSES)):
        ci = rng.randint(0, len(_MODULE_CLASSES))
        mod_names.append(f"{_MODULE_CLASSES[ci][0]}_{len(_MODULE_CLASSES) + i}")
        mod_class_idx.append(ci)
        mod_rated_mw.append(cells * _MODULE_CLASSES[ci][3] * rng.uniform(0.5, 0.8) * 1e-4)

    n_modules = len(mod_names)
    modules_df = pd.DataFrame({
        "module": mod_names,
        "clock_domain": [_MODULE_CLASSES[ci][1] for ci in mod_class_idx],
        "power_domain": [_MODULE_CLASSES[ci][2] for ci in mod_class_idx],
        "rated_power_mw": [round(v, 3) for v in mod_rated_mw],
    })

    # --- Module floorplan: a slicing partition of the die ---
    mod_cell_share = np.array(mod_rated_mw) / float(np.sum(mod_rated_mw))
    order = rng.permutation(n_modules)          # randomise which region each gets
    rects_in_order = _slice_floorplan(0.0, 0.0, die_w, die_h,
                                      mod_cell_share[order], rng)
    mod_rects: List[Optional[Tuple[float, float, float, float]]] = [None] * n_modules
    for pos, mi in enumerate(order):
        mod_rects[int(mi)] = rects_in_order[pos]

    # --- Hard macros, placed inside a module region ---
    n_macros = rng.randint(2, 6)
    macro_records, macro_rects = [], []
    for mi in range(n_macros):
        owner = int(rng.randint(0, n_modules))
        rx0, ry0, rx1, ry1 = mod_rects[owner]
        mw = (rx1 - rx0) * rng.uniform(0.25, 0.55)
        mh = (ry1 - ry0) * rng.uniform(0.25, 0.55)
        mx0 = rng.uniform(rx0, rx1 - mw)
        my0 = rng.uniform(ry0, ry1 - mh)
        macro_rects.append((mx0, my0, mx0 + mw, my0 + mh))
        macro_records.append({
            "macro_id": mi,
            "module": mod_names[owner],
            "x0_um": round(mx0, 3), "y0_um": round(my0, 3),
            "x1_um": round(mx0 + mw, 3), "y1_um": round(my0 + mh, 3),
            "power_mw": round(mod_rated_mw[owner] * rng.uniform(0.10, 0.30), 3),
        })
    macros_df = pd.DataFrame(macro_records)

    # --- Bumps: a regular array at the sampled pitch ---
    nbx = max(2, int(die_w / bump_pitch))
    nby = max(2, int(die_h / bump_pitch))
    bx = bump_pitch / 2 + np.arange(nbx) * bump_pitch
    by = bump_pitch / 2 + np.arange(nby) * bump_pitch
    bx = bx[bx < die_w]
    by = by[by < die_h]
    if bx.size == 0:
        bx = np.array([die_w / 2])
    if by.size == 0:
        by = np.array([die_h / 2])
    bgx, bgy = np.meshgrid(bx, by)
    bumps_df = pd.DataFrame({
        "bump_id": np.arange(bgx.size),
        "x_um": np.round(bgx.ravel(), 3),
        "y_um": np.round(bgy.ravel(), 3),
    })

    # --- Instances, placed inside their own module's region ---
    mod_counts = np.round(mod_cell_share * cells).astype(int)
    mod_counts = np.maximum(mod_counts, 1)
    mod_assign = np.repeat(np.arange(n_modules), mod_counts)
    n_inst = int(mod_assign.size)

    inst_x = np.empty(n_inst)
    inst_y = np.empty(n_inst)
    for mi in range(n_modules):
        sel = mod_assign == mi
        xs, ys = _place_in_rect(rng, int(sel.sum()), mod_rects[mi], macro_rects)
        inst_x[sel] = xs
        inst_y[sel] = ys
    inst_x = np.clip(inst_x, 0.0, die_w * 0.999)
    inst_y = np.clip(inst_y, 0.0, die_h * 0.999)

    # Areas are drawn around the fitted mean and then normalised so that
    # sum(area_um2) == chip_area_um2 exactly.  Without this, top_util does not
    # reconcile with core_util and the feature is meaningless.
    inst_area = rng.uniform(0.4, 1.6, size=n_inst) * _AREA_PER_CELL
    inst_area *= chip_area / inst_area.sum()
    inst_cap = inst_area * 2.2
    is_seq = (rng.random_sample(n_inst) < (seq_pct / 100.0)).astype(int)
    cell_types = np.array(["INV", "NAND2", "NOR2", "DFF", "BUF", "AOI"])
    cell_type_idx = rng.randint(0, cell_types.size, size=n_inst)

    # --- Clock network: one vertical spine per clock domain.  Cells within a
    # narrow band of their domain's spine are clock cells.  This gives
    # top_clkden real spatial structure instead of a constant zero.
    inst_clk_dom = np.array([_MODULE_CLASSES[mod_class_idx[m]][1] for m in mod_assign])
    spine_x = {cd: rng.uniform(0.15, 0.85) * die_w for cd in _CLOCK_DOMAINS}
    spine_halfwidth = 0.030 * die_w
    is_clk = np.zeros(n_inst, dtype=int)
    for cd in _CLOCK_DOMAINS:
        sel = inst_clk_dom == cd
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        is_clk[idx[np.abs(inst_x[idx] - spine_x[cd]) < spine_halfwidth]] = 1
    # scattered clock buffers elsewhere
    is_clk[rng.random_sample(n_inst) < 0.01] = 1

    instances_df = pd.DataFrame({
        "inst_id": np.arange(n_inst),
        "inst_name": "u_" + pd.Index(np.arange(n_inst)).astype(str),
        "module": np.array(mod_names)[mod_assign],
        "cell_type": cell_types[cell_type_idx],
        "x_um": np.round(inst_x, 3),
        "y_um": np.round(inst_y, 3),
        "area_um2": np.round(inst_area, 3),
        "cap_ff": np.round(inst_cap, 3),
        "is_seq": is_seq,
        "is_clk": is_clk,
        "is_macro": np.zeros(n_inst, dtype=int),
    })

    # --- Planned strap map (a FEATURE: what the floorplan intends) ---
    planned = np.clip(
        gaussian_filter(rng.uniform(0.5, 0.9, size=(ny_f, nx_f)), sigma=4.0),
        0.1, 1.0,
    )

    # --- As-built strap map (LABEL side: what routing actually delivers) ---
    asbuilt = planned.copy()

    # (1) Routing-congestion thinning where placement utilisation is high
    inst_fx = np.clip((inst_x / tile_w).astype(int), 0, nx_f - 1)
    inst_fy = np.clip((inst_y / tile_h).astype(int), 0, ny_f - 1)
    cell_density = np.zeros((ny_f, nx_f))
    np.add.at(cell_density, (inst_fy, inst_fx), inst_area)
    dmax = cell_density.max()
    norm_density = cell_density / dmax if dmax > 0 else cell_density
    asbuilt *= (1.0 - _DIVERGENCE_PARAMS["congestion_thinning_max"] * norm_density)

    # (2) Via degradation in a ring around each hard macro
    edge_r = _DIVERGENCE_PARAMS["macro_edge_radius_tiles"]
    macro_edge = np.zeros((ny_f, nx_f), dtype=bool)
    for (mx0, my0, mx1, my1) in macro_rects:
        x0t = int(mx0 / tile_w); x1t = min(int(mx1 / tile_w), nx_f - 1)
        y0t = int(my0 / tile_h); y1t = min(int(my1 / tile_h), ny_f - 1)
        macro_edge[max(0, y0t - edge_r):min(ny_f, y1t + edge_r + 1),
                   max(0, x0t - edge_r):min(nx_f, x1t + edge_r + 1)] = True
        macro_edge[y0t:y1t + 1, x0t:x1t + 1] = False
    asbuilt[macro_edge] *= (1.0 - _DIVERGENCE_PARAMS["macro_edge_degradation"])
    asbuilt = np.clip(asbuilt, 0.01, 1.0)

    # --- Bump mask on the fine grid ---
    bump_fine = np.zeros((ny_f, nx_f), dtype=bool)
    bump_fine[np.clip((bumps_df["y_um"].values / tile_h).astype(int), 0, ny_f - 1),
              np.clip((bumps_df["x_um"].values / tile_w).astype(int), 0, nx_f - 1)] = True
    if not bump_fine.any():
        bump_fine[ny_f // 2, nx_f // 2] = True

    # --- activity.csv ---
    activity_df = pd.DataFrame([
        {"scenario": scn, "module": mod_names[mi],
         "activity": _ACTIVITY_TABLE[scn][mod_class_idx[mi]],
         "mission_weight": _SCENARIO_WEIGHTS[scn]}
        for scn in _SCENARIOS for mi in range(n_modules)
    ])

    # --- paths.csv ---
    n_paths = min(100, max(10, n_inst // 500))
    path_records = []
    for pi in range(n_paths):
        n_ip = int(rng.randint(3, min(12, n_inst)))
        pids = np.sort(rng.choice(n_inst, size=n_ip, replace=False))
        path_records.append({
            "path_id": pi,
            "endpoint": f"u_{int(pids[-1])}",
            "clock_domain": str(modules_df.iloc[int(rng.randint(0, n_modules))]["clock_domain"]),
            "slack_ns": round(float(rng.uniform(-0.5, 2.0)), 3),
            "delay_ns": round(float(rng.uniform(0.5, clock_period * 0.9)), 3),
            "inst_ids": ";".join(str(int(x)) for x in pids),
        })
    paths_df = pd.DataFrame(path_records)

    return {
        "seed": seed, "design_id": design_id,
        "cells": cells, "flops": flops, "chip_area": chip_area,
        "seq_pct": seq_pct, "core_util": core_util,
        "die_w": die_w, "die_h": die_h,
        "bump_pitch": bump_pitch, "clock_period": clock_period,
        "hash": np.random.RandomState(seed + 7919).bytes(6).hex(),
        "xor_cells": int(cells * rng.uniform(0.005, 0.015)),
        "modules": modules_df, "macros": macros_df, "macro_rects": macro_rects,
        "instances": instances_df, "bumps": bumps_df,
        "activity": activity_df, "paths": paths_df,
        "planned": planned, "asbuilt": asbuilt, "bump_fine": bump_fine,
        "mod_names": mod_names, "mod_class_idx": mod_class_idx,
        "mod_rated_mw": mod_rated_mw, "mod_assign": mod_assign,
        "inst_x": inst_x, "inst_y": inst_y, "n_inst": n_inst,
        "ratio": ratio, "vdd": vdd,
    }


# ---------------------------------------------------------------------------
# Currents
# ---------------------------------------------------------------------------

def _instance_base_currents(
    mod_rated_mw, mod_class_idx, mod_assign, scenario: str, vdd: float
) -> np.ndarray:
    """Per-instance DC current [A] for a scenario.

    A module's rated power is shared equally across its instances and scaled by
    the scenario activity multiplier for that module's class.
    """
    counts = np.bincount(mod_assign, minlength=len(mod_rated_mw)).astype(float)
    counts[counts == 0] = 1.0
    act = np.array(_ACTIVITY_TABLE[scenario])[np.asarray(mod_class_idx)]
    per_inst = (np.asarray(mod_rated_mw) * 1e-3 * act) / (vdd * counts)
    return per_inst[mod_assign]


def _macro_current_grid(macros_df, activity_scn, ny, nx, die_w, die_h, vdd) -> np.ndarray:
    """Hard-macro power spread uniformly over the tiles the macro covers."""
    grid = np.zeros((ny, nx))
    if macros_df is None or len(macros_df) == 0:
        return grid
    tw, th = die_w / nx, die_h / ny
    for _, m in macros_df.iterrows():
        act = float(activity_scn.get(m["module"], 0.0))
        i_tot = float(m["power_mw"]) * 1e-3 * act / vdd
        x0 = int(np.clip(m["x0_um"] / tw, 0, nx - 1))
        x1 = int(np.clip(m["x1_um"] / tw, 0, nx - 1))
        y0 = int(np.clip(m["y0_um"] / th, 0, ny - 1))
        y1 = int(np.clip(m["y1_um"] / th, 0, ny - 1))
        n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
        grid[y0:y1 + 1, x0:x1 + 1] += i_tot / n_tiles
    return grid


def scenario_currents(design, scenario: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Fine and coarse current maps [A] for one scenario of a loaded Design.

    Reads ONLY published inputs -- instances.csv, modules.csv, macros.csv,
    activity.csv, design_stats.csv -- so it is safe under the leakage trap and
    identical on synthetic and real corpora.

    Returns
    -------
    (fine_current[ny_fine, nx_fine], coarse_current[ny_coarse, nx_coarse])
    """
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    stats = design.stats.iloc[0]
    die_w, die_h = float(stats["die_w_um"]), float(stats["die_h_um"])
    vdd = float(stats["vdd_v"])

    inst = design.instances
    mods = inst["module"].values
    rated = dict(zip(design.modules["module"], design.modules["rated_power_mw"]))
    act_scn = design.activity[design.activity["scenario"] == scenario]
    act_map = dict(zip(act_scn["module"], act_scn["activity"]))

    uniq, counts = np.unique(mods, return_counts=True)
    cnt_map = dict(zip(uniq, counts))
    per_module_i = {
        m: (float(rated.get(m, 0.0)) * 1e-3 * float(act_map.get(m, 0.0)))
           / (vdd * float(cnt_map[m]))
        for m in uniq
    }
    inst_i = pd.Series(mods).map(per_module_i).to_numpy(dtype=float)

    x = inst["x_um"].values
    y = inst["y_um"].values

    fine = np.zeros((ny_f, nx_f))
    np.add.at(fine,
              (np.clip((y / (die_h / ny_f)).astype(int), 0, ny_f - 1),
               np.clip((x / (die_w / nx_f)).astype(int), 0, nx_f - 1)),
              inst_i)
    coarse = np.zeros((ny_c, nx_c))
    np.add.at(coarse,
              (np.clip((y / (die_h / ny_c)).astype(int), 0, ny_c - 1),
               np.clip((x / (die_w / nx_c)).astype(int), 0, nx_c - 1)),
              inst_i)

    fine += _macro_current_grid(design.macros, act_map, ny_f, nx_f, die_w, die_h, vdd)
    coarse += _macro_current_grid(design.macros, act_map, ny_c, nx_c, die_w, die_h, vdd)
    return fine, coarse


def _ground_truth_current(layout: Dict, scenario: str, cfg: dict) -> np.ndarray:
    """Fine-grid current for the GROUND TRUTH solve.

    Adds sub-tile current concentration: instance currents are lognormal within
    a tile.  The draw is mean-normalised, so this REDISTRIBUTES current without
    changing total power -- otherwise the resulting bias would just be extra
    power, not a modelling gap.  Generator-only; never in scenario_currents().
    """
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    die_w, die_h = layout["die_w"], layout["die_h"]
    vdd = layout["vdd"]

    base = _instance_base_currents(layout["mod_rated_mw"], layout["mod_class_idx"],
                                   layout["mod_assign"], scenario, vdd)

    # Seeded per (design, scenario) so the draw does not depend on call order.
    sigma = _DIVERGENCE_PARAMS["lognormal_sigma"]
    rng = np.random.RandomState(1_000_003 * layout["seed"] + _SCENARIOS.index(scenario))
    conc = rng.lognormal(0.0, sigma, size=layout["n_inst"])
    conc /= np.exp(sigma ** 2 / 2.0)          # unit mean: total power preserved

    fine = np.zeros((ny_f, nx_f))
    np.add.at(fine,
              (np.clip((layout["inst_y"] / (die_h / ny_f)).astype(int), 0, ny_f - 1),
               np.clip((layout["inst_x"] / (die_w / nx_f)).astype(int), 0, nx_f - 1)),
              base * conc)

    act = layout["activity"]
    act_scn = act[act["scenario"] == scenario]
    act_map = dict(zip(act_scn["module"], act_scn["activity"]))
    fine += _macro_current_grid(layout["macros"], act_map, ny_f, nx_f, die_w, die_h, vdd)
    return fine


# ---------------------------------------------------------------------------
# Design assembly
# ---------------------------------------------------------------------------

def _reference_drop(layout: Dict, cfg: dict) -> float:
    """Max fine-grid drop for the reference scenario at unit mesh scale
    (k*r == 1).  Because sheet and bump conductance scale together, the drop at
    any other scale is exactly this divided by k*r -- no search needed."""
    solver = PDNSolver(cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"],
                       sheet_cond=1.0, bump_cond=_BUMP_TO_SHEET,
                       bump_mask=layout["bump_fine"],
                       strap_density=layout["asbuilt"])
    u = solver.solve(_ground_truth_current(layout, _REFERENCE_SCENARIO, cfg))
    return float(np.max(u))


def _select_strap_geometry(required_kr: float, k_sheet: float) -> Tuple[float, float, float]:
    """Pick the achievable (width, pitch) whose ratio is closest in log space to
    the ratio this design needs.  Returns (width, pitch, ratio)."""
    r_want = required_kr / k_sheet
    r = min(_STRAP_RATIOS, key=lambda v: abs(np.log(v) - np.log(r_want)))
    for w in _STRAP_WIDTHS:
        for p in _STRAP_PITCHES:
            if abs(w / p - r) < 1e-12:
                return w, p, r
    raise RuntimeError(f"no strap geometry realises ratio {r}")


def generate_design(seed: int, cfg: dict,
                    k_sheet: Optional[float] = None,
                    k_bump: Optional[float] = None,
                    layout: Optional[Dict] = None) -> Design:
    """Generate one synthetic design.

    k_sheet / k_bump are the global technology constants from build_corpus().
    If omitted, the design is calibrated on its own reference scenario -- used
    only for standalone inspection, never for the delivered corpus.
    """
    if layout is None:
        layout = _build_layout(seed, cfg)

    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    vdd = cfg["electrical"]["vdd"]
    budget_v = vdd * cfg["electrical"]["ir_budget_frac"]

    u_ref = _reference_drop(layout, cfg)
    if k_sheet is None:
        k_sheet = u_ref / budget_v / _R_TARGET_MEDIAN
        k_bump = k_sheet * _BUMP_TO_SHEET

    strap_w, strap_p, _r = _select_strap_geometry(u_ref / budget_v, k_sheet)

    stats = pd.DataFrame([{
        "design_id": layout["design_id"], "config": f"cfg_syn_{seed}",
        "hash": layout["hash"],
        "cells": layout["cells"], "flops": layout["flops"],
        "xor_cells": layout["xor_cells"],
        "chip_area_um2": round(layout["chip_area"], 3),
        "seq_area_um2": round(layout["chip_area"] * layout["seq_pct"] / 100.0, 3),
        "seq_pct": round(layout["seq_pct"], 2), "lint_rc": 1,
        "ts_utc": "2026-09-03T00:00:00Z",
        "die_w_um": round(layout["die_w"], 3), "die_h_um": round(layout["die_h"], 3),
        "vdd_v": vdd,
        "clock_period_ns": round(layout["clock_period"], 3),
        "core_util": round(layout["core_util"], 3),
        "bump_pitch_um": layout["bump_pitch"],
        "strap_pitch_um": strap_p,
        "strap_width_um": strap_w,
        "pdn_layers": "M4-M7",
    }])

    sheet_cond, bump_cond = design_conductances(stats.iloc[0], k_sheet, k_bump)

    # --- Ground-truth solve: fine grid, as-built straps ---
    solver_fine = PDNSolver(ny_f, nx_f, sheet_cond, bump_cond,
                            layout["bump_fine"], strap_density=layout["asbuilt"])
    solver_fine.factorise()

    fy, fx = np.meshgrid(np.arange(ny_f), np.arange(nx_f), indexing="ij")
    irmap_parts = []
    for scn in _SCENARIOS:
        u = np.maximum(solver_fine.solve(_ground_truth_current(layout, scn, cfg)), 0.0)
        irmap_parts.append(pd.DataFrame({
            "scenario": scn,
            "fy": fy.ravel(), "fx": fx.ravel(),
            "drop_v": np.round(u.ravel(), 8),
        }))
    irmap_df = pd.concat(irmap_parts, ignore_index=True)

    strap_planned_df = pd.DataFrame({
        "fy": fy.ravel(), "fx": fx.ravel(),
        "density": np.round(layout["planned"].ravel(), 6),
    })

    return Design(
        design_id=layout["design_id"], design_dir="",
        stats=stats, modules=layout["modules"], macros=layout["macros"],
        instances=layout["instances"], bumps=layout["bumps"],
        strap_planned=strap_planned_df, activity=layout["activity"],
        paths=layout["paths"], irmap=irmap_df, toggle=None,
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def build_corpus(cfg: dict) -> None:
    """Generate the corpus into data/synthetic/ and validate every design
    against its own contract."""
    n_designs = cfg["corpus"]["n_designs"]
    base_seed = cfg["corpus"]["seed"]
    vdd = cfg["electrical"]["vdd"]
    budget_v = vdd * cfg["electrical"]["ir_budget_frac"]
    data_dir = pathlib.Path("data") / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)

    seeds = [base_seed + i for i in range(n_designs)]

    # --- Pass 1: global calibration of the two technology constants ---
    # Each design is PDN-planned toward a target peak drop sampled around the
    # budget, so the corpus contains designs that meet it and designs that do
    # not.  k_sheet is then set so the MEDIAN design needs the mid-range strap
    # geometry -- one constant for the whole corpus, no per-design fudging.
    print("Pass 1/2  calibrating global mesh constants...")
    layouts: Dict[int, Dict] = {}
    required = []
    for s in tqdm(seeds, desc="calibrate"):
        layouts[s] = _build_layout(s, cfg)
        target = budget_v * np.random.RandomState(s + 31337).uniform(0.55, 1.30)
        required.append(_reference_drop(layouts[s], cfg) / target)
    k_sheet = float(np.median(required)) / _R_TARGET_MEDIAN
    k_bump = k_sheet * _BUMP_TO_SHEET

    calib = {
        "k_sheet": k_sheet, "k_bump": k_bump,
        "bump_to_sheet_ratio": _BUMP_TO_SHEET,
        "n_free_parameters": 1,
        "free_parameter": "k_sheet",
        "k_bump_is_derived": (
            f"k_bump == {_BUMP_TO_SHEET} * k_sheet, held fixed by construction. "
            "The mesh has ONE free parameter, not two. Tying them is what makes "
            "U exactly proportional to 1/(k*r), so strap geometry is selected in "
            "closed form with no search (solver property 2)."
        ),
        "r_target_median": _R_TARGET_MEDIAN,
        "reference_scenario": _REFERENCE_SCENARIO,
        "budget_v": budget_v,
        "note": (
            "Two global technology constants for the whole corpus. A design's "
            "conductances are k * (strap_width_um / strap_pitch_um), both of "
            "which are published in design_stats.csv, so feature extraction "
            "reproduces them without touching the fine solve. S8 refits these "
            "two numbers against the real signoff map."
        ),
    }
    print(f"  k_sheet = {k_sheet:.6g}   k_bump = {k_bump:.6g}")

    # --- Pass 2: build and write the corpus ---
    print("Pass 2/2  generating corpus...")
    manifest_records = []
    for s in tqdm(seeds, desc="designs"):
        design = generate_design(s, cfg, k_sheet=k_sheet, k_bump=k_bump,
                                 layout=layouts[s])
        design_dir = str(data_dir / design.design_id)
        design.design_dir = design_dir
        write_design(design, design_dir)

        failures = validate_design(design_dir)
        if failures:
            print(f"\nVALIDATION FAILED for {design.design_id}:")
            for f in failures:
                print(f"  - {f}")
            raise ValueError(f"Design {design.design_id} failed validation")

        manifest_records.append({
            "design_id": design.design_id, "corpus": "synthetic",
            "path": design.design_id,
            "n_scenarios": int(design.activity["scenario"].nunique()),
            "n_instances": int(len(design.instances)),
            "source": "synthetic",
        })

    pd.DataFrame(manifest_records).to_csv(data_dir / "manifest.csv", index=False)

    out_dir = pathlib.Path("out")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "generator_params.json", "w") as f:
        json.dump(_DIVERGENCE_PARAMS, f, indent=2)
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nCorpus written to {data_dir}/  ({len(manifest_records)} designs)")
    print("Mesh constants  -> out/calibration.json")
    print("Divergence      -> out/generator_params.json")


def load_calibration() -> Tuple[float, float]:
    """The two global mesh constants written by build_corpus()."""
    with open(pathlib.Path("out") / "calibration.json") as f:
        c = json.load(f)
    return float(c["k_sheet"]), float(c["k_bump"])


# ---------------------------------------------------------------------------
# S2 gate
# ---------------------------------------------------------------------------

def _run_sanity_checks(cfg: dict) -> None:
    """The S2 acceptance gate, printed as a table."""
    from scipy.stats import pearsonr

    data_dir = pathlib.Path("data") / "synthetic"
    manifest = pd.read_csv(data_dir / "manifest.csv")
    k_sheet, k_bump = load_calibration()

    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = ny_f // ny_c

    labels, phys = [], []
    all_nonuniform = True
    n_failures = 0
    rt_ok = True

    print("\nPer-design checks:")
    print(f"{'design':<10} {'inst':>6} {'w/p':>8} {'peak_mV':>9} {'p95_mV':>8} "
          f"{'coarse_pk':>10} {'valid':>7}")
    print("-" * 66)

    for _, row in manifest.iterrows():
        ddir = str(data_dir / row["design_id"])
        d = load_design(ddir)
        failures = validate_design(ddir)
        n_failures += len(failures)

        ir = d.irmap
        for scn in _SCENARIOS:
            if ir[ir["scenario"] == scn]["drop_v"].std() <= 1e-4:
                all_nonuniform = False

        solver_c = coarse_solver_from_design(d, cfg, k_sheet, k_bump)
        coarse_peak = 0.0
        for scn in _SCENARIOS:
            _, I_c = scenario_currents(d, scn, cfg)
            u_c = solver_c.solve(I_c)

            s = ir[ir["scenario"] == scn]
            fine = np.zeros((ny_f, nx_f))
            fine[s["fy"].values.astype(int), s["fx"].values.astype(int)] = s["drop_v"].values
            lab = fine.reshape(ny_c, ratio, nx_c, ratio).max(axis=(1, 3))

            labels.append(lab.ravel())
            phys.append(u_c.ravel())
            coarse_peak = max(coarse_peak, float(u_c.max()))

        # round trip: re-read the written CSV and compare to the loaded Design
        raw = pd.read_csv(data_dir / row["design_id"] / "irmap.csv")
        if float(np.max(np.abs(raw["drop_v"].values - d.irmap["drop_v"].values))) > 1e-12:
            rt_ok = False

        st = d.stats.iloc[0]
        print(f"{row['design_id']:<10} {row['n_instances']:>6} "
              f"{st['strap_width_um']:>4.1f}/{int(st['strap_pitch_um']):<3} "
              f"{ir['drop_v'].max()*1000:>9.2f} "
              f"{np.percentile(ir['drop_v'], 95)*1000:>8.2f} "
              f"{coarse_peak*1000:>10.2f}  "
              f"{('OK' if not failures else 'F(%d)' % len(failures)):>6}")

    L = np.concatenate(labels)
    P = np.concatenate(phys)
    lo_mv, hi_mv = L.min() * 1000, L.max() * 1000
    corr = float(pearsonr(P, L)[0])
    bias_mv = float(np.mean(L - P)) * 1000

    checks = [
        ("label_v spans roughly 0.1 - 90 mV",
         lo_mv < 1.0 and 40 < hi_mv < 90, f"[{lo_mv:.2f}, {hi_mv:.2f}] mV"),
        ("corpus max label > 45 mV", hi_mv > 45, f"{hi_mv:.2f} mV"),
        ("pearson(phys_base_v, label_v) in [0.90, 0.98]",
         0.90 <= corr <= 0.98, f"{corr:.4f}"),
        ("mean(label_v - phys_base_v) in [+5, +11] mV",
         5.0 <= bias_mv <= 11.0, f"{bias_mv:+.3f} mV"),
        ("no irmap is uniform (std > 1e-4)", all_nonuniform,
         "all non-uniform" if all_nonuniform else "SOME UNIFORM"),
        (f"manifest.csv has {cfg['corpus']['n_designs']} rows",
         len(manifest) == cfg["corpus"]["n_designs"], f"{len(manifest)} rows"),
        ("validate_design() zero failures", n_failures == 0, f"{n_failures} failures"),
        ("round trip write -> load to 1e-12", rt_ok, "exact" if rt_ok else "MISMATCH"),
    ]

    print(f"\n{'='*78}")
    print("S2 SANITY TABLE")
    print(f"{'='*78}")
    print(f"{'Check':<50} {'Result':<8} Value")
    print("-" * 78)
    for name, ok, val in checks:
        print(f"{name:<50} {'PASS' if ok else 'FAIL':<8} {val}")
    print("-" * 78)
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(f"{n_ok}/{len(checks)} passed")
    if n_ok < len(checks):
        raise SystemExit(f"S2 GATE FAILED: {len(checks) - n_ok} check(s) failed")


if __name__ == "__main__":
    _cfg = load_config()
    build_corpus(_cfg)
    _run_sanity_checks(_cfg)
