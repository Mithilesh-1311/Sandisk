"""features.py -- the 32 features and the label.

One row = one (design, scenario, coarse tile).  14 designs x 6 scenarios x 576
tiles = 48,384 rows.

Everything here reads PUBLISHED inputs only: design_stats, modules, macros,
instances, bumps, strap_planned, activity.  The as-built strap map, the fine
solve and irmap.csv are label-side and are trapped by audit.py while
build_feature_table() runs.

The coarse solve is NOT reimplemented here.  It comes from the helpers S2 froze:
    solver.coarse_solver_from_design(design, cfg, k_sheet, k_bump)
    design.scenario_currents(design, scenario, cfg) -> (fine, coarse)
    design.load_calibration() -> (k_sheet, k_bump)
so features.py and the S2 gate cannot drift apart.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import rankdata
from tqdm import tqdm

from prism.audit import leakage_trap, unstash_irmap
from prism.design import _SCENARIOS, load_calibration, scenario_currents
from prism.io_csv import load_config, load_design
from prism.solver import coarse_solver_from_design


# ---------------------------------------------------------------------------
# The 32 features, grouped.  Do not add, rename or omit.
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, List[str]] = {
    "phys": ["phys_base_v", "phys_base_s1", "phys_base_s2", "phys_base_rank"],
    "grid": ["grid_weak", "grid_strap_mean", "grid_strap_min", "grid_bumps",
             "grid_dbump_min", "grid_dbump_max"],
    "cur":  ["cur_sum", "cur_max_fine", "cur_s1", "cur_s2", "cur_s4",
             "cur_x_weak", "cur_s2_x_weak"],
    "conc": ["conc_ratio", "conc_top4", "conc_x_weak"],
    "top":  ["top_macro_frac", "top_dmacro", "top_edge_dist", "top_util",
             "top_cells", "top_capden", "top_clkden", "top_seqden"],
    "sw":   ["sw_hhi", "sw_topshare"],
    "scn":  ["scn_power_frac", "scn_weight"],
}

_FEATURES: List[str] = [f for g in FEATURE_GROUPS.values() for f in g]

# Carried for grouping and joins; never fed to the model.
ID_COLUMNS = ["design", "scenario", "ty", "tx"]

# Permanently banned as features (§5B.5).  `hash` in particular is a perfect
# design fingerprint: with 14 designs and GroupKFold by design, it would let
# the model memorise the design instead of learning physics.
BANNED_COLUMNS = ["hash", "config", "design_id", "ts_utc", "lint_rc"]

# Macro coverage is rasterised at this multiple of the fine grid so that
# top_macro_frac is a true union area, not a sum that double-counts overlaps.
_MACRO_SUBGRID = 4


def feature_columns() -> List[str]:
    """The exactly-32 model input columns.

    Identifier columns and every banned design-level column are excluded.
    """
    assert len(_FEATURES) == 32, f"expected 32 features, have {len(_FEATURES)}"
    return list(_FEATURES)


# ---------------------------------------------------------------------------
# Scenario-independent tile geometry and netlist density
# ---------------------------------------------------------------------------

def _static_tile_features(design, cfg: dict, k_sheet: float, k_bump: float) -> Dict:
    """Everything that does not depend on the scenario.

    Computed once per design and reused across its six scenarios: the coarse
    solver and its factorisation, grid strength, macro/edge geometry and
    netlist density.
    """
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = ny_f // ny_c
    n_tiles = ny_c * nx_c

    stats = design.stats.iloc[0]
    die_w, die_h = float(stats["die_w_um"]), float(stats["die_h_um"])
    ctw, cth = die_w / nx_c, die_h / ny_c
    tile_area = ctw * cth

    solver = coarse_solver_from_design(design, cfg, k_sheet, k_bump)
    solver.factorise()

    # grid_weak: coarse solve with 1 A spread uniformly.  Activity-independent,
    # so it is a pure property of the planned grid and the bump array -- the
    # floorplan-time stand-in for effective resistance to the supply.
    grid_weak = solver.solve(np.full((ny_c, nx_c), 1.0 / n_tiles))

    # --- Planned strap density, fine -> per-tile mean and min ---
    sp = design.strap_planned
    planned = np.zeros((ny_f, nx_f))
    planned[sp["fy"].values.astype(int), sp["fx"].values.astype(int)] = sp["density"].values
    blocks = planned.reshape(ny_c, ratio, nx_c, ratio)
    grid_strap_mean = blocks.mean(axis=(1, 3))
    grid_strap_min = blocks.min(axis=(1, 3))

    # --- Bumps ---
    bx = design.bumps["x_um"].values
    by = design.bumps["y_um"].values
    grid_bumps = np.zeros((ny_c, nx_c))
    np.add.at(grid_bumps,
              (np.clip((by / cth).astype(int), 0, ny_c - 1),
               np.clip((bx / ctw).astype(int), 0, nx_c - 1)), 1.0)

    bump_fine = np.zeros((ny_f, nx_f), dtype=bool)
    bump_fine[np.clip((by / (die_h / ny_f)).astype(int), 0, ny_f - 1),
              np.clip((bx / (die_w / nx_f)).astype(int), 0, nx_f - 1)] = True
    # Distance from each fine cell to the nearest bump, in fine tiles.
    dbump = distance_transform_edt(~bump_fine)
    dblocks = dbump.reshape(ny_c, ratio, nx_c, ratio)
    grid_dbump_min = dblocks.min(axis=(1, 3))
    grid_dbump_max = dblocks.max(axis=(1, 3))

    # --- Macros: union coverage on a sub-grid, and distance to nearest macro ---
    sub = _MACRO_SUBGRID
    sy, sx = ny_c * ratio * sub, nx_c * ratio * sub
    macro_sub = np.zeros((sy, sx), dtype=bool)
    macro_fine = np.zeros((ny_f, nx_f), dtype=bool)
    for _, m in design.macros.iterrows():
        x0 = int(np.floor(float(m["x0_um"]) / die_w * sx))
        x1 = int(np.ceil(float(m["x1_um"]) / die_w * sx))
        y0 = int(np.floor(float(m["y0_um"]) / die_h * sy))
        y1 = int(np.ceil(float(m["y1_um"]) / die_h * sy))
        macro_sub[max(0, y0):min(sy, y1), max(0, x0):min(sx, x1)] = True
        fx0 = int(np.floor(float(m["x0_um"]) / die_w * nx_f))
        fx1 = int(np.ceil(float(m["x1_um"]) / die_w * nx_f))
        fy0 = int(np.floor(float(m["y0_um"]) / die_h * ny_f))
        fy1 = int(np.ceil(float(m["y1_um"]) / die_h * ny_f))
        macro_fine[max(0, fy0):min(ny_f, fy1), max(0, fx0):min(nx_f, fx1)] = True

    per_tile_sub = (ratio * sub)
    top_macro_frac = macro_sub.reshape(
        ny_c, per_tile_sub, nx_c, per_tile_sub).mean(axis=(1, 3))

    if macro_fine.any():
        dmacro = distance_transform_edt(~macro_fine)
    else:
        # No macros: every tile is maximally far from one.  Using the die
        # diagonal keeps the feature finite and monotone rather than NaN.
        dmacro = np.full((ny_f, nx_f), float(np.hypot(ny_f, nx_f)))
    top_dmacro = dmacro.reshape(ny_c, ratio, nx_c, ratio).min(axis=(1, 3))

    # --- Distance to die edge, in coarse tiles ---
    ty_i, tx_i = np.meshgrid(np.arange(ny_c), np.arange(nx_c), indexing="ij")
    top_edge_dist = np.minimum.reduce([ty_i, tx_i, ny_c - 1 - ty_i, nx_c - 1 - tx_i]
                                      ).astype(float)

    # --- Netlist density from instances.csv ---
    inst = design.instances
    ix = np.clip((inst["x_um"].values / ctw).astype(int), 0, nx_c - 1)
    iy = np.clip((inst["y_um"].values / cth).astype(int), 0, ny_c - 1)
    flat = iy * nx_c + ix

    top_cells = np.bincount(flat, minlength=n_tiles).astype(float)
    area_sum = np.bincount(flat, weights=inst["area_um2"].values.astype(float),
                           minlength=n_tiles)
    cap_sum = np.bincount(flat, weights=inst["cap_ff"].values.astype(float),
                          minlength=n_tiles)
    clk_sum = np.bincount(flat, weights=inst["is_clk"].values.astype(float),
                          minlength=n_tiles)
    seq_sum = np.bincount(flat, weights=inst["is_seq"].values.astype(float),
                          minlength=n_tiles)

    # An empty tile has no cells, so its clock/sequential FRACTION is 0 by
    # definition rather than 0/0.  This is the correct limit, not an imputation.
    with np.errstate(divide="ignore", invalid="ignore"):
        top_clkden = np.where(top_cells > 0, clk_sum / np.maximum(top_cells, 1), 0.0)
        top_seqden = np.where(top_cells > 0, seq_sum / np.maximum(top_cells, 1), 0.0)

    top_util = (area_sum / tile_area).reshape(ny_c, nx_c)
    top_capden = (cap_sum / tile_area).reshape(ny_c, nx_c)

    # --- Per (tile, module) current at unit activity, for the sw_* features ---
    # Scenario activity is per module, so holding the module decomposition lets
    # every scenario's clock-domain shares be derived by one multiply.
    mod_names = list(design.modules["module"])
    mod_index = {m: i for i, m in enumerate(mod_names)}
    n_mod = len(mod_names)
    vdd = float(stats["vdd_v"])

    inst_mod = inst["module"].map(mod_index).to_numpy()
    counts = np.bincount(inst_mod, minlength=n_mod).astype(float)
    counts[counts == 0] = 1.0
    rated = design.modules["rated_power_mw"].to_numpy(dtype=float)
    unit_i = (rated * 1e-3) / (vdd * counts)          # per instance, activity 1

    tile_mod = np.bincount(flat * n_mod + inst_mod,
                           weights=unit_i[inst_mod],
                           minlength=n_tiles * n_mod).reshape(n_tiles, n_mod)

    # Hard-macro power belongs to its owning module, spread over its tiles.
    for _, m in design.macros.iterrows():
        mi = mod_index.get(m["module"])
        if mi is None:
            continue
        x0 = int(np.clip(float(m["x0_um"]) / ctw, 0, nx_c - 1))
        x1 = int(np.clip(float(m["x1_um"]) / ctw, 0, nx_c - 1))
        y0 = int(np.clip(float(m["y0_um"]) / cth, 0, ny_c - 1))
        y1 = int(np.clip(float(m["y1_um"]) / cth, 0, ny_c - 1))
        cover = [yy * nx_c + xx for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1)]
        tile_mod[cover, mi] += (float(m["power_mw"]) * 1e-3 / vdd) / len(cover)

    domains = list(dict.fromkeys(design.modules["clock_domain"]))
    dom_onehot = np.zeros((n_mod, len(domains)))
    for i, cd in enumerate(design.modules["clock_domain"]):
        dom_onehot[i, domains.index(cd)] = 1.0

    return {
        "solver": solver,
        "grid_weak": grid_weak,
        "grid_strap_mean": grid_strap_mean,
        "grid_strap_min": grid_strap_min,
        "grid_bumps": grid_bumps,
        "grid_dbump_min": grid_dbump_min,
        "grid_dbump_max": grid_dbump_max,
        "top_macro_frac": top_macro_frac,
        "top_dmacro": top_dmacro,
        "top_edge_dist": top_edge_dist,
        "top_util": top_util,
        "top_cells": top_cells.reshape(ny_c, nx_c),
        "top_capden": top_capden,
        "top_clkden": top_clkden.reshape(ny_c, nx_c),
        "top_seqden": top_seqden.reshape(ny_c, nx_c),
        "tile_mod": tile_mod,
        "dom_onehot": dom_onehot,
        "mod_index": mod_index,
        "rated_total": float(rated.sum()),
    }


# ---------------------------------------------------------------------------
# Per (design, scenario) feature rows
# ---------------------------------------------------------------------------

def design_features(design, scenario: str, cfg: dict,
                    static: Optional[Dict] = None,
                    k_sheet: Optional[float] = None,
                    k_bump: Optional[float] = None) -> pd.DataFrame:
    """The 32 features for one (design, scenario): 576 rows, one per coarse tile.

    `static` is the cache from _static_tile_features(); pass it to avoid
    refactorising the mesh once per scenario.
    """
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = ny_f // ny_c

    if static is None:
        if k_sheet is None:
            k_sheet, k_bump = load_calibration()
        static = _static_tile_features(design, cfg, k_sheet, k_bump)

    # --- Currents (published inputs only) ---
    fine_I, coarse_I = scenario_currents(design, scenario, cfg)

    # --- phys_*: the physics prior ---
    phys_base_v = static["solver"].solve(coarse_I)
    phys_base_s1 = gaussian_filter(phys_base_v, sigma=1.0)
    phys_base_s2 = gaussian_filter(phys_base_v, sigma=2.0)
    # Percentile rank within this (design, scenario) map.  Explicitly NOT a
    # global rank: a corpus-wide rank would encode absolute design scale.
    phys_base_rank = (rankdata(phys_base_v.ravel()) / phys_base_v.size).reshape(ny_c, nx_c)

    # --- cur_* ---
    grid_weak = static["grid_weak"]
    cur_sum = coarse_I
    cur_s1 = gaussian_filter(cur_sum, sigma=1.0)
    cur_s2 = gaussian_filter(cur_sum, sigma=2.0)
    cur_s4 = gaussian_filter(cur_sum, sigma=4.0)

    fine_blocks = fine_I.reshape(ny_c, ratio, nx_c, ratio)
    cur_max_fine = fine_blocks.max(axis=(1, 3))

    # --- conc_*: sub-tile concentration, the largest single source of the
    # physics baseline's error.  A tile carrying no current is perfectly
    # uniform by definition: ratio 1.0 and top-4 share 4/16.  Those are limits,
    # not imputed values.
    n_sub = ratio * ratio
    fine_flat = fine_blocks.transpose(0, 2, 1, 3).reshape(ny_c, nx_c, n_sub)
    tile_total = fine_flat.sum(axis=2)
    tile_mean = tile_total / n_sub
    has_cur = tile_total > 0
    conc_ratio = np.where(has_cur, cur_max_fine / np.where(has_cur, tile_mean, 1.0), 1.0)
    top4 = np.sort(fine_flat, axis=2)[:, :, -4:].sum(axis=2)
    conc_top4 = np.where(has_cur, top4 / np.where(has_cur, tile_total, 1.0), 4.0 / n_sub)

    # --- sw_*: simultaneous switching by clock domain ---
    act = design.activity[design.activity["scenario"] == scenario]
    act_map = dict(zip(act["module"], act["activity"]))
    act_vec = np.array([float(act_map.get(m, 0.0)) for m in static["mod_index"]])
    tile_dom = (static["tile_mod"] * act_vec) @ static["dom_onehot"]
    dom_total = tile_dom.sum(axis=1)
    active = dom_total > 0
    shares = np.where(active[:, None],
                      tile_dom / np.where(active, dom_total, 1.0)[:, None], 0.0)
    # An idle tile has no simultaneous switching; 0.0 encodes that, and keeps
    # it distinguishable from a tile genuinely owned by one domain (hhi 1.0).
    sw_hhi = (shares ** 2).sum(axis=1).reshape(ny_c, nx_c)
    sw_topshare = shares.max(axis=1).reshape(ny_c, nx_c)

    # --- scn_* ---
    rated = dict(zip(design.modules["module"], design.modules["rated_power_mw"]))
    scn_power = sum(float(rated.get(m, 0.0)) * float(a)
                    for m, a in zip(act["module"], act["activity"]))
    scn_power_frac = scn_power / static["rated_total"] if static["rated_total"] > 0 else 0.0
    scn_weight = float(act["mission_weight"].iloc[0])

    ty_i, tx_i = np.meshgrid(np.arange(ny_c), np.arange(nx_c), indexing="ij")

    data = {
        "design": design.design_id,
        "scenario": scenario,
        "ty": ty_i.ravel(),
        "tx": tx_i.ravel(),
        "phys_base_v": phys_base_v.ravel(),
        "phys_base_s1": phys_base_s1.ravel(),
        "phys_base_s2": phys_base_s2.ravel(),
        "phys_base_rank": phys_base_rank.ravel(),
        "grid_weak": grid_weak.ravel(),
        "grid_strap_mean": static["grid_strap_mean"].ravel(),
        "grid_strap_min": static["grid_strap_min"].ravel(),
        "grid_bumps": static["grid_bumps"].ravel(),
        "grid_dbump_min": static["grid_dbump_min"].ravel(),
        "grid_dbump_max": static["grid_dbump_max"].ravel(),
        "cur_sum": cur_sum.ravel(),
        "cur_max_fine": cur_max_fine.ravel(),
        "cur_s1": cur_s1.ravel(),
        "cur_s2": cur_s2.ravel(),
        "cur_s4": cur_s4.ravel(),
        "cur_x_weak": (cur_sum * grid_weak).ravel(),
        "cur_s2_x_weak": (cur_s2 * grid_weak).ravel(),
        "conc_ratio": conc_ratio.ravel(),
        "conc_top4": conc_top4.ravel(),
        "conc_x_weak": (conc_ratio * grid_weak).ravel(),
        "top_macro_frac": static["top_macro_frac"].ravel(),
        "top_dmacro": static["top_dmacro"].ravel(),
        "top_edge_dist": static["top_edge_dist"].ravel(),
        "top_util": static["top_util"].ravel(),
        "top_cells": static["top_cells"].ravel(),
        "top_capden": static["top_capden"].ravel(),
        "top_clkden": static["top_clkden"].ravel(),
        "top_seqden": static["top_seqden"].ravel(),
        "sw_hhi": sw_hhi.ravel(),
        "sw_topshare": sw_topshare.ravel(),
        "scn_power_frac": scn_power_frac,
        "scn_weight": scn_weight,
    }
    return pd.DataFrame(data, columns=ID_COLUMNS + _FEATURES)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def add_labels(df: pd.DataFrame, irmap_fine: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach `label_v` -- the MAX fine-grid drop inside each coarse tile.

    Not the mean.  A hotspot is a local worst case; averaging it away makes the
    task easy and the result useless.

    `irmap_fine` is one design's irmap in long format (scenario, fy, fx, drop_v).
    This is label-side data, so it must be called OUTSIDE leakage_trap().
    """
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = ny_f // ny_c

    label_by_scn: Dict[str, np.ndarray] = {}
    for scn, sub in irmap_fine.groupby("scenario", sort=False):
        fine = np.zeros((ny_f, nx_f))
        fine[sub["fy"].values.astype(int), sub["fx"].values.astype(int)] = (
            sub["drop_v"].values
        )
        label_by_scn[scn] = fine.reshape(ny_c, ratio, nx_c, ratio).max(axis=(1, 3))

    out = df.copy()
    tile = out["ty"].to_numpy() * nx_c + out["tx"].to_numpy()
    labels = np.empty(len(out))
    for scn, idx in out.groupby("scenario", sort=False).indices.items():
        labels[idx] = label_by_scn[scn].ravel()[tile[idx]]
    out["label_v"] = labels
    return out


# ---------------------------------------------------------------------------
# Corpus feature table
# ---------------------------------------------------------------------------

def build_feature_table(cfg: dict, data_dir: str = "data/synthetic",
                        out_path: str = "out/features.csv") -> pd.DataFrame:
    """Build out/features.csv for a whole corpus.

    Feature extraction runs inside audit.leakage_trap(); labels are attached
    afterwards, outside it.
    """
    k_sheet, k_bump = load_calibration()
    manifest = pd.read_csv(pathlib.Path(data_dir) / "manifest.csv")

    parts = []
    trapped_surfaces: List[str] = []
    for _, row in tqdm(list(manifest.iterrows()), total=len(manifest), desc="features"):
        ddir = str(pathlib.Path(data_dir) / row["path"])

        with leakage_trap() as guarded:
            trapped_surfaces = guarded
            design = load_design(ddir)
            static = _static_tile_features(design, cfg, k_sheet, k_bump)
            fdf = pd.concat(
                [design_features(design, scn, cfg, static=static) for scn in _SCENARIOS],
                ignore_index=True,
            )

        # Outside the trap: labels are label-side by definition.
        unstash_irmap(design)
        parts.append(add_labels(fdf, design.irmap, cfg))

    table = pd.concat(parts, ignore_index=True)

    out_file = pathlib.Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_file, index=False)
    table.attrs["trapped_surfaces"] = trapped_surfaces
    return table


# ---------------------------------------------------------------------------
# Advanced Layout Feature Extractors (ICCAD 2023 Extensions)
# ---------------------------------------------------------------------------

def compute_effective_distance(bumps_grid: np.ndarray,
                               x_pitch_ratio: float = 1.0,
                               y_pitch_ratio: float = 1.0) -> np.ndarray:
    """Compute 2D anisotropic Manhattan effective distance to nearest power bumps.

    Unlike simple Euclidean distance, on-chip current flows along orthogonal metal
    tracks (horizontal lower straps vs vertical upper trunks). This function
    weights X and Y distances by directional pitch/sheet resistance ratios.

    Args:
        bumps_grid: 2D binary matrix (ny, nx) where 1 indicates C4 bump location.
        x_pitch_ratio: Directional resistance/pitch weighting along X axis.
        y_pitch_ratio: Directional resistance/pitch weighting along Y axis.

    Returns:
        2D float matrix (ny, nx) of effective anisotropic resistance distances.
    """
    ny, nx = bumps_grid.shape
    bump_locs = np.argwhere(bumps_grid > 0)
    if len(bump_locs) == 0:
        return np.ones((ny, nx), dtype=float) * float(max(ny, nx))

    yy, xx = np.indices((ny, nx))
    eff_dist = np.full((ny, nx), np.inf, dtype=float)

    for by, bx in bump_locs:
        d = y_pitch_ratio * np.abs(yy - by) + x_pitch_ratio * np.abs(xx - bx)
        eff_dist = np.minimum(eff_dist, d)

    return eff_dist


def compute_pdn_density(macro_mask: np.ndarray,
                        base_layers: int = 3,
                        halo_tiles: int = 1) -> np.ndarray:
    """Compute local PDN track and via availability density rho_pdn in [0, base_layers].

    Standard cell power rails (M1/M2) and through-vias cannot traverse large
    SRAM/hard macros. Near macro keepout boundaries, PDN layer density degrades,
    causing localized resistance bottlenecks and via starvation.

    Args:
        macro_mask: 2D binary array (ny, nx) where 1 indicates macro occupancy.
        base_layers: Nominal available PDN metal routing layers (default: 3).
        halo_tiles: Radius of macro boundary via degradation halo.

    Returns:
        2D float matrix (ny, nx) of available PDN density.
    """
    ny, nx = macro_mask.shape
    density = np.full((ny, nx), float(base_layers), dtype=float)

    # Macros block lower rails
    density[macro_mask > 0] = 0.0

    # Macro boundary halo degradation
    if halo_tiles > 0 and np.any(macro_mask > 0):
        dist_to_macro = distance_transform_edt(macro_mask == 0)
        halo_mask = (dist_to_macro > 0) & (dist_to_macro <= halo_tiles)
        density[halo_mask] = float(base_layers) * 0.5

    return density


# ---------------------------------------------------------------------------
# S3 gate
# ---------------------------------------------------------------------------

def _run_feature_gate(cfg: dict, table: pd.DataFrame) -> None:
    """Print the S3 gate and the three diagnostics that decide the S4 design."""
    from scipy.stats import spearmanr

    feats = feature_columns()
    budget_v = cfg["electrical"]["vdd"] * cfg["electrical"]["ir_budget_frac"]

    X = table[feats].to_numpy(dtype=float)
    n_bad = int((~np.isfinite(X)).sum())

    print()
    print(f"rows = {len(table)}    "
          f"({table['design'].nunique()} designs x {table['scenario'].nunique()} "
          f"scenarios x {table['ty'].nunique() * table['tx'].nunique()} tiles)")
    print(f"features = {len(feats)}")
    print(f"NaN/inf count = {n_bad}")

    banned_present = [c for c in BANNED_COLUMNS if c in feats]
    print(f"LEAKAGE AUDIT: {'PASS' if not banned_present else 'FAIL ' + str(banned_present)}")
    print(f"  trapped during extraction: {', '.join(table.attrs.get('trapped_surfaces', []))}")
    print(f"  banned columns in feature_columns(): "
          f"{banned_present if banned_present else 'none'}")

    if n_bad:
        cols = [c for c in feats if not np.isfinite(table[c].to_numpy(dtype=float)).all()]
        raise SystemExit(f"S3 GATE FAILED: non-finite values in {cols}")

    # ---------------- Diagnostic 1: violation base rate ----------------
    viol = table["label_v"] > budget_v
    per_design = table.groupby("design")["label_v"].max() > budget_v
    print(f"\n{'='*74}")
    print(f"DIAGNOSTIC 1  VIOLATION BASE RATE  (budget {budget_v*1000:.0f} mV)")
    print(f"{'='*74}")
    print(f"  violating rows        : {int(viol.sum()):,} / {len(table):,} "
          f"= {viol.mean()*100:.2f}%")
    print(f"  designs with >=1 viol : {int(per_design.sum())} / {len(per_design)}")
    print("  per scenario:")
    for scn, sub in table.groupby("scenario", sort=False):
        v = sub["label_v"] > budget_v
        print(f"    {scn:<14} {int(v.sum()):>6,} / {len(sub):,} = {v.mean()*100:>6.2f}%")
    if viol.mean() < 0.02:
        print("  NOTE: base rate under 2% -- violation F1 will be noisy and S4 "
              "must weight the positive class.")

    # ---------------- Diagnostic 2: residual structure ----------------
    resid = (table["label_v"] - table["phys_base_v"]).to_numpy()
    mu, sd = float(resid.mean()), float(resid.std())
    centred = resid - table.groupby("design")["label_v"].transform("mean").to_numpy() \
                    + table.groupby("design")["phys_base_v"].transform("mean").to_numpy()
    ratio = sd / abs(mu) if mu != 0 else float("inf")
    print(f"\n{'='*74}")
    print("DIAGNOSTIC 2  RESIDUAL STRUCTURE   (residual = label_v - phys_base_v)")
    print(f"{'='*74}")
    print(f"  mean(residual)                    = {mu*1000:+9.4f} mV")
    print(f"  std(residual)                     = {sd*1000:9.4f} mV")
    print(f"  std/|mean|                        = {ratio:9.4f}")
    print(f"  std(residual) after removing the")
    print(f"    per-design mean                 = {float(centred.std())*1000:9.4f} mV "
          f"({float(centred.std())/sd*100:.1f}% of raw std)")
    # How much of the residual is a plain affine rescale of the physics? This
    # decides what S4's baseline row has to be.
    p_v = table["phys_base_v"].to_numpy()
    slope, intercept = np.polyfit(p_v, resid, 1)
    affine = intercept + slope * p_v
    r2_affine = 1.0 - ((resid - affine) ** 2).sum() / ((resid - resid.mean()) ** 2).sum()
    print(f"  best affine fit  residual ~ {intercept*1000:+.4f} mV "
          f"{slope:+.4f} * phys_base_v")
    print(f"    R2 of that affine fit           = {r2_affine:9.4f}")
    print(f"    std(residual - affine fit)      = {float((resid-affine).std())*1000:9.4f} mV "
          f"({(1-(resid-affine).std()/sd)*100:.1f}% of the spread removed)")
    if ratio < 0.5:
        print("  VERDICT: residual is mostly a CONSTANT OFFSET. S4 must beat a")
        print("           physics+constant baseline, not just the raw physics.")
    else:
        print("  VERDICT: residual is NOT a constant offset (std/|mean| = "
              f"{ratio:.2f}), and the")
        print("           per-design mean explains almost none of it "
              f"({100-float(centred.std())/sd*100:.1f}%).")
        print(f"           But an affine rescale of the physics alone captures "
              f"R2={r2_affine:.2f},")
        print("           so S4's baseline row must be PHYSICS+AFFINE, not")
        print("           physics+constant. The hybrid has to beat that.")

    # ---------------- Diagnostic 3: group Spearman vs residual ----------------
    print(f"\n{'='*74}")
    print("DIAGNOSTIC 3  SPEARMAN vs RESIDUAL, BY FEATURE GROUP")
    print(f"{'='*74}")
    print(f"  {'group':<6} {'n':>3} {'max|rho|':>9} {'mean|rho|':>10}  strongest feature")
    print("  " + "-" * 68)
    for gname, gfeats in FEATURE_GROUPS.items():
        rhos = {}
        for f in gfeats:
            col = table[f].to_numpy(dtype=float)
            if np.allclose(col, col[0]):
                rhos[f] = 0.0          # constant column: rho undefined, report 0
            else:
                rhos[f] = float(spearmanr(col, resid).statistic)
        best = max(rhos, key=lambda k: abs(rhos[k]))
        print(f"  {gname:<6} {len(gfeats):>3} {abs(rhos[best]):>9.4f} "
              f"{np.mean([abs(v) for v in rhos.values()]):>10.4f}  "
              f"{best} ({rhos[best]:+.4f})")
    print()


if __name__ == "__main__":
    _cfg = load_config()
    _table = build_feature_table(_cfg)
    _run_feature_gate(_cfg, _table)
