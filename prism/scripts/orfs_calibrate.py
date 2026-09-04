"""S8 Task 2 -- fit the two-parameter resistive mesh to real PDNSim IR maps.

The question real data is actually able to answer:
*does a two-constant resistive mesh reproduce a signoff tool's voltage map?*

ONE free parameter
------------------
k_bump is tied at 10 x k_sheet by construction (out/calibration.json), and
solver property 2 says scaling every conductance by s scales U by 1/s.  So with
k_sheet = 1 solved once, the drop at any other k_sheet is U1 / k_sheet, and the
least-squares fit through the origin

    k_sheet = sum(U1^2) / sum(U1 * L)

is closed form -- no search, no optimiser, no local minimum.

What is actually independent in this data
-----------------------------------------
prism/orfs.py (lines 226-252) builds all six scenario maps from ONE PDNSim
as-run solve, `irmap[s] = clip(asrun * scale[s], 0, 0.199)`, and injects a faint
synthetic die-position ramp into any map whose std falls below 1.6e-4 V.  So:

  * there is exactly ONE independent signoff map per design, not six;
  * fitting per scenario would measure orfs.py's scalar rescale, not physics;
  * on the four run_0x designs the injection fired on every map including the
    reference, so their maps are partly (run_01: 88% of variance) a synthetic
    ramp and are excluded from the headline.  See scripts/orfs_irmap_audit.py
    and scripts/orfs_gradient_check.py.

The headline fit is therefore ONE fit per design: the mesh under the as-run
condition against the as-run map.  The as-run condition is every module at its
rated power (activity = 1), because orfs.py normalises rated_power_mw to the
flow's own reported total power -- the same power PDNSim solved.  That removes
the scenario-rescale confound entirely.

Targets
-------
A coarse mesh node carries a tile's average potential, so the physically matched
target is the tile MEAN of the 96x96 map.  The ML pipeline's label is the tile
MAX (a hotspot is a local worst case), so both are fitted and both reported.

Writes out/orfs_calibration.csv and prints the tables.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr, spearmanr

from prism.design import _SCENARIOS, scenario_currents
from prism.io_csv import load_config, load_design
from prism.solver import coarse_solver_from_design

DATA_DIR = pathlib.Path("data") / "orfs"
OUT_CSV = pathlib.Path("out") / "orfs_calibration.csv"

BUMP_TO_SHEET = 10.0                 # k_bump is not a second degree of freedom
TARGETS = ["tile_mean", "tile_max"]
REF_SCENARIO = "seq_write"           # largest scale factor -> least rescaled
REF_SCALE = 1.2337662337662337       # orfs.py: mean(_ACT[seq_write]) / mean(_ACT[seq_read])
INJECTION_STD_V = 1.6e-4             # orfs.py's non-uniformity trigger


def _fine_map(irmap: pd.DataFrame, scenario: str, cfg: dict) -> np.ndarray:
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    sub = irmap[irmap["scenario"] == scenario]
    m = np.zeros((ny_f, nx_f))
    m[sub["fy"].to_numpy(int), sub["fx"].to_numpy(int)] = sub["drop_v"].to_numpy(float)
    return m


def _to_coarse(fine: np.ndarray, cfg: dict) -> dict:
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = cfg["grid"]["ny_fine"] // ny_c
    b = fine.reshape(ny_c, ratio, nx_c, ratio)
    return {"tile_mean": b.mean(axis=(1, 3)).ravel(),
            "tile_max": b.max(axis=(1, 3)).ravel()}


def _unit_activity_coarse_current(d, cfg: dict) -> np.ndarray:
    """Coarse current map [A] with every module at its rated power.

    Same construction as design.scenario_currents(), with activity fixed at 1:
    a module's rated power is shared equally over its instances and dropped at
    the design's own vdd.  This is the as-run condition -- orfs.py normalises
    rated_power_mw to the flow's reported total power, which is the power
    PDNSim solved for.
    """
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    stats = d.stats.iloc[0]
    die_w, die_h = float(stats["die_w_um"]), float(stats["die_h_um"])
    vdd = float(stats["vdd_v"])

    inst = d.instances
    mods = inst["module"].values
    rated = dict(zip(d.modules["module"], d.modules["rated_power_mw"]))
    uniq, counts = np.unique(mods, return_counts=True)
    cnt = dict(zip(uniq, counts))
    per_module = {m: (float(rated.get(m, 0.0)) * 1e-3) / (vdd * float(cnt[m]))
                  for m in uniq}
    inst_i = pd.Series(mods).map(per_module).to_numpy(dtype=float)

    coarse = np.zeros((ny_c, nx_c))
    np.add.at(coarse,
              (np.clip((inst["y_um"].values / (die_h / ny_c)).astype(int), 0, ny_c - 1),
               np.clip((inst["x_um"].values / (die_w / nx_c)).astype(int), 0, nx_c - 1)),
              inst_i)
    return coarse


def _dbump_coarse(d, cfg: dict) -> np.ndarray:
    """Distance from each coarse tile to the nearest bump, in fine tiles.

    Reported alongside the fit as the null model: if the PDNSim map is simply
    'far from a bump is worse', a mesh that fails to beat this is adding nothing.
    """
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ratio = ny_f // ny_c
    stats = d.stats.iloc[0]
    die_w, die_h = float(stats["die_w_um"]), float(stats["die_h_um"])
    bf = np.zeros((ny_f, nx_f), dtype=bool)
    bf[np.clip((d.bumps["y_um"].values / (die_h / ny_f)).astype(int), 0, ny_f - 1),
       np.clip((d.bumps["x_um"].values / (die_w / nx_f)).astype(int), 0, nx_f - 1)] = True
    db = distance_transform_edt(~bf)
    return db.reshape(ny_c, ratio, nx_c, ratio).mean(axis=(1, 3)).ravel()


def _uniform_supply_solver(d, cfg: dict, k_sheet: float, k_bump: float):
    """The same mesh, but every tile tied to the supply instead of a bump array.

    bumps.csv on this corpus is not a 2-D bump array: every entry lies on a
    single column at x = die_w/2 (gcd has one point at the die centre).  The
    PDN those designs were actually solved on is a dense M1-M4-M7 strap grid
    covering the whole die, which reaches every tile.  This builds that supply
    model so the fit separates "the mesh abstraction is wrong" from "the
    delivered supply geometry is wrong".

    Uses the frozen PDNSolver constructor only; solver.py is not modified.
    """
    from prism.solver import PDNSolver, design_conductances

    ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    ratio = ny_f // ny_c
    stats = d.stats.iloc[0]
    sheet_cond, bump_cond = design_conductances(stats, k_sheet, k_bump)

    straps = d.strap_planned
    planned_fine = np.zeros((ny_f, nx_f))
    planned_fine[straps["fy"].to_numpy(int), straps["fx"].to_numpy(int)] = (
        straps["density"].to_numpy(float))
    planned_coarse = planned_fine.reshape(ny_c, ratio, nx_c, ratio).mean(axis=(1, 3))

    mask = np.ones((ny_c, nx_c), dtype=bool)
    return PDNSolver(ny_c, nx_c, sheet_cond, bump_cond, mask,
                     strap_density=planned_coarse)


def _fit_k(u1: np.ndarray, label: np.ndarray) -> float:
    denom = float(np.dot(u1, label))
    if denom <= 0:
        return float("nan")
    return float(np.dot(u1, u1) / denom)


def _metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """R2, MAE/RMSE/bias in mV, rank agreement.  bias = mean(pred - label)."""
    resid = pred - label
    ss_tot = float(np.sum((label - label.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot
    const = label.std() < 1e-15 or pred.std() < 1e-15
    return {
        "r2": r2,
        "mae_mv": float(np.mean(np.abs(resid))) * 1e3,
        "rmse_mv": float(np.sqrt(np.mean(resid ** 2))) * 1e3,
        "bias_mv": float(np.mean(resid)) * 1e3,
        "spearman": float("nan") if const else float(spearmanr(pred, label).statistic),
        "pearson": float("nan") if const else float(pearsonr(pred, label)[0]),
        "label_max_mv": float(label.max()) * 1e3,
        "n": int(label.size),
    }


def main() -> None:
    cfg = load_config()
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    designs, u_asrun, u_scn, lab_asrun, lab_scn = {}, {}, {}, {}, {}
    clean, dbump = {}, {}

    for did in manifest["design_id"]:
        d = load_design(str(DATA_DIR / did))
        designs[did] = d
        solver = coarse_solver_from_design(d, cfg, 1.0, BUMP_TO_SHEET)
        solver.factorise()

        ref_fine = _fine_map(d.irmap, REF_SCENARIO, cfg)
        # A reference map whose std sits under orfs.py's trigger had the
        # synthetic ramp injected into it; its spatial content is not signoff.
        clean[did] = bool(ref_fine.std() >= INJECTION_STD_V)
        lab_asrun[did] = _to_coarse(ref_fine / REF_SCALE, cfg)
        u_asrun[did] = solver.solve(_unit_activity_coarse_current(d, cfg)).ravel()
        dbump[did] = _dbump_coarse(d, cfg)

        for scn in _SCENARIOS:
            _, ci = scenario_currents(d, scn, cfg)
            u_scn[(did, scn)] = solver.solve(ci).ravel()
            lab_scn[(did, scn)] = _to_coarse(_fine_map(d.irmap, scn, cfg), cfg)
        print(f"  solved {did}  (reference map signoff-clean: {clean[did]})")

    rows = []

    def record(scope, design, scenario, target, k, u, l, is_clean):
        rows.append(dict(fit_scope=scope, design=design, scenario=scenario,
                         target=target, signoff_clean=is_clean,
                         k_sheet=k, k_bump=k * BUMP_TO_SHEET,
                         **_metrics(u / k, l)))

    clean_ids = [d_ for d_ in manifest["design_id"] if clean[d_]]

    for t in TARGETS:
        # --- headline: one fit per design, mesh as-run vs PDNSim as-run ------
        for did in manifest["design_id"]:
            u, l = u_asrun[did], lab_asrun[did][t]
            record("asrun", did, REF_SCENARIO + "/scale", t, _fit_k(u, l), u, l, clean[did])

        # --- one global constant across the signoff-clean designs -----------
        u_all = np.concatenate([u_asrun[d_] for d_ in clean_ids])
        l_all = np.concatenate([lab_asrun[d_][t] for d_ in clean_ids])
        k_glob = _fit_k(u_all, l_all)
        record("global", "CLEAN_POOLED", "all", t, k_glob, u_all, l_all, True)
        for did in clean_ids:
            record("global_applied", did, "all", t, k_glob,
                   u_asrun[did], lab_asrun[did][t], True)

        # --- per scenario, kept for completeness -----------------------------
        for did in manifest["design_id"]:
            for scn in _SCENARIOS:
                u, l = u_scn[(did, scn)], lab_scn[(did, scn)][t]
                record("per_scenario", did, scn, t, _fit_k(u, l), u, l, clean[did])

    # --- null models -----------------------------------------------------
    # Two things the mesh combines: proximity to the supply (a spreading
    # resistance term) and local current density.  Scoring each alone against
    # the signoff map says which one PDNSim's structure actually follows, and
    # whether the mesh is adding anything over the simpler of the two.
    for did in manifest["design_id"]:
        cur = _unit_activity_coarse_current(designs[did], cfg).ravel()
        for t in TARGETS:
            l = lab_asrun[did][t]
            for name, x in (("null_dbump", dbump[did]), ("null_current", cur)):
                const = x.std() < 1e-15
                rows.append(dict(
                    fit_scope=name, design=did, scenario=REF_SCENARIO + "/scale",
                    target=t, signoff_clean=clean[did], k_sheet=np.nan, k_bump=np.nan,
                    r2=np.nan, mae_mv=np.nan, rmse_mv=np.nan, bias_mv=np.nan,
                    spearman=np.nan if const else float(spearmanr(x, l).statistic),
                    pearson=np.nan if const else float(pearsonr(x, l)[0]),
                    label_max_mv=float(l.max()) * 1e3, n=int(l.size)))

    # --- the same mesh with a whole-die supply instead of a bump array -----
    for did in manifest["design_id"]:
        d = designs[did]
        sv = _uniform_supply_solver(d, cfg, 1.0, BUMP_TO_SHEET)
        sv.factorise()
        u = sv.solve(_unit_activity_coarse_current(d, cfg)).ravel()
        for t in TARGETS:
            l = lab_asrun[did][t]
            k = _fit_k(u, l)
            record("uniform_supply", did, REF_SCENARIO + "/scale", t, k, u, l, clean[did])

    # --- sensitivity to the one constant that is HELD, not fitted ----------
    # k_sheet is a pure scale, so it cannot change the shape of the predicted
    # map; the shape is set entirely by the bump:sheet conductance ratio, which
    # is held at 10 by construction.  Sweeping it separates "the mesh is
    # miscalibrated in a held constant" from "the mesh is structurally wrong".
    for did in manifest["design_id"]:
        d = designs[did]
        cur = _unit_activity_coarse_current(d, cfg)
        for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0]:
            sv = coarse_solver_from_design(d, cfg, 1.0, ratio)
            sv.factorise()
            u = sv.solve(cur).ravel()
            for t in TARGETS:
                l = lab_asrun[did][t]
                k = _fit_k(u, l)
                m = _metrics(u / k, l)
                rows.append(dict(fit_scope="ratio_sweep", design=did,
                                 scenario=f"bump_ratio={ratio:g}", target=t,
                                 signoff_clean=clean[did], k_sheet=k,
                                 k_bump=k * ratio, **m))

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:9.4f}"
    for t in TARGETS:
        print(f"\n=== MESH vs PDNSim, as-run condition, target = {t} ===")
        sub = df[(df.fit_scope == "asrun") & (df.target == t)]
        head = sub[sub.signoff_clean]
        print("  designs whose signoff map is genuine PDNSim content:")
        print(head[["design", "k_sheet", "r2", "mae_mv", "rmse_mv", "bias_mv",
                    "spearman", "label_max_mv"]].to_string(index=False, float_format=fmt))
        excl = sub[~sub.signoff_clean]
        print("  EXCLUDED -- orfs.py injected a synthetic ramp into these maps:")
        print(excl[["design", "k_sheet", "r2", "mae_mv", "spearman",
                    "label_max_mv"]].to_string(index=False, float_format=fmt))
        g = df[(df.fit_scope == "global") & (df.target == t)].iloc[0]
        print(f"  one global constant over the clean designs: k_sheet = {g.k_sheet:.3f}, "
              f"k_bump = {g.k_bump:.3f}, pooled R2 = {g.r2:.4f}, MAE = {g.mae_mv:.4f} mV")
        ga = df[(df.fit_scope == "global_applied") & (df.target == t)]
        print(ga[["design", "r2", "mae_mv", "bias_mv"]].to_string(index=False, float_format=fmt))

    print("\n=== what the PDNSim map actually follows (Spearman, target = tile_mean) ===")
    sel = lambda sc: df[(df.fit_scope == sc) & (df.target == "tile_mean")][
        ["design", "spearman"]].rename(columns={"spearman": sc})
    cmp = (df[(df.fit_scope == "asrun") & (df.target == "tile_mean")]
           [["design", "signoff_clean", "spearman"]]
           .rename(columns={"spearman": "mesh"})
           .merge(sel("null_dbump"), on="design")
           .merge(sel("null_current"), on="design"))
    print(cmp.to_string(index=False, float_format=fmt))

    print("\n=== same mesh, supply model corrected to whole-die strap grid "
          "(target = tile_mean) ===")
    us = df[(df.fit_scope == "uniform_supply") & (df.target == "tile_mean")]
    ar = df[(df.fit_scope == "asrun") & (df.target == "tile_mean")]
    both = (ar[["design", "signoff_clean", "r2", "spearman"]]
            .merge(us[["design", "k_sheet", "r2", "spearman"]], on="design",
                   suffixes=("_bumpcol", "_gridsupply")))
    print(both.to_string(index=False, float_format=fmt))

    print("\n=== sensitivity to the HELD bump:sheet ratio "
          "(signoff-clean designs, target = tile_mean, R2) ===")
    sw = df[(df.fit_scope == "ratio_sweep") & (df.target == "tile_mean")
            & df.signoff_clean]
    print(sw.pivot(index="design", columns="scenario", values="r2")
            .to_string(float_format=lambda v: f"{v:8.3f}"))
    print("  same sweep, Spearman:")
    print(sw.pivot(index="design", columns="scenario", values="spearman")
            .to_string(float_format=lambda v: f"{v:8.3f}"))

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
