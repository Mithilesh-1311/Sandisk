"""io_csv.py — THE ONLY WAY DATA ENTERS.

Load, validate, and write the CSV-based data contract defined in
docs/DATA_SCHEMA.md.  Every load goes through this module.  Silent
coercion is banned; a wrong file must stop the run, not produce quiet
garbage.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: Optional[str] = None) -> dict:
    """Load YAML config.  Defaults to config/default.yaml."""
    p = pathlib.Path(path) if path else _CFG_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Design data container
# ---------------------------------------------------------------------------

@dataclass
class Design:
    """Validated, typed container for one design directory."""
    design_id: str
    design_dir: str

    # design_stats (1-row scalars)
    stats: pd.DataFrame

    # per-file DataFrames
    modules: pd.DataFrame
    macros: pd.DataFrame
    instances: pd.DataFrame
    bumps: pd.DataFrame
    strap_planned: pd.DataFrame
    activity: pd.DataFrame
    paths: pd.DataFrame
    irmap: pd.DataFrame

    # optional
    toggle: Optional[pd.DataFrame] = None

    # Resolved by design.from_csv() at load time (S8).  On synthetic data these
    # simply echo the CSVs; on real ORFS data they carry the per-design supply
    # voltage, the budget that follows from it, a bump pitch measured from the
    # bump array when the header field is absent, and whether the scenario
    # activity is MEASURED from RTL toggle counts or assumed from the table.
    vdd_v: Optional[float] = None
    budget_v: Optional[float] = None
    bump_pitch_um: Optional[float] = None
    macro_sentinel_tiles: Optional[float] = None
    activity_source: str = "assumed"
    assumptions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Required files and columns
# ---------------------------------------------------------------------------

_REQUIRED_FILES = {
    "design_stats.csv": [
        "design_id", "config", "hash", "cells", "flops", "xor_cells",
        "chip_area_um2", "seq_area_um2", "seq_pct", "lint_rc", "ts_utc",
        "die_w_um", "die_h_um", "vdd_v", "clock_period_ns", "core_util",
        "bump_pitch_um", "strap_pitch_um", "strap_width_um", "pdn_layers",
    ],
    "modules.csv": ["module", "clock_domain", "power_domain", "rated_power_mw"],
    "macros.csv": ["macro_id", "module", "x0_um", "y0_um", "x1_um", "y1_um", "power_mw"],
    "instances.csv": [
        "inst_id", "inst_name", "module", "cell_type", "x_um", "y_um",
        "area_um2", "cap_ff", "is_seq", "is_clk", "is_macro",
    ],
    "bumps.csv": ["bump_id", "x_um", "y_um"],
    "strap_planned.csv": ["fy", "fx", "density"],
    "activity.csv": ["scenario", "module", "activity", "mission_weight"],
    "paths.csv": [
        "path_id", "endpoint", "clock_domain", "slack_ns", "delay_ns", "inst_ids",
    ],
    "irmap.csv": ["scenario", "fy", "fx", "drop_v"],
}

_OPTIONAL_FILES = {
    "toggle.csv": ["scenario", "inst_id", "toggle_rate"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv(filepath: str) -> pd.DataFrame:
    """Read CSV or CSV.GZ, sniffing the extension."""
    p = pathlib.Path(filepath)
    if p.suffix == ".gz":
        return pd.read_csv(p, compression="gzip")
    return pd.read_csv(p)


def _find_csv(design_dir: str, basename: str) -> Optional[str]:
    """Return the path to `basename` or `basename.gz` inside design_dir, or None."""
    d = pathlib.Path(design_dir)
    plain = d / basename
    if plain.exists():
        return str(plain)
    gz = d / (basename + ".gz")
    if gz.exists():
        return str(gz)
    return None


# ---------------------------------------------------------------------------
# validate_design — returns a list of failure strings (empty == OK)
# ---------------------------------------------------------------------------

def validate_design(design_dir: str) -> List[str]:
    """Validate a design directory against the DATA_SCHEMA contract.

    Returns a list of human-readable failure strings.
    An empty list means the design passes all checks.
    """
    failures: List[str] = []
    d = pathlib.Path(design_dir)

    if not d.exists():
        failures.append(f"Design directory does not exist: {design_dir}")
        return failures

    if not d.is_dir():
        failures.append(f"Path is not a directory: {design_dir}")
        return failures

    # --- Check every required file is present with required columns ---
    loaded: dict[str, pd.DataFrame] = {}
    for fname, required_cols in _REQUIRED_FILES.items():
        fpath = _find_csv(design_dir, fname)
        if fpath is None:
            failures.append(f"Missing required file: {fname}")
            continue
        try:
            df = _read_csv(fpath)
        except Exception as e:
            failures.append(f"Cannot read {fname}: {e}")
            continue
        loaded[fname] = df
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            failures.append(
                f"{fname}: missing required columns: {missing_cols}"
            )

    # If critical files are missing we can't do deeper checks
    if "design_stats.csv" not in loaded:
        return failures

    stats = loaded["design_stats.csv"]

    # --- design_stats must be exactly 1 row ---
    if len(stats) != 1:
        failures.append(
            f"design_stats.csv: expected exactly 1 row, got {len(stats)}"
        )

    # --- design_stats checks ---
    if "die_w_um" in stats.columns and "die_h_um" in stats.columns:
        if (stats["die_w_um"].iloc[0] <= 0):
            failures.append("design_stats.csv: die_w_um must be > 0")
        if (stats["die_h_um"].iloc[0] <= 0):
            failures.append("design_stats.csv: die_h_um must be > 0")
        die_w = stats["die_w_um"].iloc[0]
        die_h = stats["die_h_um"].iloc[0]
    else:
        return failures  # cannot do spatial checks without die dimensions

    # --- UNIT SANITY: instances inside the die ---
    if "instances.csv" in loaded:
        inst = loaded["instances.csv"]
        if "x_um" in inst.columns and "y_um" in inst.columns:
            max_x = inst["x_um"].max()
            max_y = inst["y_um"].max()
            if max_x > die_w * 1.01:
                if max_x > die_w * 100:
                    failures.append(
                        f"instances.csv: max(x_um)={max_x:.1f} is ~{max_x/die_w:.0f}x "
                        f"die_w_um={die_w:.1f}. Role A likely exported DEF database "
                        f"units (nanometers), not microns. Divide by 1000."
                    )
                else:
                    failures.append(
                        f"instances.csv: max(x_um)={max_x:.1f} exceeds "
                        f"die_w_um={die_w:.1f} by more than 1%"
                    )
            if max_y > die_h * 1.01:
                if max_y > die_h * 100:
                    failures.append(
                        f"instances.csv: max(y_um)={max_y:.1f} is ~{max_y/die_h:.0f}x "
                        f"die_h_um={die_h:.1f}. Role A likely exported DEF database "
                        f"units (nanometers), not microns. Divide by 1000."
                    )
                else:
                    failures.append(
                        f"instances.csv: max(y_um)={max_y:.1f} exceeds "
                        f"die_h_um={die_h:.1f} by more than 1%"
                    )

    # --- macros inside the die ---
    if "macros.csv" in loaded:
        mac = loaded["macros.csv"]
        if all(c in mac.columns for c in ["x0_um", "y0_um", "x1_um", "y1_um"]):
            if len(mac) > 0:
                if mac["x1_um"].max() > die_w * 1.01:
                    failures.append(
                        f"macros.csv: max(x1_um)={mac['x1_um'].max():.1f} exceeds die_w_um={die_w:.1f}"
                    )
                if mac["y1_um"].max() > die_h * 1.01:
                    failures.append(
                        f"macros.csv: max(y1_um)={mac['y1_um'].max():.1f} exceeds die_h_um={die_h:.1f}"
                    )

    # --- bumps non-empty ---
    if "bumps.csv" in loaded:
        if len(loaded["bumps.csv"]) == 0:
            failures.append("bumps.csv: file is empty (no bumps). A will be singular.")

    # --- strap_planned covers full grid, no duplicates, density in [0,1] ---
    cfg = load_config()
    ny_fine = cfg["grid"]["ny_fine"]
    nx_fine = cfg["grid"]["nx_fine"]

    if "strap_planned.csv" in loaded:
        sp = loaded["strap_planned.csv"]
        if "fy" in sp.columns and "fx" in sp.columns and "density" in sp.columns:
            expected = ny_fine * nx_fine
            if len(sp) != expected:
                failures.append(
                    f"strap_planned.csv: expected {expected} rows "
                    f"({ny_fine}x{nx_fine}), got {len(sp)}"
                )
            # check for duplicates
            if sp.duplicated(subset=["fy", "fx"]).any():
                failures.append("strap_planned.csv: duplicate (fy, fx) entries found")
            # density range
            if sp["density"].min() < 0 or sp["density"].max() > 1:
                failures.append(
                    f"strap_planned.csv: density must be in [0,1], "
                    f"got [{sp['density'].min():.4f}, {sp['density'].max():.4f}]"
                )

    # --- activity: mission_weight sums to 1.0, activity in [0,1] ---
    if "activity.csv" in loaded:
        act = loaded["activity.csv"]
        if "scenario" in act.columns and "mission_weight" in act.columns:
            # mission_weight is constant per scenario
            weight_per_scn = act.groupby("scenario")["mission_weight"].first()
            total = weight_per_scn.sum()
            if abs(total - 1.0) > 1e-6:
                failures.append(
                    f"activity.csv: mission_weight sums to {total:.6f}, "
                    f"expected 1.0 (±1e-6)"
                )
        if "activity" in act.columns:
            act_min = act["activity"].min()
            act_max = act["activity"].max()
            if act_min < 0 or act_max > 1:
                failures.append(
                    f"activity.csv: activity must be in [0,1], "
                    f"got [{act_min:.4f}, {act_max:.4f}]"
                )

    # --- every module in activity and instances exists in modules ---
    if "modules.csv" in loaded:
        mod_names = set(loaded["modules.csv"]["module"]) if "module" in loaded["modules.csv"].columns else set()
        if "activity.csv" in loaded and "module" in loaded["activity.csv"].columns:
            act_mods = set(loaded["activity.csv"]["module"])
            missing = act_mods - mod_names
            if missing:
                failures.append(
                    f"activity.csv: modules not found in modules.csv: {missing}"
                )
        if "instances.csv" in loaded and "module" in loaded["instances.csv"].columns:
            inst_mods = set(loaded["instances.csv"]["module"])
            missing = inst_mods - mod_names
            if missing:
                failures.append(
                    f"instances.csv: modules not found in modules.csv: {missing}"
                )

    # --- every inst_id in paths.inst_ids exists in instances ---
    if "paths.csv" in loaded and "instances.csv" in loaded:
        paths_df = loaded["paths.csv"]
        inst_df = loaded["instances.csv"]
        if "inst_ids" in paths_df.columns and "inst_id" in inst_df.columns:
            valid_ids = set(inst_df["inst_id"].astype(int))
            all_path_ids: set[int] = set()
            for ids_str in paths_df["inst_ids"].dropna():
                for tok in str(ids_str).split(";"):
                    tok = tok.strip()
                    if tok:
                        try:
                            all_path_ids.add(int(tok))
                        except ValueError:
                            failures.append(
                                f"paths.csv: non-integer inst_id token '{tok}'"
                            )
            missing = all_path_ids - valid_ids
            if missing:
                n = min(len(missing), 10)
                failures.append(
                    f"paths.csv: {len(missing)} inst_ids not in instances.csv, "
                    f"first {n}: {sorted(missing)[:n]}"
                )

    # --- irmap: full grid per scenario, drop_v in [0, 0.2] ---
    if "irmap.csv" in loaded:
        ir = loaded["irmap.csv"]
        if all(c in ir.columns for c in ["scenario", "fy", "fx", "drop_v"]):
            scenarios = ir["scenario"].unique()
            for scn in scenarios:
                sub = ir[ir["scenario"] == scn]
                expected = ny_fine * nx_fine
                if len(sub) != expected:
                    failures.append(
                        f"irmap.csv: scenario '{scn}' has {len(sub)} rows, "
                        f"expected {expected}"
                    )
                # drop_v range
                max_drop = sub["drop_v"].max()
                min_drop = sub["drop_v"].min()
                if max_drop > 1.0:
                    failures.append(
                        f"irmap.csv: scenario '{scn}' max(drop_v)={max_drop:.4f} > 1.0. "
                        f"Role A likely exported absolute voltage or millivolts, "
                        f"not drop in volts. Reject."
                    )
                if min_drop < 0:
                    failures.append(
                        f"irmap.csv: scenario '{scn}' has negative drop_v={min_drop:.6f}"
                    )
                if max_drop > 0.2 and max_drop <= 1.0:
                    failures.append(
                        f"irmap.csv: scenario '{scn}' max(drop_v)={max_drop:.4f} > 0.2V. "
                        f"Unusually high — verify units."
                    )
                # NON-UNIFORM: std > 1e-4
                std_drop = sub["drop_v"].std()
                if std_drop <= 1e-4:
                    failures.append(
                        f"irmap.csv: scenario '{scn}' is effectively uniform "
                        f"(std={std_drop:.2e}). A flat map means the analysis ran "
                        f"without doing anything useful. Refuse to ingest."
                    )

    return failures


# ---------------------------------------------------------------------------
# load_design — validated load from a design directory
# ---------------------------------------------------------------------------

def load_design(design_dir: str) -> Design:
    """Load and validate a design directory.  Raises ValueError on validation failure."""
    failures = validate_design(design_dir)
    if failures:
        raise ValueError(
            f"Validation failed for {design_dir}:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    d = pathlib.Path(design_dir)

    def _load(name: str) -> pd.DataFrame:
        path = _find_csv(design_dir, name)
        assert path is not None, f"Missing {name} — should have been caught by validate"
        return _read_csv(path)

    stats = _load("design_stats.csv")
    design_id = str(stats["design_id"].iloc[0])

    toggle = None
    toggle_path = _find_csv(design_dir, "toggle.csv")
    if toggle_path is not None:
        toggle = _read_csv(toggle_path)

    design = Design(
        design_id=design_id,
        design_dir=str(d),
        stats=stats,
        modules=_load("modules.csv"),
        macros=_load("macros.csv"),
        instances=_load("instances.csv"),
        bumps=_load("bumps.csv"),
        strap_planned=_load("strap_planned.csv"),
        activity=_load("activity.csv"),
        paths=_load("paths.csv"),
        irmap=_load("irmap.csv"),
        toggle=toggle,
    )

    # Every load goes through the S8 adapter, so real and synthetic designs
    # reach features.py in exactly the same state.  Imported here rather than
    # at module scope: design.py imports this module.
    from prism.design import from_csv
    return from_csv(design, load_config())


# ---------------------------------------------------------------------------
# write_design — used by design.py to emit the exact same CSVs
# ---------------------------------------------------------------------------

def write_design(design: Design, design_dir: str) -> None:
    """Write a Design object to the design directory as CSVs."""
    d = pathlib.Path(design_dir)
    d.mkdir(parents=True, exist_ok=True)

    design.stats.to_csv(d / "design_stats.csv", index=False)
    design.modules.to_csv(d / "modules.csv", index=False)
    design.macros.to_csv(d / "macros.csv", index=False)
    design.instances.to_csv(d / "instances.csv", index=False)
    design.bumps.to_csv(d / "bumps.csv", index=False)
    design.strap_planned.to_csv(d / "strap_planned.csv", index=False)
    design.activity.to_csv(d / "activity.csv", index=False)
    design.paths.to_csv(d / "paths.csv", index=False)
    design.irmap.to_csv(d / "irmap.csv", index=False)
    if design.toggle is not None:
        design.toggle.to_csv(d / "toggle.csv", index=False)


# ---------------------------------------------------------------------------
# load_corpus — load all designs listed in manifest.csv
# ---------------------------------------------------------------------------

def load_corpus(data_dir: str) -> List[Design]:
    """Load all designs from a corpus directory using manifest.csv."""
    manifest_path = pathlib.Path(data_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {data_dir}")

    manifest = pd.read_csv(manifest_path)
    if "path" not in manifest.columns:
        raise ValueError("manifest.csv must have a 'path' column")

    designs: List[Design] = []
    for _, row in manifest.iterrows():
        dpath = row["path"]
        # resolve relative paths against the data_dir
        if not os.path.isabs(dpath):
            dpath = str(pathlib.Path(data_dir) / dpath)
        designs.append(load_design(dpath))

    return designs
