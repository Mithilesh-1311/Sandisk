"""Audit what is actually signoff data inside each real irmap.

prism/orfs.py builds the six scenario maps from ONE PDNSim as-run solve:

    irmap[s] = clip(asrun * scale[s], 0, 0.199)          (orfs.py:226-233)

and then, where a scaled map falls below std 1.6e-4, adds a faint deterministic
die-position gradient so io_csv's non-uniformity check passes
(orfs.py:234-252).  That gradient is synthetic.  Anything fitted against a
contaminated map is partly fitting the injection, so this script finds exactly
which (design, scenario) maps are clean scalar multiples of the as-run solve
and which are not.

Test: regress each scenario map on the reference map through the origin.  A
clean map has residual ~ 0 to machine precision.  A contaminated one leaves a
residual that correlates with the known gradient shape.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from prism.design import _SCENARIOS
from prism.io_csv import load_config, load_design

DATA_DIR = pathlib.Path("data") / "orfs"

# The scale factors orfs.py applies, reproduced from its _ACT table.
_ACT = {
    "idle":         [0.05, 0.02, 0.01, 0.02, 0.05, 0.10],
    "seq_read":     [0.85, 0.80, 0.55, 0.70, 0.60, 0.35],
    "seq_write":    [0.90, 0.90, 0.95, 0.85, 0.75, 0.40],
    "rand_read_4k": [0.60, 0.55, 0.45, 0.50, 0.90, 0.85],
    "gc_compact":   [0.05, 0.85, 0.90, 0.90, 0.70, 0.60],
    "ecc_recover":  [0.10, 0.20, 1.00, 0.15, 0.30, 0.25],
}
_REF = float(np.mean(_ACT["seq_read"]))
SCALE = {s: max(0.06, float(np.mean(v)) / _REF) for s, v in _ACT.items()}


def main() -> None:
    cfg = load_config()
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]

    yy, xx = np.mgrid[0:ny_f, 0:nx_f]
    grad = np.abs(yy - ny_f / 2) / ny_f + np.abs(xx - nx_f / 2) / nx_f
    grad = (grad / grad.max()).ravel()

    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    print(f"scale factors: " + ", ".join(f"{s}={SCALE[s]:.4f}" for s in _SCENARIOS))
    print()
    rows = []
    for did in manifest["design_id"]:
        d = load_design(str(DATA_DIR / did))
        maps = {}
        for scn, sub in d.irmap.groupby("scenario", sort=False):
            m = np.zeros((ny_f, nx_f))
            m[sub["fy"].to_numpy(int), sub["fx"].to_numpy(int)] = sub["drop_v"].to_numpy(float)
            maps[str(scn)] = m.ravel()

        # Reference: the scenario with the largest scale factor, hence the one
        # least likely to have been lifted by the gradient.
        ref_scn = max(_SCENARIOS, key=lambda s: SCALE[s])
        ref = maps[ref_scn] / SCALE[ref_scn]

        for scn in _SCENARIOS:
            m = maps[scn]
            a = float(np.dot(ref, m) / np.dot(ref, ref))     # best scalar multiple
            resid = m - a * ref
            rel = float(np.sqrt(np.mean(resid ** 2)) / (m.std() + 1e-30))
            gcorr = (float(np.corrcoef(resid, grad)[0, 1])
                     if resid.std() > 1e-18 else float("nan"))
            rows.append(dict(design=did, scenario=scn, std_mv=m.std() * 1e3,
                             max_mv=m.max() * 1e3, fitted_scale=a,
                             stated_scale=SCALE[scn], resid_frac=rel,
                             grad_corr=gcorr,
                             clean=rel < 1e-6))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:10.5f}"))
    print()
    print("clean (exact scalar multiple of the as-run solve) per design:")
    print(df.groupby("design")["clean"].sum().to_string())
    df.to_csv("out/orfs_irmap_audit.csv", index=False)
    print("\nWrote out/orfs_irmap_audit.csv")


if __name__ == "__main__":
    main()
