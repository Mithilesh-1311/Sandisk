"""How much of each design's as-run map is PDNSim, and how much is injected?

orfs.py adds `amp * grad` (amp = 4 * 1.6e-4 V, grad = a normalised distance-from-
die-centre ramp) to any scenario map whose std falls below 1.6e-4 V.  The four
run_0x designs have seq_write std ~1.2e-4 V, i.e. BELOW that trigger, so the
injection fired on their reference map too.  This subtracts the known gradient
and reports what is left.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from prism.io_csv import load_config, load_design

AMP = 4.0 * 1.6e-4
SEQ_WRITE_SCALE = 1.2337662337662337
DATA_DIR = pathlib.Path("data") / "orfs"


def main() -> None:
    cfg = load_config()
    ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
    yy, xx = np.mgrid[0:ny_f, 0:nx_f]
    grad = np.abs(yy - ny_f / 2) / ny_f + np.abs(xx - nx_f / 2) / nx_f
    grad = grad / grad.max()

    rows = []
    for did in pd.read_csv(DATA_DIR / "manifest.csv")["design_id"]:
        d = load_design(str(DATA_DIR / did))
        sub = d.irmap[d.irmap["scenario"] == "seq_write"]
        m = np.zeros((ny_f, nx_f))
        m[sub["fy"].to_numpy(int), sub["fx"].to_numpy(int)] = sub["drop_v"].to_numpy(float)
        resid = m - AMP * grad
        # best scalar multiple of grad inside m -- how much of m the ramp explains
        a = float(np.sum(m * grad) / np.sum(grad * grad))
        r2_grad = 1.0 - float(np.sum((m - a * grad) ** 2)) / float(np.sum((m - m.mean()) ** 2))
        rows.append(dict(
            design=did,
            seqwrite_std_mv=m.std() * 1e3,
            triggered_injection=bool(m.std() < 1.6e-4),
            seqwrite_max_mv=m.max() * 1e3,
            after_removing_ramp_max_mv=resid.max() * 1e3,
            after_removing_ramp_std_mv=resid.std() * 1e3,
            var_explained_by_ramp=r2_grad,
            asrun_max_mv=(m / SEQ_WRITE_SCALE).max() * 1e3,
        ))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:10.5f}"))
    df.to_csv("out/orfs_gradient_check.csv", index=False)
    print("\nWrote out/orfs_gradient_check.csv")


if __name__ == "__main__":
    main()
