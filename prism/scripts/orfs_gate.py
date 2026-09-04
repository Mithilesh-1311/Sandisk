"""Gate 1 for S8: the feature pipeline must run unmodified on real ORFS data.

For each design under data/orfs/, load it through io_csv.load_design (which now
runs the from_csv adapter) and extract features for one scenario.  576 rows,
32 features, zero NaN/inf is the pass condition.
"""
import sys
import numpy as np
from prism.io_csv import load_design, load_config, validate_design
from prism import features
from prism.design import _SCENARIOS
import pandas as pd

cfg = load_config()
cols = features.feature_columns()
manifest = pd.read_csv("data/orfs/manifest.csv")

ok = True
for did in manifest["design_id"]:
    ddir = f"data/orfs/{did}"
    fails = validate_design(ddir)
    d = load_design(ddir)
    for scn in _SCENARIOS:
        f = features.design_features(d, scn, cfg)
        X = f[cols].to_numpy(float)
        bad = ~np.isfinite(X)
        if len(f) != 576 or X.shape[1] != 32 or bad.any():
            ok = False
            names = [cols[i] for i in np.unique(np.where(bad)[1])] if bad.any() else []
            print(f"FAIL {did}/{scn}: rows={len(f)} cols={X.shape[1]} nonfinite={names}")
    print(f"OK   {did:32s} 6 scenarios x 576 rows x 32 features, "
          f"validate_design failures={len(fails)}")

print("GATE 1", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
