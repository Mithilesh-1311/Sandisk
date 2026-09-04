"""S8 Task 3 -- run the EXISTING hybrid model on real ORFS data.

No retraining, no refitting, no threshold changes.  models/hybrid.joblib is
loaded exactly as trained on the synthetic corpus and applied to features
extracted from the eight ORFS designs.

Two passes, because the physics prior is a model input:

  as_trained     phys_base_v uses out/calibration.json's k_sheet = 4.31, which
                 was calibrated on the synthetic corpus.  This is the pipeline
                 run verbatim on new data.
  recalibrated   phys_base_v uses each design's own k_sheet from
                 out/orfs_calibration.csv (Task 2).  The model weights are
                 untouched; only the physics feature is put on the right scale.
                 This separates "the learned residual does not transfer" from
                 "the physics prior was on the wrong scale".

Violation F1, PR-AUC, precision and recall are NOT computed.  The real corpus
peaks at 11.84 mV against a 55 mV budget, so zero tiles violate and every
classification metric is undefined.  A zero would be a fabrication.

Writes out/orfs_transfer.csv.
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prism.audit import leakage_trap, unstash_irmap
from prism.design import _SCENARIOS
from prism.features import _static_tile_features, add_labels, design_features
from prism.io_csv import load_config, load_design
from prism.model import apply_conformal, predict_variant

DATA_DIR = pathlib.Path("data") / "orfs"
OUT_CSV = pathlib.Path("out") / "orfs_transfer.csv"
REF_SCENARIO = "seq_write"          # the one map per design that is not rescaled


def _feature_table(cfg: dict, k_by_design: dict | None) -> pd.DataFrame:
    """Features + labels for the ORFS corpus.  k_by_design=None -> synthetic k."""
    from prism.design import load_calibration
    k_syn, kb_syn = load_calibration()
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    parts = []
    for did in manifest["design_id"]:
        k_sheet = k_syn if k_by_design is None else float(k_by_design[did])
        k_bump = 10.0 * k_sheet
        with leakage_trap():
            d = load_design(str(DATA_DIR / did))
            static = _static_tile_features(d, cfg, k_sheet, k_bump)
            fdf = pd.concat(
                [design_features(d, s, cfg, static=static, k_sheet=k_sheet,
                                 k_bump=k_bump) for s in _SCENARIOS],
                ignore_index=True)
        unstash_irmap(d)
        parts.append(add_labels(fdf, d.irmap, cfg))
    return pd.concat(parts, ignore_index=True)


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    resid = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae_mv": float(np.mean(np.abs(resid))) * 1e3,
        "rmse_mv": float(np.sqrt(np.mean(resid ** 2))) * 1e3,
        "bias_mv": float(np.mean(resid)) * 1e3,
        "r2": float("nan") if ss_tot <= 0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot,
        "spearman": (float("nan") if y.std() < 1e-15 or pred.std() < 1e-15
                     else float(spearmanr(pred, y).statistic)),
        "n": int(y.size),
    }


def main() -> None:
    cfg = load_config()
    models = joblib.load("models/hybrid.joblib")

    calib = pd.read_csv("out/orfs_calibration.csv")
    asrun = calib[(calib.fit_scope == "asrun") & (calib.target == "tile_max")]
    k_by_design = dict(zip(asrun["design"], asrun["k_sheet"]))
    clean = dict(zip(asrun["design"], asrun["signoff_clean"]))

    rows = []
    for pass_name, kmap in (("as_trained", None), ("recalibrated", k_by_design)):
        t = _feature_table(cfg, kmap)
        pred, q10, q90 = predict_variant(models, t)
        lo, hi = apply_conformal(q10, q90, models["Q_add"], models["Q_ratio"])
        t = t.assign(pred=pred, lo=lo, hi=hi)

        for did, sub in t.groupby("design"):
            cov = float(np.mean((sub["label_v"] >= sub["lo"]) & (sub["label_v"] <= sub["hi"])))
            ref = sub[sub["scenario"] == REF_SCENARIO]
            rows.append(dict(
                pass_name=pass_name, design=did, scope="all_scenarios",
                signoff_clean=bool(clean.get(did, False)), picp=cov,
                label_max_mv=float(sub["label_v"].max()) * 1e3,
                pred_max_mv=float(sub["pred"].max()) * 1e3,
                **_metrics(sub["pred"].to_numpy(), sub["label_v"].to_numpy())))
            rows.append(dict(
                pass_name=pass_name, design=did, scope=f"{REF_SCENARIO}_only",
                signoff_clean=bool(clean.get(did, False)),
                picp=float(np.mean((ref["label_v"] >= ref["lo"]) & (ref["label_v"] <= ref["hi"]))),
                label_max_mv=float(ref["label_v"].max()) * 1e3,
                pred_max_mv=float(ref["pred"].max()) * 1e3,
                **_metrics(ref["pred"].to_numpy(), ref["label_v"].to_numpy())))

        # Violations, counted only to prove there are none.
        budget = 0.05 * 1.1
        n_viol = int((t["label_v"] > budget).sum())
        print(f"[{pass_name}] tiles over the 55.0 mV budget: {n_viol} of {len(t)}"
              f"   (corpus max label {t['label_v'].max()*1e3:.2f} mV)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    f = lambda v: f"{v:9.4f}"
    for p in ["as_trained", "recalibrated"]:
        for sc in ["all_scenarios", f"{REF_SCENARIO}_only"]:
            print(f"\n=== hybrid model, {p}, {sc} ===")
            s = df[(df.pass_name == p) & (df.scope == sc)]
            print(s[["design", "signoff_clean", "mae_mv", "rmse_mv", "r2",
                     "spearman", "bias_mv", "picp", "label_max_mv",
                     "pred_max_mv"]].to_string(index=False, float_format=f))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
