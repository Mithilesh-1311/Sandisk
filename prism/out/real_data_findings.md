# What the real ORFS data can and cannot show

Session S8.  Eight OpenROAD-flow-scripts designs on nangate45, delivered by role
A as the standard nine-CSV contract plus `out/activity/toggle_rates.csv`.
Every number here is measured by a script in `scripts/` and stored in
`out/orfs_calibration.csv`, `out/orfs_transfer.csv`, `out/orfs_irmap_audit.csv`
and `out/orfs_gradient_check.csv`.  Nothing is retrained; nothing is re-fitted
on the synthetic corpus.

---

## 0. What is actually in the data

| design | cells | die µm | vdd V | bumps | irmap max mV |
|---|---|---|---|---|---|
| orfs_ibex | 31,761 | 239.28 | 1.1 | 16 | 8.60 |
| orfs_gcd | 986 | 35.76 | 1.1 | 1 | 11.84 |
| orfs_ssd_ctrl_cfg_small | 71,251 | 444.22 | 1.1 | 29 | 1.80 |
| orfs_ssd_ctrl_cfg_run_01_p12 | 71,250 | 444.22 | 1.1 | 73 | 0.64 |
| orfs_ssd_ctrl_cfg_run_02_p24 | 71,250 | 444.22 | 1.1 | 37 | 1.06 |
| orfs_ssd_ctrl_cfg_run_03_p24 | 73,493 | 444.21 | 1.1 | 37 | 1.17 |
| orfs_ssd_ctrl_cfg_run_04_p16 | 71,251 | 444.22 | 1.1 | 55 | 0.69 |
| orfs_ssd_ctrl_cfg_mid | 297,102 | 889.09 | 1.1 | 59 | 1.57 |

`vdd_v` is 1.1 V per design, not the 0.90 V in `config/default.yaml`, so the IR
budget on this corpus is **55.0 mV**, not 45.0 mV.  Every per-design resolution
(vdd, derived bump pitch, the empty-`macros.csv` sentinel, measured activity) is
recorded in `out/assumptions.log`.

**Two provenance facts that constrain every claim below.**  Both are properties
of `prism/orfs.py`, the delivered extractor, not of anything done in S8.

1. **There is one independent signoff map per design, not six.**  `orfs.py:226`
   builds all six scenario irmaps as `clip(asrun × scale[s], 0, 0.199)` from a
   single PDNSim solve.  The six maps are scalar multiples of one another
   (verified to 1e-5 in `out/orfs_irmap_audit.csv`).  Any statistic pooled over
   scenarios measures that rescale, not physics — pooled Spearman on this data
   reads ~0.4–0.7 purely because the six rescaled blocks separate in rank.
   Per-scenario and pooled fits are kept in the CSV for completeness and are not
   used for any claim.

2. **Four of the eight maps are partly synthetic.**  `orfs.py:234-252` adds a
   deterministic die-centre ramp (0.64 mV peak) to any map whose std falls below
   1.6e-4 V, so that `io_csv`'s non-uniformity check passes.  The four `run_0x`
   designs have reference-map std ≈ 1.2e-4 V — under the trigger — so the ramp
   was injected into their reference map too.  It explains 88% of the variance
   of `run_01_p12`, whose map max is exactly 0.64000 mV in all six scenarios.
   **Those four designs are excluded from every physics claim.**  Usable:
   `orfs_ibex`, `orfs_gcd`, `orfs_ssd_ctrl_cfg_small`, `orfs_ssd_ctrl_cfg_mid`.

Effective sample size for the physics claim is therefore **four designs, one
independent map each** — three once `orfs_gcd` is set aside as degenerate (986
cells and a single supply point on a 35.76 µm die under a 24×24 grid).

---

## 1. What the real data DOES validate

### 1a. The feature pipeline runs unmodified on genuine ORFS artefacts

`io_csv.validate_design` returns zero failures on all eight designs.
`features.design_features` returns **576 rows × 32 features with zero NaN and
zero inf** for all 8 designs × 6 scenarios (48 extractions), through the same
code path the synthetic corpus uses.  Reproduce with
`PYTHONPATH=. python scripts/orfs_gate.py`.

This required no change to the feature definitions.  It required one bug fix and
one adapter:

- `solver.py:379` shadowed `scipy.sparse as sp` with a DataFrame, crashing the
  bump-multiplicity branch.  That branch is dead on synthetic data (bump pitch
  150–300 µm ≫ tile) and live on every real design.
- `design.from_csv()` resolves per-design vdd, derives bump pitch from the bump
  coordinates when the header field is absent (never imputing a constant),
  records the empty-`macros.csv` sentinel, and prefers measured activity.

**Activity on the six `ssd_ctrl` designs is MEASURED, not assumed.**
`toggle_rates.csv` supplies per-module RTL toggle counts for five of the six
scenarios; activity is each module's rate normalised by its own peak.  An
independent re-derivation in `from_csv` reproduces role A's delivered
`activity.csv` to better than 1e-4 on all 25 (scenario, module) values, which
checks the RTL → CSV chain end to end.  `rand_read_4k` has no simulation and
keeps the assumed multipliers; `gcd` and `ibex` are single-module and stay
assumed.  All of this is logged per design.

### 1b. The mesh abstraction — the substantive result

Question: *does a one-parameter resistive mesh reproduce a signoff tool's
voltage map?*  `k_bump` is tied at 10 × `k_sheet`, so the mesh is a pure scale
and the fit is closed form, `k_sheet = Σu² / Σ(u·L)`.  One fit per design,
against the as-run map, with every module at its rated power — the condition
PDNSim actually solved.

**As delivered, it does not.**

| design | k_sheet | R² | MAE mV | bias mV | Spearman |
|---|---|---|---|---|---|
| orfs_ibex | 164.44 | −0.939 | 0.873 | −0.346 | −0.131 |
| orfs_gcd | 100.04 | +0.026 | 1.598 | −0.018 | +0.234 |
| orfs_ssd_ctrl_cfg_small | 489.86 | −0.981 | 0.252 | −0.088 | −0.226 |
| orfs_ssd_ctrl_cfg_mid | 1983.59 | −0.571 | 0.227 | −0.070 | −0.090 |

A negative R² is worse than predicting the mean.  This is reported as measured.

**The cause is identifiable, and it is the input, not the abstraction.**
`bumps.csv` on this corpus is not a bump array: every entry lies on a single
column at `x = die_w/2` (`orfs_gcd` has one point at the die centre).  The mesh
therefore predicts a ramp away from that centre column.  The PDNSim map does not
follow supply proximity at all — it follows local current density:

| design | Spearman vs mesh | vs distance-to-bump | vs current density |
|---|---|---|---|
| orfs_ibex | −0.131 | −0.321 | **+0.469** |
| orfs_ssd_ctrl_cfg_small | −0.226 | −0.416 | **+0.570** |
| orfs_ssd_ctrl_cfg_mid | −0.090 | −0.315 | **+0.536** |

The mesh tracks the distance term, with the wrong sign for this data.  This is
not a miscalibration of the held constant: sweeping the bump:sheet ratio over
0.1 → 1000 moves Spearman by less than 0.05.

**Correcting only the supply model recovers the fit.**  A dense M1-M4-M7 strap
grid on a 240–890 µm die reaches every tile, so the same `PDNSolver` (frozen
constructor, unmodified `solver.py`, still one free parameter, `k_bump` still
10×) was given a whole-die supply instead of the delivered bump column:

| design | R² bump-column → grid | Spearman → | MAE mV | bias mV |
|---|---|---|---|---|
| orfs_ibex | −0.939 → **+0.212** | −0.131 → **+0.475** | 0.574 | −0.059 |
| orfs_ssd_ctrl_cfg_small | −0.981 → **+0.466** | −0.226 → **+0.575** | 0.118 | −0.007 |
| orfs_ssd_ctrl_cfg_mid | −0.571 → **+0.462** | −0.090 → **+0.540** | 0.136 | +0.007 |
| orfs_gcd | +0.026 → −1.191 | +0.234 → +0.187 | 2.304 | −1.116 |

Against the tile-max target the same three give R² 0.22 / 0.44 / 0.40 and
Spearman 0.49 / 0.57 / 0.48.  `orfs_gcd` degrades and is degenerate as noted.

**Claim, stated at the strength the evidence supports:** on three real ORFS
designs with genuine PDNSim content, a resistive mesh with a single fitted
constant reproduces roughly 21–47% of the tile-to-tile variance of a signoff IR
map (Spearman 0.47–0.58, MAE 0.12–0.57 mV against map peaks of 0.86–5.96 mV) —
*provided the supply geometry it is given is correct*.  With the supply geometry
as delivered it reproduces none of it.  n = 3.  This is evidence that the
abstraction is sound and that its accuracy is bounded by the quality of the PDN
description it is handed; it is not a demonstration that the mesh is
signoff-accurate.

---

## 2. What the real data CANNOT validate

### Hotspot classification. There is nothing to classify.

**0 of 27,648 real tiles exceed the budget.**  The largest drop anywhere in the
real corpus is 11.84 mV against a 55.0 mV budget — a factor of 4.6 of margin on
the single worst tile of the single worst design.  Six of the eight designs peak
below 1.8 mV.

Therefore **violation F1, PR-AUC, precision, recall, F-beta crossover and the
confusion matrix are undefined on this data.**  They are not computed anywhere
in S8, and no zero is reported in their place.  A zero would state that the
classifier failed; the truth is that the question was never posed.  The
threshold was not lowered to manufacture positives — doing so would report a
classifier's skill at separating 3 mV from 2 mV and label it hotspot detection.

The reason is structural, not a sampling accident: OpenROAD's default PDN on
designs of this size is heavily over-provisioned.  A 240 µm ibex die with a
30 µm-pitch, 1.4 µm-wide M1-M4-M7 grid has orders of magnitude more supply metal
per unit current than a design that is actually IR-limited.  Add that PDNSim was
run at the flow's own static power estimate, and the analysis is simply not near
its budget.

**This is precisely why the synthetic corpus exists.**  Validating a hotspot
classifier requires a corpus that contains hotspots.  The synthetic corpus is
constructed so that violations occur at a rate the metric can resolve, with a
known ground truth at two fidelities, which is what makes F1, PR-AUC and the
conformal interval measurable at all.  The honest division of labour:

- **synthetic corpus** — validates the *task*: residual learning, hotspot
  ranking, interval calibration, the ablation ordering.
- **real ORFS corpus** — validates the *ingestion and the physics abstraction*:
  the contract, the feature pipeline, and how well a coarse mesh tracks a
  signoff solver.

Neither substitutes for the other.  This is a limitation to state, not to hide.

### Other things this data cannot support

- **Scenario-to-scenario generalisation.**  One PDNSim solve per design, rescaled
  six ways.  There is no independent per-scenario ground truth.
- **Cross-design variance of anything.**  Six of the eight designs are
  PDN-parameter sweeps of one netlist (`ssd_ctrl_cfg`), so they are not eight
  independent points; and four of those six carry the injected ramp.
- **Interval calibration.**  See below — coverage collapses, and with n = 3 clean
  designs there is no basis for recalibrating it.

---

## 3. Predicted vs PDNSim: the existing hybrid model, no retraining

`models/hybrid.joblib` loaded exactly as trained on the synthetic corpus and
applied to features from the eight ORFS designs.  Run
`PYTHONPATH=. python scripts/orfs_transfer.py`; full table in
`out/orfs_transfer.csv`.  Reference numbers, synthetic canonical holdout:
**MAE 1.868 mV, RMSE 2.845 mV, R² 0.966, Spearman 0.988, PICP 0.8205.**

### 3a. Pipeline verbatim (`as_trained`)

The physics prior uses the synthetic corpus's `k_sheet = 4.31`.  Signoff-clean
designs, reference scenario:

| design | MAE mV | RMSE mV | R² | Spearman | bias mV | PICP | label max mV | pred max mV |
|---|---|---|---|---|---|---|---|---|
| orfs_ibex | 54.33 | 57.36 | −1964.9 | −0.148 | +54.33 | 0.000 | 8.60 | 82.51 |
| orfs_gcd | 92.11 | 92.52 | −1400.6 | +0.204 | +92.11 | 0.000 | 11.84 | 106.42 |
| orfs_ssd_ctrl_cfg_small | 45.78 | 48.23 | −22794.1 | −0.190 | +45.78 | 0.000 | 1.80 | 68.70 |
| orfs_ssd_ctrl_cfg_mid | 128.41 | 138.27 | −177922.6 | −0.049 | +128.41 | 0.000 | 1.57 | 200.33 |

**The model does not transfer.**  `bias` equals `MAE` to four figures on every
design: the model over-predicts *every single tile*, by 25–128 mV against labels
that peak at 1.6–11.8 mV.  Interval coverage is 0.000–0.072 against a 0.82
target.  This is a total failure and is reported as one.

The cause is a scale shift in an input, not a defect in the learned residual:
`phys_base_v` is the model's dominant feature (permutation importance 4.49 mV,
17× the next group) and it is computed with a conductance constant calibrated to
a different technology and a different PDN.  The mesh is fed a `k_sheet` roughly
two orders of magnitude too small for these designs, so the physics prior — and
with it the prediction — lands two orders of magnitude too high.

### 3b. Physics prior put on the right scale (`recalibrated`)

Identical model weights; only `k_sheet` is replaced by each design's own fitted
value from §1b.  Signoff-clean designs, reference scenario:

| design | MAE mV | RMSE mV | R² | Spearman | bias mV | PICP |
|---|---|---|---|---|---|---|
| orfs_ibex | 1.198 | 1.458 | −0.271 | +0.326 | +0.701 | 0.300 |
| orfs_gcd | 2.418 | 2.975 | −0.449 | +0.269 | +1.584 | 0.151 |
| orfs_ssd_ctrl_cfg_small | 1.397 | 1.447 | −19.52 | +0.396 | +1.397 | 0.000 |
| orfs_ssd_ctrl_cfg_mid | 2.692 | 2.710 | −67.36 | +0.351 | +2.692 | 0.000 |

**This is an oracle bound, not a deployment number.**  `k_sheet` here was fitted
against the same PDNSim maps being scored.  It answers "is the learned residual
salvageable once the prior is on-scale?" and nothing more.  It must not be quoted
as transfer performance.

Read at that strength: absolute error falls from 25–128 mV to 1.2–2.7 mV, and
rank agreement becomes positive on all four designs (Spearman 0.27–0.40, versus
−0.19 to +0.20 before).  But R² stays negative on every design, the model still
over-predicts every tile (bias > 0 throughout, equal to MAE on two designs), and
interval coverage is 0.00–0.30 against 0.82.  MAE of 1.4 mV against a map whose
peak is 1.8 mV is not comparable to MAE 1.87 mV against a synthetic holdout whose
peak is an order of magnitude larger; the dynamic ranges differ by ~10×, which is
exactly why R² is the informative statistic here and why it is negative.

### 3c. What the failure mode actually says

Bias equals MAE to four figures on every design — every tile over-predicted,
none under.  That is a constant offset, not a broken model, and its magnitude is
exactly what `phys_base_v` produces when handed `k_sheet = 4.31` instead of the
100–1984 these designs need.

The evidence that the learned component *did* transfer is the ranking.  With the
prior on-scale, Spearman is positive on all four designs (+0.27 to +0.40): the
model still orders tiles correctly on a different technology, a different tool
and a different generator from the one it was trained on.  Negative R² together
with positive rho is the signature of surviving ordering and lost scale — R²
penalises the offset, rho ignores it.  Both negative would mean nothing was
learned.

This is the measured version of the project's stated position: **the mesh
abstraction and the residual formulation transfer; the trained weights are
node-specific and would need retraining from tapeout history.**

**Honest summary of transfer:** the model as trained does not transfer to this
data at all.  With the physics prior oracle-scaled it recovers the right order of
magnitude and a weak positive ranking, but not calibration and not R².  A model
that transfers imperfectly and quantifies the failure is worth more than one that
claims transfer it cannot show — and the failure here is diagnostic, pointing at
a single input (the conductance constant, and behind it the supply geometry)
rather than at the residual formulation.

---

## 4. Limitations, collected

1. Four of eight designs carry a synthetic ramp injected by the extractor and are
   excluded from all physics claims.
2. One independent signoff map per design; the six scenarios are scalar rescales.
3. `bumps.csv` is a single column of source points, not a bump array — the
   delivered supply geometry does not describe the PDN that was solved.
4. Zero violations in the real corpus; all classification metrics are undefined
   and are not reported.
5. n = 3 usable designs for the mesh claim, six of eight being sweeps of one
   netlist.
6. `macros.csv` is empty in all eight, so `top_macro_frac` and `top_dmacro` carry
   no signal on real data; the two features are constant sentinels there.
7. The `recalibrated` transfer pass fits `k_sheet` on the evaluation labels and
   is an oracle bound.
8. PDNSim was run at the flow's static power estimate, not at a per-scenario
   power; scenario structure in the real labels is imposed arithmetically.

## Reproduce

```
PYTHONPATH=. python scripts/orfs_gate.py             # ingest gate, 8 designs
PYTHONPATH=. python scripts/orfs_irmap_audit.py      # which maps are signoff
PYTHONPATH=. python scripts/orfs_gradient_check.py   # how much is injected ramp
PYTHONPATH=. python scripts/orfs_calibrate.py        # mesh calibration, Task 2
PYTHONPATH=. python scripts/orfs_transfer.py         # model transfer, Task 3
```
