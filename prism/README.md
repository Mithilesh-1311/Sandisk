# PRISM — floorplan-stage IR-drop hotspot prediction

PRISM predicts where a chip's power grid will violate its IR-drop budget while
the floorplan is still editable, hours before a signoff tool could tell you.
It does this by correcting a fast coarse physics solve with a learned residual,
rather than by asking a neural network to rediscover Ohm's law.

## The residual formulation

```
U_hat  =  U_coarse_solve  +  g(early features)
```

The coarse solve is a resistive mesh, factorised once and back-substituted per
scenario (<5 ms at 96×96). It already knows Ohm's law, current spreading and
bump topology **exactly** — there is nothing to learn there. What it cannot
know at floorplan time is the as-built strap map and sub-tile current
concentration, so `g` learns only that gap, from 32 features available before
detailed routing. Learning a small correction to a correct physical prior needs
far less data than learning the whole map, and it fails in an interpretable way
when it fails.

## Headline result

**Canonical holdout — 3 designs never seen in training or calibration, 10,368 tiles**
(`out/predictions.csv`, `partition == holdout`):

| MAE | interval coverage | R² | violation recall |
|---|---|---|---|
| **1.868 mV** | **0.8205** (target 0.80) | **0.966** | **0.892** |

The finding the study exists to make: **physics has the best ranking of any
variant in the study (PR-AUC 0.9584) and still misses 78% of violations,
because it is 9.22 mV biased low.** Calibration, not ranking, is the
differentiator. A perfectly ordered map with a systematic offset finds no
hotspots; the residual model does not improve the ordering much, it fixes the
offset — and that is what turns recall from 0.216 into 0.892.

(Across 25 train/calibration partitions the same model gives MAE 1.957 ± 0.183 mV,
R² 0.961 ± 0.007, recall 0.894 ± 0.055, coverage 0.820 ± 0.074. Those are a
different set from the canonical numbers above; the two are never mixed.)

## Real data

Eight OpenROAD-flow-scripts designs (ibex, gcd, six `ssd_ctrl` PDN sweeps) were
ingested through the unmodified feature pipeline: 576 rows × 32 features, zero
NaN, zero inf, on all 8 × 6 scenario extractions. **Zero tiles violate** — the
whole real corpus peaks at 11.84 mV against a 55 mV budget — so violation F1,
PR-AUC, precision and recall are *undefined* on this data and are not computed
or reported. Validating a hotspot classifier needs a corpus containing
hotspots; that is what the synthetic corpus is for.

What the real data did do is act as a diagnostic. The mesh fitted against the
signoff PDNSim maps returned **negative R²**, which traced to a delivered
`bumps.csv` containing a single column of supply points at `x = die_w/2` rather
than a bump array. Correcting only the supply model — same solver, same single
free parameter — flips Spearman from negative to **+0.475 / +0.575 / +0.540** on
the three designs with genuine signoff content. The mesh abstraction and the
residual formulation transfer; the trained weights are node-specific.

Full analysis, with limitations: **[out/real_data_findings.md](out/real_data_findings.md)**.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py all                              # gen → features → train → eval → figures
pytest tests/ -q                                   # 77 acceptance gates
streamlit run app.py                               # 4-page dashboard, reads CSVs only
```

`run_all.py all` is reproducible: models, predictions, features and run tables
come out byte-identical across runs. The only exception is the bootstrap `ci_lo`
/ `ci_hi` columns of `out/validation.csv`, which are drawn unseeded and move in
the 4th decimal; every mean and std is stable.

## Repo map

| path | contents |
|---|---|
| `prism/` | the package: `solver` (mesh), `design` (generator + adapter), `features` (the 32), `model` (residual + conformal), `evaluate`, `viz`, `audit` (leakage trap), `io_csv` (the only data entry point), `orfs` (ORFS extractor) |
| `config/` | `default.yaml` — grid, electrical budget, scenarios, seeds |
| `data/synthetic/` | 14 generated designs, two fidelities, with hotspots |
| `data/orfs/` | 8 real OpenROAD designs, same nine-CSV contract |
| `out/` | features, predictions, validation tables, findings, `assumptions.log` |
| `models/` | trained variants + provenance manifest |
| `figures/` | the 11 result figures |
| `scripts/` | S8 real-data analysis: ingest gate, irmap audit, mesh calibration, model transfer |
| `flow/` | `orfs_extract.tcl` — the OpenROAD-side extractor |
| `tests/` | acceptance gates for every session, as pytest |
| `app.py` | Streamlit dashboard; reads `out/*.csv`, never loads a model |

## Findings documents

- **[out/headline_findings.md](out/headline_findings.md)** — the synthetic-corpus result and ablation.
- **[out/real_data_findings.md](out/real_data_findings.md)** — what the real ORFS data can and cannot show.
