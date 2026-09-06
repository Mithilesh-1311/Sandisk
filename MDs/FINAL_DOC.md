# PROJECT PRISM
## Power-Integrity Risk Identification, Slack-Impact Ranking and Mitigation

**Physics-Guided IR Prediction, Slack-Impact Ranking and Measured Mitigation**

**Team BAND4BAND · Vellore Institute of Technology**
SanDisk One-Day University Hackathon · VLSI Design · **Problem Statement 1** (IR-drop
prediction & mitigation), with **Problem Statement 2** closed as the telemetry extension.

| Member | Reg. no | Role |
|---|---|---|
| Mithilesh Angal | 24BEC0767 | B — physics engine + hybrid predictor |
| Anusha Ghose | 24BEC0649 | D — dashboard, report, slides |
| Barath Srinivasan | 24BEC0207 | A — parameterised RTL + physical-design flow |
| Sanika Jamkhedkar | 24BEC0657 | C — risk engine, DSE, telemetry, mitigation |

**Links**
- Repository — https://github.com/Mithilesh-1311/Sandisk.git
- Demo / supporting material — https://drive.google.com/file/d/14JkfCSoOgsSeJk7WCMn1qoeY85Sl4Mpr/view?usp=drive_link

---

## 1. Problem Statement

Discovering voltage drops (IR-drop) late in the physical design cycle forces costly chip
redesigns, delays product launch and degrades efficiency. Traditional tools take hours to
calculate exact physics and treat every voltage drop the same, missing high-risk hotspots
that only trigger during heavy workload spikes.

> ### The PRISM Insight
> **IR-drop is not just a voltage problem, it is a timing problem measured in volts.**
> A 15 mV drop on a tight timing path breaks the chip, while a 40 mV drop on a
> non-critical path does not matter at all.

Everything in PRISM follows from that sentence.

---

## 2. Mapping Voltage Drop to Timing Paths

PRISM transforms standard floorplan, activity, and timing files into a simplified physical
conductance grid mesh (**L·U = I**). Instead of analysing voltage in isolation, it models how
power-grid droop slows down individual transistors along critical logic timing paths.

**PRISM pipeline mapping:**

```
Power Bumps ──▶ Strap Network ──▶ Active Logic ──▶ Delay Sensitivity ──▶ Effective Slack
```

---

## 3. Our Pillars of Novelty

*Converting simple millivolt estimates into actionable, physics-backed optimization.*

### Pillar 1 — Physics-Guided Hybrid Learning
- **Physics-First Baseline** — solves Kirchhoff's equations (`L·U = I`) for a fast, exact baseline.
- **Residual ML Correction** — trains a lightweight model on unresolved local gaps to prevent
  "black box" failures.
- **mV to Picoseconds Shift** — evaluates drop against transistor delays via the Alpha-Power Law.
- **Impact-Driven Ranking** — targets high-risk critical paths over uncritical high-drop regions.

### Pillar 2 — Measured Fixes & DSE
- **Re-solved Physics** — re-runs the linear solver with fixes applied to measure true impact.
- **Knapsack Optimization** — algorithmically selects optimal mitigations within area and
  routing limits.
- **24,000× Faster DSE** — uses a 50 ms surrogate model to evaluate 500 k+ chip configurations.
- **Pareto Trade-offs** — sweeps budget curves to maximise drop reduction per unit cost.

### Pillar 3 — Economic Risk & Sensor Map
- **Scenario-Weighted Risk** — ranks risk by weighting failure severity with workload frequency
  (e.g. GC compaction).
- **10×–50× Multi-Scenario Speedup** — factorises the matrix once to instantly solve multiple
  operating modes.
- **Sensor Prior Loop (PS2 Integration)** — uses IR risk heatmaps as priors for placing
  TDET/PDET sensors.
- **Submodular Max-Coverage** — guarantees optimal sensor placement via a greedy algorithm
  (1 − 1/e bound).

---

## 4. Hotspot Prediction Engine — Physics-Guided Hybrid Predictor

Instead of forcing AI to rediscover Ohm's Law from scratch, PRISM solves a fast linear physics
baseline (**R² = 0.80**) and uses machine learning solely to predict the remaining local
residual error:

```
U' = U_coarse + g(features)
```

**Key predictor inputs & components**

| | |
|---|---|
| **Physics** | Base solver | 
| **Conductance** | Strap matrix |
| **Current** | Switching activity |
| **Geometry** | Layout features |
| **ML** | Gradient trees |
| **Uncertainty** | Confidence range |
| **Validation** | Unseen floorplans |
| **Target** | Hotspot hit-rate |

**Why it wins:** pure ML models fail on unseen floorplans because they memorise patterns
without understanding physics. PRISM's physics baseline guarantees generalisation across any
chip layout.

### The two fidelities

| Attribute | Early-stage coarse physics (input) | Fine-grid ground truth (label) |
|---|---|---|
| Grid resolution | 24 × 24 tiles (576 nodes) | 96 × 96 mesh (9,216 nodes) |
| PDN topology | Planned global strap density (M4–M7) | As-built routed wires, local necking & vias |
| Current distribution | Tile-averaged macroscopic current | Instance-level switching currents |
| Macro handling | Bounding-box clearance | Edge via starvation & shadow effects |

The residual the model learns is not noise — it is **sub-tile current concentration**,
**macro-boundary via starvation** (15–30 % conductance loss), and **congestion-induced strap
necking** (10–40 % local resistance increase).

### Three mathematical invariants the solver exploits
1. **Exact linearity in current** — any current-reducing fix has an exactly computable benefit.
2. **Homogeneous conductance scaling** — `A → sA` implies `U → U/s`; calibration is closed-form,
   no iterative search.
3. **Factorise once, back-substitute many** — `A` depends only on floorplan geometry and bump
   coordinates, so every extra scenario is an O(N) back-substitution. This is the 10×–50×
   multi-scenario speedup.

---

## 5. Measured Mitigation Engine — Targeted Fix Catalog

Rather than guessing fix benefits using rules of thumb, PRISM mathematically **re-solves the
physical grid mesh** `(L + D_bump)·U = I` for every candidate fix. A Knapsack algorithm then
optimises fix combinations under strict budget constraints to build a benefit-versus-cost
Pareto curve.

| Fix | Effect |
|---|---|
| **Clock-Skew Staggering** | Cheap, smooths current spikes |
| **Decap Insertion** | Local charge reservoir |
| **Strap Widening** | Lowers grid wire resistance |
| **Cell De-densification** | Spreads local peak current |
| **Extra Power Bump** | Adds package supply pins |
| **Vt Swap / Downsize** | Reduces non-critical power |

Engine components: **Physical Grid Re-Solver (L·U = I)** → **Knapsack Budget Optimizer** →
**Benefit vs. Cost Pareto Curve**.

---

## 6. Pipeline Architecture & Data Leakage Audit

How early physical-design artefacts flow from RTL synthesis through hybrid prediction, risk
ranking, and mitigation.

**Core toolchain & predictor**
- **Parameterised RTL & Synthesis** — inputs SystemVerilog `ssd_ctrl_top` through Yosys to
  generate standard logic netlists.
- **Physical Design (ORFS)** — runs floorplanning, placement, and clock-tree synthesis to yield
  early DEF artefacts.
- **Feature Extraction (B1)** — extracts ~30 early-stage spatial and physical features
  (power density, topology, switching). *Implementation: exactly 32 features.*
- **Hybrid Predictor (B2)** — merges the `L·U = I` coarse physics estimate with gradient-boosted
  residual ML corrections.

**Risk engine & outputs**
- **Slack Risk Engine (B3)** — maps voltage droop mV directly to critical path delays ps using
  Alpha-Power Law equations.
- **Mitigation Catalog (B4)** — evaluates fixes (strap widening, decap insertion, clock-skew
  staggering) by re-solving the conductance grid.
- **Design Space Exploration (B5)** — feeds surrogate predictions into NSGA-II to optimise global
  chip configurations.
- **Telemetry Extension (B6)** — maps high-risk IR-drop hotspots as priors for physical sensor
  placement.

> ### Strict Data Leakage Prevention
> Ground-truth signoff maps are exclusively stored as **labels** for validation and **never**
> exposed as inputs to feature extraction. Verified live via `prism/audit.py`.

Enforced rules: **GroupKFold by design** (random tile splitting on spatially correlated grids is
prohibited); **label isolation** (any feature pipeline touching the fine-grid solve raises
`LeakageError`); **design-fingerprint ban** (`hash`, `config`, `design_id`, `ts_utc`, `lint_rc`
permanently excluded from model inputs).

---

## 7. Design Space Exploration & ROI

Moving from local point-fixes to intelligent, multi-objective chip floorplanning. Rather than
patching sub-optimal designs late in the flow, PRISM leverages surrogate models to find ideal
configurations early.

| Component | What it does |
|---|---|
| **Parameter Knobs** | Adjusts NAND channels, FIFO depths, strap pitches, and aspect ratios dynamically |
| **Sobol Sensitivity** | Ranks critical design knobs (strap / bump pitch) driving voltage drop |
| **NSGA-II Optimization** | Balances 5 targets: drop, timing slack, total area, power, and routing space |
| **Re-Solved Mitigations** | Evaluates decaps, strap widening, and cell de-densification directly on the physics grid |
| **Knapsack Allocation** | Selects optimal fix combinations given strict area and routing resource budgets |
| **Clock Skew Staggering** | Low-cost fix that offsets switching times to smooth total current draw spikes |

> **50 ms fast model allows sampling 500 k+ configurations, replacing 20-minute physical design
> runs to uncover true Pareto trade-offs.**

Tiering: **Tier 1** Sobol screening → **Tier 2** NSGA-II Pareto front → **Tier 3** Gaussian-Process
Bayesian optimisation proposing configurations that deserve a real PnR run.

---

## 8. Validation, Demo & Sensor Closure

Proving accuracy, ranking impact, and closing the loop with PS2 telemetry.

**Important evaluation metrics**

- Top-5 % Hotspot Hit-Rate
- Millivolt vs. Slack Ranking
- Ground-Truth Prediction Map
- Measured Mitigation ROI
- Pareto Efficiency Frontiers
- Scenario Risk Weighting
- PS2 TDET/PDET Placement
- Precomputed Demo Safety

*Precomputed demo safety:* the dashboard reads results from disk and never trains or solves
live, so the demo cannot fail on stage.

---

## 9. Results

### 9.1 Heatmap results — open-source validation designs

Both run through the identical OpenROAD flow on Nangate45, PDNSim static IR analysis.

| Design | Description | Worst sag |
|---|---|---|
| `gcd` | A small, off-the-shelf sample design run through the tool to confirm it works before trusting it on our own chip. Higher because this chip's power grid was never tuned. | **10.29 mV** |
| `ibex` | RISC-V core reference design (~32 k cells). A bigger sample chip run through the same tool as a second check. | **13.04 mV** |

Together these show the tool gives sensible, consistent answers on chips we did not design
ourselves.

### 9.2 Our own chip — real place-and-route sweep

A parameterised SystemVerilog SSD/NAND controller (`ssd_ctrl_top`, 12 modules, 3 asynchronous
clock domains) taken through the full OpenROAD flow — synthesis → floorplan → place → CTS →
route → PDNSim static IR — on Nangate45, at 8 design points.

| Config | PDN strap pitch | ECC lanes | Instances | Worst IR drop | Power |
|---|---|---|---|---|---|
| `cfg_small` (baseline) | 56/30 µm (platform) | 1 | 71,251 | **1.80 mV** | 17.1 mW |
| `cfg_mid` (4.2× scale) | 56/30 µm | 2 | 297,102 | 2.03 mV | 55.9 mW |
| `cfg_run_02` | 24 µm | 1 | 71,250 | 0.71 mV | 17.1 mW |
| `cfg_run_03` | 24 µm | 4 | 73,493 | 0.69 mV | 17.1 mW |
| `cfg_run_04` | 16 µm | 1 | 71,251 | 0.38 mV | 17.1 mW |
| `cfg_run_01` | 12 µm | 1 | 71,250 | **0.31 mV** | 17.1 mW |

**What the sweep proves**

- **PDN strap pitch is the dominant IR-drop lever.** 56 → 12 µm cuts worst-case IR drop
  **1.80 → 0.31 mV — 6× — at flat power** (17.1 mW at every strap-pitch point).
- **ECC lanes 1 → 4 change static IR by < 3 %** (0.71 vs 0.69 mV). Extra logic is not the driver;
  the wiring is.
- **Architectural scale barely moves it.** `cfg_small` → `cfg_mid` is 4.2× the instance count for
  a 1.13× change in worst IR drop — size alone is not what causes voltage problems.
- These match the team's pre-work prediction that strap and bump pitch dominate, with ECC lanes
  mattering only through local current density.
- The 3-config synthesis sweep shows a **26× cell-count spread** (`cfg_small` 26.8 k →
  `cfg_large` 695 k cells), monotone in every knob — the parameterisation produces genuinely
  different designs, not twelve copies of one.

### 9.3 Hybrid predictor ablation (held-out designs)

14 synthetic designs partitioned 8 train / 3 calibration / 3 holdout; 25 independent evaluations
via `GroupShuffleSplit` across 5 seeds; 95 % bootstrap CIs over 1,000 resamples.

| Metric | physics_only | physics_affine | learned_only | **hybrid (PRISM)** |
|---|:---:|:---:|:---:|:---:|
| Violation F1 (≥ 45 mV) | 0.355 ± 0.021 | 0.880 ± 0.009 | 0.803 ± 0.015 | **0.847 ± 0.011** |
| Violation recall | 21.6 % | 82.7 % | 80.2 % | **89.2 %** |
| Missed hotspots | 638 | 141 | 161 | **88 (lowest)** |
| MAE (mV) | 9.22 ± 0.15 | 3.59 ± 0.08 | 3.08 ± 0.06 | **1.87 ± 0.03** |
| RMSE (mV) | 11.45 ± 0.18 | 4.96 ± 0.10 | 4.85 ± 0.09 | **2.84 ± 0.05** |
| Spearman ρ | 0.958 | 0.958 | 0.952 | **0.988** |
| Systematic bias (mV) | −9.22 | −1.98 | +0.03 | **+0.25** |
| Conformal coverage (PICP) | n/a | n/a | 81.7 % | **82.1 %** |
| Mean interval width | — | — | 10.27 mV | **4.39 mV** |

**The bias-asymmetry trap — why ranking alone deceives.** `physics_only` reaches an impressive
Spearman ρ = 0.958, yet **misses 638 of 814 real violations** (21.6 % recall), because a
systematic **−9.22 mV** bias keeps hotspots near the 45 mV threshold from ever tripping the
alarm. Under-prediction lets a chip tape out with undetected timing faults; over-prediction only
causes harmless conservative strap widening. When false negatives are penalised as engineering
signoff actually penalises them (F-β, β > 1.42), **hybrid dominates every variant** —
β = 1.5: 0.8638 vs 0.8589; β = 2.0: 0.8734 vs 0.8474.

**Conformal calibration.** Raw nominal 80 % quantile bands achieved only **48.2 %** empirical
coverage on unseen designs. Inductive conformal calibration on held-out calibration designs
restores **82.1 % coverage at a 4.39 mV interval width** — 2.5× tighter than the
width-proportional alternative (15.66 mV / 94 %).

**Grouped permutation importance** confirms the physics prior is the dominant anchor:
`phys_` +4.485 mV MAE when shuffled, then `grid_` +0.266, `cur_` +0.119, `scn_` +0.115,
`conc_` +0.019.

### 9.4 Risk ranking, mitigation, DSE, telemetry

- **Ranking by effective slack reorders 87.6 % of paths** (1,403 / 1,601) relative to naive
  STA-slack ranking (Spearman ρ = 0.9985, max shift 203 positions). The droop-aware ranking is
  materially different from "sort by millivolts" — this is the project's core argument, measured.
- **83 / 1,601 paths (5.18 %)** fall in the at-risk set (worst-5 % effective slack); **0** have
  negative effective slack. Per-path droop delay penalties span **13.1 – 37.4 ps**.
- **Mitigation Pareto (re-solved, verified):** 37.7 ps recovered at budget 10 → 69.6 ps at
  budget 50, with the chosen fix set changing along the curve.
- **DSE:** 512 Sobol screening samples (tier 1), a 5-point NSGA-II Pareto front (tier 2), and a
  GP-BO surrogate (tier 3) fitted on the 6 real measured PnR points, proposing 4 new unobserved
  candidates for a future real run. Candidate diversity is enforced by a greedy normalised
  spacing constraint `d ≥ 0.25` in 5-D knob space, so BO does not waste PnR budget on four
  near-identical configurations.
- **Telemetry:** **K = 4** TDET/PDET sensors placed on the 24 × 24 risk grid by greedy submodular
  coverage, sensing radius R = 2.5 grid units ≈ 46.3 µm (101 tiles per sensor). Coverage is
  reported as a cumulative spatial sum of per-tile path risk in picoseconds — Sensor 1 at grid
  (15, 13) covers 1,088.96 ps across its 101 tiles.

### 9.5 Real-data transfer study

The identical synthetic-trained pipeline was evaluated against **8 real ORFS designs**
(`orfs_gcd`, `orfs_ibex`, six `orfs_ssd_ctrl_*` variants):

1. **Zero pipeline failures** — all 8 validated with 0 schema errors, 576 × 32 feature matrices,
   zero NaNs or infinities.
2. **The real supply-mesh discovery** — with the delivered `bumps.csv` (power taps in a single
   central column) the physical mesh had negative R². Correcting the boundary condition to the
   true all-die M1–M4–M7 strap mesh let the coarse solver alone recover 21–47 % of signoff
   variance (R² = 0.47, ρ = 0.58).
3. **The zero-violation reality** — open-source designs at this scale are heavily
   over-provisioned; the maximum drop across 27,648 real tiles was 11.84 mV against a 55 mV
   budget, with **zero violations**. This is precisely why the synthetic stress corpus was
   mandatory: real open-source testchips operate far from signoff margins, leaving hotspot
   classification undefined without it.
4. **Transferability** — the physics engine and residual formulation transfer across tools;
   regression weights require local fine-tuning for node-specific drive strengths and PDN pitch.

---

## 10. Repository layout

```
rtl/                     parameterised SystemVerilog for ssd_ctrl_top (12 modules)
  params.svh             GENERATED by scripts/gen_config.py — never hand-edit
configs/
  param_space.json       legal knob value sets (single source of truth)
  cfg_*.json             design configurations (small / mid / large / BO runs)
  bo_runs.csv            BO proposal → config → strap pitch → rank
scripts/
  gen_config.py          config JSON → rtl/params.svh, refuses to widen a range
  bo_to_configs.py       C's BO proposals → gated config files
  orfs_sweep_summary.py  8 ORFS runs → out/orfs/ssd_ctrl_sweep_summary.csv
  extract_activity.py    VCD → out/activity/toggle_rates.csv
flow/
  synth.sh               lint + Yosys synth + gate-level stat per config
  run_orfs.sh            one design through OpenROAD-flow-scripts
  ssd_ctrl_orfs.sh       ssd_ctrl_top through ORFS, optional strap-pitch override
  orfs_extract.tcl       the ONLY ORFS parser (OpenDB + OpenSTA + PSM)
  run_sim.sh             Verilator testbench, 6 scenarios, activity extraction
prism/
  solver.py              PDNSolver — CSC assembly, splu factorisation, Ohm's law
  design.py              synthetic floorplan & current-map generator
  features.py            the 32 feature extractors
  audit.py               leakage trap banning signoff arrays during extraction
  model.py               HistGradientBoosting point + quantile, conformalisation
  evaluate.py            GroupShuffleSplit CV, permutation importance, bootstrap CIs
  timing.py              volts → picoseconds risk engine
  mitigation.py          re-solve-based mitigation scoring
  io_csv.py              schema validation — every load goes through this
  orfs.py                ORFS output → the 9-CSV data contract
  viz.py                 figure rendering
dse/                     Sobol / NSGA-II / BO design-space exploration
data/orfs/               8 designs in the 9-CSV contract + manifest.csv
out/
  synth/stats.csv        per-config cell / flop / area
  activity/toggle_rates.csv   measured per-module switching activity
  orfs/<design>/         raw ORFS results, reports, logs, IR heatmap (.webp)
app.py                   Streamlit dashboard (5 tabs)
run_all.py               pipeline driver — one subcommand per block
```

### The 9-CSV data contract

Every design — synthetic or extracted from a real tool run — enters through identical schemas
verified by `prism/io_csv.py`:

```
data/<corpus>/<design_id>/
├── design_stats.csv   (1 row)      die dimensions, utilisation, vdd, clock
├── modules.csv        (~8 rows)    hierarchy definitions & rated power
├── macros.csv         (2-5 rows)   hard macro bounding boxes
├── instances.csv      (N rows)     cell coords, area, capacitance, sequential flag
├── bumps.csv          (M rows)     C4 power bump coordinates
├── strap_planned.csv  (9216 rows)  96×96 planned strap conductance
├── activity.csv       (48 rows)    per-scenario switching activity multipliers
├── paths.csv          (P rows)     timing paths with chained instance IDs
└── irmap.csv          (55296 rows) [LABEL ONLY] 96×96 fine voltage drop per scenario
```

---

## 11. Running the project

### Environment

All EDA runs inside a Linux environment (this build used WSL2 Ubuntu):

- **OSS CAD Suite** — Yosys, Verilator, Icarus (`source ~/oss-cad-suite/environment`)
- **OpenROAD-flow-scripts** — built locally (`./build_openroad.sh --local`); the prebuilt
  `openroad/orfs` Docker image SIGILLs at CTS on some hybrid-core CPUs under WSL2
- **Python** — `pip install -r requirements.txt`
- **Nangate45 PDK** — `pdk/nangate45.lib`

### RTL & flow (Role A)

```bash
python scripts/gen_config.py --config configs/cfg_small.json --out rtl/params.svh
flow/synth.sh cfg_small                     # lint + Yosys synth, one config
flow/run_sim.sh                             # Verilator testbench, 6 scenarios, activity
flow/run_orfs.sh nangate45 gcd              # one open-source design through ORFS
flow/ssd_ctrl_orfs.sh cfg_run_01 12         # our design through ORFS, strap pitch 12 µm
python -m prism.orfs --design ssd_ctrl_cfg_small --orfs out/orfs/ssd_ctrl_cfg_small
python scripts/orfs_sweep_summary.py        # roll the 8 runs into one summary CSV
```

### Predictor, risk, DSE (Roles B, C)

```bash
python run_all.py gen        # synthetic corpus (14 designs × 6 scenarios)
python run_all.py features   # 32 features + leakage audit
python run_all.py train      # 4 model variants + conformal calibration
python run_all.py eval       # ablation metrics + bootstrap CIs
python run_all.py figures    # 200 DPI publication figures
python run_all.py all        # end to end
pytest tests/test_gates.py -v
```

### Dashboard

```bash
streamlit run app.py
```

Five tabs: **Predict** (predicted / signoff / error heatmaps with conformal band toggles) ·
**Validate** (ablation table, live leakage audit button) · **Scenarios** (6-panel small multiples,
tiles-over-budget ranking) · **Findings** (why hybrid beats pure ML) · **Custom Upload** (design
bundle ZIP → feature extraction → inference).

---

## 12. What is real vs. modelled

| Result | Source | Status |
|---|---|---|
| IR-drop heatmaps, worst/avg IR per config | OpenROAD PDNSim static solve, Nangate45 | **measured** |
| Strap-pitch / ECC-lane / scale sensitivity | 8 real place-and-route runs | **measured** |
| Cell / flop / area per config | Yosys gate-level `stat` | **measured** |
| Per-module switching activity | Verilator simulation, 6 scenarios | **measured** |
| Timing paths, slack, per-domain clock periods | OpenSTA `report_checks` on routed DEF | **measured** |
| Rank churn, at-risk set, mitigation Pareto | re-solving the extracted grid | **computed** from measured inputs |
| Hybrid predictor metrics | ML on the 14-design synthetic corpus | **computed** — see §13 |
| DSE Pareto front (tier 2) | surrogate model, not yet a real run | **surrogate-predicted** |
| BO next candidates | GP-LCB over unobserved knob space | **proposed**, not run |

The synthetic corpus and the real `ssd_ctrl` runs sit at different droop scales (~76 mV vs
~1.8 mV) and are **not cross-scaled**. Proper cross-validation needs the predictor retrained on
the real `orfs_ssd_ctrl` layout.

---

## 13. Known open items

- **Physics-baseline R² is quoted inconsistently across our own documents** — 0.80 (presentation),
  0.485 (Role A independent recompute), −0.12 (Role B ablation table). The hybrid figure (≈ 0.966)
  is consistent across sources. This must be reconciled to a single number before any baseline R²
  is quoted as final.
- **Hybrid R² jump** — the baseline-to-hybrid improvement on holdout is larger than a
  residual-only correction should produce, and is being checked for feature/partition leakage.
- **`paths.csv` clock-domain labels** — 25 rows are recovery/removal checks and true async CDC
  paths that OpenSTA groups outside the three named clocks. Whether they carry their own label,
  are dropped, or stay folded into `clk_core` is an open team decision, not something to relabel
  silently to pass a gate.
- **`cfg_large` through ORFS** — skipped (≈ 2.8 M instances, 2.5–4 h, OOM risk); `cfg_small` and
  `cfg_mid` already establish the architectural-scale trend.
- **Predictor ↔ real-design cross-validation** — pending training on `orfs_ssd_ctrl`.

---

## 14. Final overview

PRISM transforms early IR-drop analysis by bridging exact physics equations with residual ML
predictions. Physics solvers evaluate voltage maps, timing engines translate millivolts to
picoseconds, knapsack algorithms select measured fixes, and telemetry loops map sensor placement.

> **Converting millivolts into timing slack turns risk identification into clear, practical, and
> measurable budget optimization.**

---

## 15. Further reading

| Document | What it covers |
|---|---|
| `TEAM_README.md` | The full argument for PS1, interface contracts, novelty pillars |
| `MODEL_README.md` | Role B — the complete theoretical & engineering reference for the model |
| `PROGRESS.md` | Role A's session log — every block, gate, bug found, and fix |
| `docs/DESIGN.md` | `ssd_ctrl_top` — hierarchy, clock domains, the 8 knobs, the ECC hotspot |
| `docs/DATA_SCHEMA.md` | The 9-CSV data contract, column by column, with validation rules |
| `docs/ML_INPUT_SPEC.md` | The 32 features, grids, leakage rules |
| `docs/RUNBOOK.md` | Per-block detail and the sanity checks each must pass |


