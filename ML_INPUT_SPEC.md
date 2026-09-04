# PRISM — ML Input Specification

For **role B (Physics & Prediction)**, produced by **role A (RTL & Flow)**.

Two things are specified here:

1. **§1–2** the 32 features the model consumes and the target it predicts —
   what B sets up.
2. **§3** the raw artefacts those features are computed *from* — what A must
   extract out of ORFS so the same pipeline runs on real data.

Agree §3's column names before either of you writes code. That handoff is the
likeliest thing to break.

---

## 0. Shape of the problem

| | |
|---|---|
| **One row =** | one (design, scenario, coarse tile) |
| **Rows per design** | `n_scenarios × ny_coarse × nx_coarse` = 6 × 24 × 24 = **3,456** |
| **Corpus** | 14 designs → **48,384 rows × 32 features** |
| **Target** | IR-drop residual, volts |
| **Grids** | fine 96×96 (labels), coarse 24×24 (prediction). Ratio must be an integer. |
| **Split** | **by design**, never by tile |

The coarse grid is the prediction resolution. The fine grid is what the
sign-off-fidelity solve runs on; the label for a coarse tile is the **maximum**
fine-grid drop inside it, because a hotspot is a local worst case and averaging
it away makes the task easy and useless.

---

## 1. The 32 features

All are `float64`, one row per tile per scenario. Produced by
`prism/features.py :: design_features()`.

### `phys_*` — the physics prior (4)
The floorplan-time coarse solve and its neighbourhood. **This is the half of the
hybrid that is not learned.**

| Feature | Meaning | Units |
|---|---|---|
| `phys_base_v` | coarse-grid solve of `(L + D_bump)·U = I` | V |
| `phys_base_s1` | same, Gaussian-blurred σ=1 tile | V |
| `phys_base_s2` | same, σ=2 tiles | V |
| `phys_base_rank` | percentile rank of `phys_base_v` within the design | 0–1 |

### `grid_*` — grid strength, activity-independent (6)

| Feature | Meaning | Units |
|---|---|---|
| `grid_weak` | coarse solve with 1 A spread uniformly — a pure property of the planned grid and bump array. The floorplan-time stand-in for "effective resistance to the supply" | V |
| `grid_strap_mean` | mean planned strap density in tile | 0–1 |
| `grid_strap_min` | min planned strap density in tile | 0–1 |
| `grid_bumps` | bumps falling inside the tile | count |
| `grid_dbump_min` | min distance from any fine cell in the tile to a bump | fine tiles |
| `grid_dbump_max` | max of the same | fine tiles |

### `cur_*` — current draw at several scales (7)
Droop is regional: a tile is hot partly because its neighbours are hot.

| Feature | Meaning | Units |
|---|---|---|
| `cur_sum` | total current in tile | A |
| `cur_max_fine` | max current in any fine cell inside the tile | A |
| `cur_s1` / `cur_s2` / `cur_s4` | `cur_sum` blurred at σ = 1, 2, 4 tiles | A |
| `cur_x_weak` | `cur_sum × grid_weak` | A·V |
| `cur_s2_x_weak` | `cur_s2 × grid_weak` | A·V |

The two interaction terms matter: high current in a strong-grid region is fine,
high current in a weak-grid region is the hotspot.

### `conc_*` — sub-tile concentration (3)
**The largest single source of the physics baseline's error.** The coarse solve
cannot see structure below its own resolution.

| Feature | Meaning | Units |
|---|---|---|
| `conc_ratio` | max fine-cell current ÷ mean fine-cell current in tile | ratio |
| `conc_top4` | share of tile current in its 4 hottest fine cells | 0–1 |
| `conc_x_weak` | `conc_ratio × grid_weak` | — |

### `top_*` — topology and netlist density (8)

| Feature | Meaning | Units |
|---|---|---|
| `top_macro_frac` | fraction of tile covered by hard macro | 0–1 |
| `top_dmacro` | distance to nearest macro edge | fine tiles |
| `top_edge_dist` | distance to die edge | coarse tiles |
| `top_util` | placement utilisation (cell area ÷ tile area) | ratio |
| `top_cells` | instance count in tile | count |
| `top_capden` | switched capacitance density | fF/µm² |
| `top_clkden` | clock-buffer fraction of cells in tile | 0–1 |
| `top_seqden` | sequential-cell fraction of cells in tile | 0–1 |

### `sw_*` — simultaneous switching (2)
Proxy for how correlated the switching in this tile is.

| Feature | Meaning | Units |
|---|---|---|
| `sw_hhi` | Herfindahl index of current share by clock domain. 1.0 = one domain owns the tile, so everything here switches on the same edge | 0–1 |
| `sw_topshare` | largest single clock domain's share | 0–1 |

### `scn_*` — which scenario this row is (2)

| Feature | Meaning | Units |
|---|---|---|
| `scn_power_frac` | this scenario's total power ÷ design's rated power | ratio |
| `scn_weight` | mission-profile weight of the scenario | 0–1, sums to 1 |

`scenario` itself is dropped as a string; these two carry it numerically.

### Identifier columns (not features)
`design`, `scenario`, `ty`, `tx` — kept for grouping and joins,
excluded by `feature_columns()`.

---

## 2. Target and model setup

### Target
```python
y = label_v - phys_base_v          # the RESIDUAL, volts
prediction = phys_base_v + model.predict(X)
```

The model never predicts droop. It predicts what the physics solve gets wrong.
`label_v` (volts) is the max fine-grid drop in the tile, added by
`features.add_labels()`.

**Ablation for the slides** — `use_physics=False` drops the four `phys_*`
columns and regresses on the label directly. That is the "learned only" row.

### Estimator
`sklearn.ensemble.HistGradientBoostingRegressor` — no xgboost or lightgbm.

```python
max_iter            = 400
learning_rate       = 0.06
max_leaf_nodes      = 31
min_samples_leaf    = 20
l2_regularization   = 1.0
early_stopping      = True
validation_fraction = 0.15
random_state        = 0
```

Three fitted per model: median (`squared_error`), plus `quantile` at 0.10 and
0.90 for the interval.

### Split protocol — do not deviate
- **Group by `design`.** Never a random tile split: neighbouring tiles are
  strongly correlated and a tile split gives a flattering number that says
  nothing about a new block.
- Holdout ≈ 25% of designs (3 of 14).
- Of the remaining training designs, ~25% again are held out as a **conformal
  calibration** set — they are not used to fit.

### Conformal calibration — not optional
Raw quantile bands covered **48%** when they promised 80%. Conformalised
quantile regression widens the band using a calibration set, taking the wider of
an additive and a width-proportional correction. Measured: **PICP 0.860** at
target 0.80. If B reproduces PICP far from 0.80, the interval is not
trustworthy and should not be quoted.

### Metrics
Headline is **violation F1 at the 45 mV budget**, not top-5% hit rate. The
physics baseline ranks well (Spearman 0.958) but is biased low, so it catches
2% of real violations against the hybrid's 77%. Bootstrap CIs on top-5% overlap
between models — that metric does not separate them.

---

## 3. What role A must deliver

These are the raw artefacts `features.py` computes from. Everything below is
available at or just after floorplan.

### 3.1 `artefacts.npz` — geometry, netlist, floorplan

| Key | dtype / shape | Units | ORFS source |
|---|---|---|---|
| `die_w`, `die_h` | float | µm | DEF `DIEAREA` |
| `nx_fine`, `ny_fine` | int | — | our choice, 96 |
| `nx_coarse`, `ny_coarse` | int | — | our choice, 24. Fine must be an integer multiple |
| `inst_x`, `inst_y` | float[N] | µm | DEF `COMPONENTS` placement |
| `inst_area` | float[N] | µm² | LEF cell size |
| `inst_cap_ff` | float[N] | fF | LEF pin cap + est. net cap |
| `inst_toggle` | float[N] | switches/cycle | VCD if session A4 lands, else default |
| `inst_module` | int[N] | index | hierarchy prefix of the instance name |
| `inst_is_seq` | bool[N] | — | cell is a flop/latch (liberty `ff` group) |
| `inst_is_clk` | bool[N] | — | instance drives a clock net |
| `bump_xy` | float[M,2] | µm | floorplan / package bump array |
| `strap_density` | float[ny_fine, nx_fine] | 0–1 | **planned** PDN straps, `pdn.tcl` |
| `clock_period_ns` | float | ns | SDC |
| `vdd` | float | V | 0.90 for nangate45 |

**Module table** (one row per module index):
`name`, `clock_domain`, `power_domain`, plus an activity multiplier per scenario.

**Macro table**: `x0, y0, x1, y1` (µm), `power_w`, `module_idx`.

### 3.2 `paths.csv` — the timing report

| Column | Type | Units |
|---|---|---|
| `endpoint` | str | — |
| `slack_ns` | float | ns |
| `delay_ns` | float | ns |
| `inst_idx` | list[int] | indices into the instance arrays |
| `clock_domain` | str | — |

**`inst_idx` is not optional.** `prism/timing.py` attributes slack loss to the
tiles a path crosses; an endpoint-only report cannot do that, and the risk
engine is the project's strongest differentiator. Report the cells on the path,
not just where it ends.

### 3.3 `irmap.npy` — the label

`float[ny_fine, nx_fine]`, volts of drop (not absolute voltage). From PDNSim /
`analyze_power_grid`.

**Must be non-uniform.** A flat map means the analysis ran without doing
anything useful, and it will silently poison every label.

### 3.4 Calibration constants
`sheet_cond`, `bump_cond` (siemens). On synthetic designs `solver.calibrate()`
sets these so the reference scenario just meets the 5% budget. On real ORFS data
they should be **fitted** so the solver reproduces the PDNSim map — that fit is
itself a validation of the mesh abstraction and worth a slide.

---

## 4. Minimum viable inputs

If ORFS cannot supply something, these fallbacks keep the pipeline running.
Every one of them is a documented assumption, so log it.

| Missing | Fallback | Cost |
|---|---|---|
| `inst_toggle` from VCD | `_ACTIVITY_TABLE` in `design.py` | activity becomes assumed, not measured |
| `inst_cap_ff` | area × 2.2 fF/µm² | weakens `top_capden`, `cur_*` |
| `inst_is_clk` | name match on `clk`/`CTS` | weakens `top_clkden` — a strong feature, try harder |
| macro `power_w` | 20% of block power, split by area | affects hotspots near macros |
| `strap_density` | uniform 1.0 | **kills `grid_strap_*`.** Extract it if at all possible |
| `paths.inst_idx` | launch/capture only | risk engine degrades to endpoint attribution |
| `irmap.npy` | our own solver | this is the circularity a judge will probe |

---

## 5. Sanity checks before B trains on anything

- [ ] No NaNs or infinities in any feature column
- [ ] `label_v` spans roughly 0.1 – 90 mV; a max under 45 mV means nothing
      violates the budget and the demo has no story
- [ ] `phys_base_v` correlates ~0.95 with `label_v` but is **biased low** by
      5–11 mV. If the bias is near zero, the two fidelities have collapsed into
      one and there is nothing left to learn
- [ ] Row count = `n_designs × 6 × 576`
- [ ] `run_all.py b2` reports the **leakage audit PASS** — it traps
      `solver.ground_truth`, `as_built_strap` and `label_map` during feature
      extraction and fails loudly if any is called
- [ ] Every feature is computable from §3 alone. Anything needing the as-built
      grid or the sign-off solve is a label, not a feature
