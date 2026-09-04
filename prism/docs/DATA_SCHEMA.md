# DATA SCHEMA — CSV Contract for PRISM

> This document is the authoritative schema for all data exchanged between
> roles A and B.  Every CSV file described here must conform exactly.
> Role A must produce these files; role B's `io_csv.py` validates them.

---

## Directory shape

```
data/<corpus>/<design_id>/
    design_stats.csv      1 row     design-level scalars
    modules.csv           ~8 rows   module → clock/power domain
    macros.csv            2-5 rows  hard macro rectangles
    instances.csv         N rows    every placed cell        (gzip if > 50 MB)
    bumps.csv             M rows    power bump coordinates
    strap_planned.csv     9216 rows fine-grid PLANNED strap density   [FEATURE]
    activity.csv          48 rows   per (scenario, module) multipliers
    paths.csv             P rows    timing paths  → handed to role C
    irmap.csv             55296 rows fine-grid drop per scenario      [LABEL ONLY]
data/<corpus>/manifest.csv          index of all designs
```

`.csv.gz` is accepted anywhere; `io_csv.py` sniffs the extension.

---

## `design_stats.csv` — 1 row

Role A already produces the first block. **The second block is missing and they must
add it** — without die dimensions nothing spatial can be constructed at all.

```
# --- already produced by role A's Yosys step ---
design_id, config, hash, cells, flops, xor_cells,
chip_area_um2, seq_area_um2, seq_pct, lint_rc, ts_utc

# --- ROLE A MUST ADD THESE. Blocking. ---
die_w_um, die_h_um,          float   DEF DIEAREA
vdd_v,                       float   0.90 for nangate45
clock_period_ns,             float   from SDC
core_util,                   float   0.55-0.85
bump_pitch_um,               float   150 / 200 / 300
strap_pitch_um,              float   8 / 12 / 16 / 24
strap_width_um,              float   0.4 / 0.8 / 1.6
pdn_layers,                  str     e.g. "M4-M7"
```

Reference row from role A's current output, for column-name matching:
```
cfg_small,f8b182b2ad48,26793,7961,233,63008.218,36683.528,58.22,1,2026-09-03T14:23:36Z
```

**`design_id` is the GroupKFold group key.** It must be stable across reruns. Use
`config` if unique, else `config + "_" + hash[:8]`.

---

## The rest of the schema

| File | Columns | Notes |
|---|---|---|
| `modules.csv` | `module, clock_domain, power_domain, rated_power_mw` | one row per module instance in the hierarchy |
| `macros.csv` | `macro_id, module, x0_um, y0_um, x1_um, y1_um, power_mw` | drives `top_macro_frac`, `top_dmacro` |
| `instances.csv` | `inst_id, inst_name, module, cell_type, x_um, y_um, area_um2, cap_ff, is_seq, is_clk, is_macro` | `inst_id` is a stable integer, referenced by `paths.csv` |
| `bumps.csv` | `bump_id, x_um, y_um` | ≥ 1 required or `A` is singular |
| `strap_planned.csv` | `fy, fx, density` | long format, fine grid, `density ∈ [0,1]`, **must cover all 96×96 cells with no gaps** |
| `activity.csv` | `scenario, module, activity, mission_weight` | `activity ∈ [0,1]`; `mission_weight` constant per scenario, must sum to 1.0 |
| `paths.csv` | `path_id, endpoint, clock_domain, slack_ns, delay_ns, inst_ids` | `inst_ids` semicolon-joined ints. **Not optional** — role C attributes slack loss to the tiles a path crosses; an endpoint-only report cannot do that |
| `irmap.csv` | `scenario, fy, fx, drop_v` | **LABEL. Volts of drop, not absolute voltage.** From PDNSim |
| `toggle.csv` *(optional)* | `scenario, inst_id, toggle_rate` | from VCD if role A's testbench lands. Absent → fall back to `activity.csv` and log it |
| `manifest.csv` | `design_id, corpus, path, n_scenarios, n_instances, source` | `source ∈ {synthetic, orfs}` |

---

## `io_csv.py` — validation, and it must fail loudly

Every load goes through this module. Silent coercion is banned; a wrong file must stop
the run, not produce quiet garbage.

```python
def load_design(design_dir) -> Design       # validated, typed, unit-checked
def write_design(design, design_dir) -> None   # used by design.py in S2
def validate_design(design_dir) -> list[str]   # returns failures, empty == OK
def load_corpus(data_dir) -> list[Design]
```

Checks, each with a specific error message naming the file and column:

```
[ ] every required file present; every required column present
[ ] die_w_um > 0 and die_h_um > 0
[ ] UNIT SANITY: max(inst_x_um) <= die_w_um * 1.01
    → if inst coords are ~1000x the die, role A exported DEF database units,
      not microns. This is the single most common ingest bug. Name it explicitly.
[ ] all instance coords inside the die; macros inside the die
[ ] bumps.csv non-empty
[ ] strap_planned covers all ny_fine x nx_fine cells, no duplicates, density in [0,1]
[ ] activity mission_weight sums to 1.0 +/- 1e-6 across scenarios
[ ] every module in activity.csv and instances.csv exists in modules.csv
[ ] every inst_id in paths.inst_ids exists in instances.csv
[ ] irmap: full grid per scenario, drop_v in [0, 0.2] V
    → if max(drop_v) > 1.0 role A exported absolute voltage or millivolts. Reject.
[ ] irmap NON-UNIFORM: std(drop_v) > 1e-4 for every scenario
    → a flat map means the analysis ran without doing anything useful and it will
      silently poison every label. Refuse to ingest it.
```

---

## Design-level columns: a leakage hazard

`design_stats.csv` is constant within a design. With only 14 designs and GroupKFold by
design, a design-constant column lets the model **fingerprint the design instead of
learning physics**. It will inflate CV scores and collapse on anything new.

```
PERMANENTLY BANNED AS FEATURES — enforce in feature_columns()
  hash, config, design_id, ts_utc, lint_rc, any path or filename
```

---

## ⚠️ BLOCKING for Role A

Role A's `design_stats.csv` currently does NOT contain:
- `die_w_um`, `die_h_um` (from DEF DIEAREA)
- `vdd_v` (0.90 for nangate45)
- `clock_period_ns` (from SDC)
- `core_util` (0.55–0.85)
- `bump_pitch_um` (150/200/300)
- `strap_pitch_um` (8/12/16/24)
- `strap_width_um` (0.4/0.8/1.6)

**Nothing spatial can be built without these.** This is blocking.

---

## `out/predictions.csv` — Frozen Delivery Contract with Role C

Role B provides `out/predictions.csv` for downstream consumption by Role C (adjoint solve, timing risk, and slack attribution).

### Exact Columns and Order
```
design, scenario, partition, tile_id, pred_v, lo_v, hi_v, label_v, coarse_v
```

All voltages are reported in **Volts** (not millivolts). Exactly 48,384 rows (14 designs × 6 scenarios × 576 tiles).

- `design`: design identifier (e.g. `syn_000`)
- `scenario`: workload scenario (e.g. `seq_read`, `gc_compact`)
- `partition`: dataset partition in `{"train", "calib", "holdout"}` defined by `models/manifest.json`
- `tile_id`: stable deterministic integer index in `[0, 575]`
- `pred_v`: point prediction from canonical hybrid model in Volts
- `lo_v`: lower conformal bound (additive-only) in Volts
- `hi_v`: upper conformal bound (additive-only) in Volts
- `label_v`: ground truth signoff IR drop in Volts
- `coarse_v`: baseline coarse physics solve in Volts (`phys_base_v`)

### Deterministic `tile_id` Coordinate Mapping
On the coarse grid of dimensions `nx_coarse = 24`, `ny_coarse = 24`:
- **Forward formula**:
  $$\text{tile\_id} = ty \times 24 + tx$$
- **Inversion formula**:
  $$ty = \lfloor \text{tile\_id} / 24 \rfloor$$
  $$tx = \text{tile\_id} \pmod{24}$$
where $ty, tx \in [0, 23]$ are 0-indexed tile grid coordinates.
