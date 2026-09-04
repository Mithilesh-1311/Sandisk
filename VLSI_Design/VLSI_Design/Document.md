# PRISM — Team Brief

**P**ower-integrity **R**isk **I**dentification, **S**lack-impact ranking and **M**itigation

SanDisk One-Day University Hackathon · Theme: VLSI Design · **Problem Statement 1**

> Read this whole file before you write a line of code. The interfaces in §7 are
> what let four people work in parallel without blocking each other. If we get
> those wrong, we lose more time to merge conflicts than we save by splitting up.

---

## 1. The decision in one paragraph

We are doing **PS1 (IR-drop prediction and mitigation)**, not PS2 (TDET/PDET
telemetry). PS2 is the less crowded problem and has a higher novelty ceiling,
but 25% of its grade is *"demonstrated power-speed benefit"* — a number that sits
at the end of a chain of unverifiable assumptions (sensor reading → estimated
field → margin decision → voltage → power) and that nobody can validate without
silicon. PS1's equivalent 25% line is *hotspot prediction accuracy*, which we can
prove in a table with error bars against a solve we control. **With a fixed time
box, take the problem whose claims are falsifiable.** We still get the PS2 story:
telemetry placement rides along as our closing module (§6.6), which is what the
hackathon's own title is about.

---

## 2. Why PS1 over PS2 — the full argument

| | PS1 — IR-drop | PS2 — Telemetry |
|---|---|---|
| Biggest rubric line | Hotspot accuracy **25%** | Demonstrated power-speed benefit **25%** |
| Can we *prove* that line? | Yes — hold out designs, predict, compare to a solve | No — requires measured silicon |
| Ground truth | A linear system we can solve exactly | Needs a thermal model **and** a process-variation model, both invented |
| Judging question given? | **Yes, verbatim** on the slide | No — we'd have to guess the target |
| Day-1 output | A heatmap. Reads from across a room | A sensor list. Needs explaining |
| Mitigation evidence | *Measured* by re-solving the grid | Narrative |
| Crowdedness | High (PowerNet, IREDGe, ICCAD'20 contest) | Low |
| Time to first result | ~4 h | ~12 h |

The crowdedness of PS1 is a real cost, and we pay for it in §4 — our novelty is
deliberately **not** in the prediction step, because that step is where everyone
else will compete.

### The honest counter-argument
If our team's strength were optimisation theory rather than anything
EDA-shaped, or if we knew every other team was doing PS1, PS2 would be the
better bet. Neither is true, and neither is worth the execution risk.

---

## 3. What we are building

A framework that takes **floorplan-stage artefacts** — synthesised netlist,
preliminary placement, timing report, preliminary power grid, activity
estimates — and produces:

1. a **predicted IR-drop hotspot map** with an uncertainty band,
2. a **risk ranking** of tiles, blocks, power domains and operating scenarios,
   ranked by *picoseconds of timing slack actually lost*, not by millivolts,
3. a set of **mitigation recommendations** whose benefit is *measured* by
   re-solving the grid, ranked on a **Pareto curve of benefit vs implementation
   cost**,
4. a **design-space exploration** that answers the bigger question — not "how do
   we patch this floorplan" but "which floorplan and PDN configuration should we
   have built in the first place" (§8),
5. a **TDET/PDET placement** that closes the loop from design-time prediction to
   runtime verification (§6.6).

### The one-sentence pitch

> IR-drop is not a voltage problem. It is a timing problem measured in volts —
> and once you convert millivolts into picoseconds, ranking becomes meaningful
> and mitigation becomes a budget optimisation instead of a guess.

---

## 4. Novelty — the five pillars

Memorise these. Every one of them is a slide, and every one of them is a
defensible answer to *"how is this different from PowerNet?"*

### N1 · Physics-guided residual learning
We do **not** predict IR-drop. We predict the **residual that a floorplan-time
physics solve cannot explain**:

```
U_hat  =  U_coarse_solve  +  g(early features)
```

*Why it's better:* a CNN on a power map has to re-learn Ohm's law, current
spreading and bump topology from a handful of training designs, and it breaks on
a floorplan whose bump pitch or die size it has never seen. The coarse solve
already knows all of that **exactly**. What it cannot know is (a) the strap map
that detailed routing will actually build, (b) current concentration below its
own resolution, (c) via degradation at macro edges. Those are local, statistical
and learnable — so that is the *only* thing we ask the model to learn.

*Measured today:* the coarse baseline alone sits at R² ≈ 0.80 with a systematic
+5 to +11 mV underestimate. That gap is the thing we learn.

### N2 · Volts → picoseconds (the risk answer)
Alpha-power law: `delay ∝ V / (V − Vth)^α`, so the sensitivity
`d(ln delay)/d(ln V) ≈ −0.95` at Vdd = 0.9 V, Vth = 0.3 V, α = 1.3. Roughly
**1% droop costs 1% delay**.

For every path in the timing report, sum the droop-induced delay of the cells on
it and subtract from its slack → **effective slack**. Then rank by slack
actually lost.

> 40 mV of droop where there is 500 ps of slack is a non-issue.
> 15 mV on a critical path is a respin.

Ranking by millivolts cannot tell those apart. Ranking by effective slack can.
**This is our single strongest differentiator** and it is what the 20% "risk
assessment" line is asking for.

### N3 · Mitigations measured, not asserted
Every recommendation's benefit is computed by **re-solving the power grid** with
the change applied. Not a rule of thumb, not a lookup table. Because the drop is
*exactly linear in current* (§6.1), current-reducing fixes are exact and free to
evaluate. Then a knapsack under an area / routing / effort budget produces a
**Pareto curve of picoseconds recovered per unit cost** — which is verbatim what
the judging question calls a "measurable benefit-to-cost advantage".

### N4 · The surrogate is what makes DSE possible
A real PnR + signoff evaluation of one configuration takes ~20 minutes. Our
predictor takes ~50 milliseconds. That is a **~24,000× speedup**, and it is the
whole reason design-space exploration over PDN and floorplan knobs is even
thinkable. We use Bayesian optimisation to decide which handful of configurations
deserve a *real* PnR run, and NSGA-II on the surrogate to map the full Pareto
front. See §8.

### N5 · Scenario economics
Weight each operating scenario by its share of the mission profile and report
**expected** risk. This lets us say things like:

> "GC compaction is 10% of runtime but 60% of expected IR risk."

Which is exactly the statement's "identify and rank high-risk ... operating
scenarios", and no one ranks scenarios by anything but peak power.

### Bonus: the scalability argument
The conductance matrix factorises **once**; each new operating scenario is only a
new right-hand side. Commercial tools re-run the full analysis per corner. That
is a real 10–50× argument, and Laplacians are the best-scaling linear system that
exists (algebraic multigrid is near-O(N)). Put this on the scalability slide.

---

## 5. Architecture

```
                    ┌──────────────────────────────────────────┐
                    │  PARAMETERISED RTL  (rtl/, §7)           │
                    │  ssd_ctrl_top #(NUM_CH, DATA_W, ...)     │
                    └────────────────┬─────────────────────────┘
                                     │ Yosys
                    ┌────────────────▼─────────────────────────┐
                    │  OpenROAD / ORFS                          │
                    │  floorplan → PDN → place → CTS → route    │
                    └──────┬──────────────────────┬─────────────┘
                           │ early artefacts      │ PDNSim IR map
                           │ (DEF, .rpt, netlist) │ (LABELS ONLY)
              ┌────────────▼──────────┐    ┌──────▼─────────────┐
              │  B1 FEATURES          │    │  B1' GROUND TRUTH  │
              │  30 early-only feats  │    │  fine-grid solve   │
              └────────────┬──────────┘    └──────┬─────────────┘
                           │                      │
                    ┌──────▼──────────────────────▼─────┐
                    │  B2  HYBRID PREDICTOR (N1)         │
                    │  coarse solve + GBM residual       │
                    │  + 10th/90th percentile band       │
                    └──────────────┬─────────────────────┘
                                   │ predicted drop map
                    ┌──────────────▼─────────────────────┐
                    │  B3  RISK ENGINE (N2, N5)          │
                    │  volts→ps, effective slack,        │
                    │  tile→block→domain→scenario rollup │
                    └──────────────┬─────────────────────┘
                                   │ ranked risk list
                    ┌──────────────▼─────────────────────┐
                    │  B4  MITIGATION + ROI (N3)         │
                    │  6 actions × re-solve × knapsack   │
                    └──────────────┬─────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
     │ B5  DSE (N4)    │  │ B6  TELEMETRY   │  │ B7  DASHBOARD  │
     │ NSGA-II + BO    │  │ TDET/PDET       │  │ Streamlit      │
     │ Pareto front    │  │ greedy coverage │  │ demo           │
     └─────────────────┘  └─────────────────┘  └────────────────┘
```

**The rule that keeps us honest:** anything derived from the as-built grid or the
signoff solve is a **label**, never a feature. There is an automated audit
(`prism/audit.py`) that asserts this. If a judge asks "did you leak the answer",
we run the audit on stage.

---

## 6. The solution, block by block

### 6.1 · B0 — Physics engine [P0]
Reduce the PDN to a resistive mesh: one node per tile, a conductance on each
edge, a conductance from each bump to the supply. KCL gives one linear system:

```
(L + D_bump) · U = I            U = Vdd − V  is the IR drop
```

Two properties of this equation carry the entire project:

* **U is exactly linear in I.** Any mitigation that removes current has an
  exactly computable benefit — no re-simulation error. This is what makes B4 work.
* **Scaling every conductance by `s` scales U by `1/s`.** Closed-form grid
  calibration instead of a search, and the matrix factorises once for all
  scenarios.

This mesh *is* the "additional artefact we derived" that the problem statement
explicitly invites us to justify. Say that out loud during the presentation.

### 6.2 · B1 — Two fidelities, and the gap between them [P0]
The methodological move that makes the whole thing honest.

| | Grid | Strap map | Current | Role |
|---|---|---|---|---|
| **Ground truth** | fine | **as-built** | per-instance | stands in for Voltus / RedHawk / PDNSim. **Labels only** |
| **Early estimate** | coarse | **preliminary** | tile-averaged | what a designer could genuinely compute at floorplan |

The gap between them is caused by exactly what causes it in real life: routing
congestion forces strap thinning nobody planned, via arrays degrade at macro
edges, and current concentrates below the resolution of any coarse estimate.

**If there were no gap, early IR prediction would already be a solved problem.**

### 6.3 · B2 — Hybrid predictor [P0]
Gradient-boosted trees (sklearn `HistGradientBoostingRegressor` — no xgboost
dependency, one less thing to install at 2 a.m.) on ~30 early-only features, in
seven groups: `phys_*`, `grid_*`, `cur_*`, `conc_*`, `top_*`, `sw_*`, `scn_*`.

Plus **10th and 90th percentile quantile models**, so the output is
*"31–48 mV, budget is 45"* rather than a fake-precise *"37.2 mV"*. The width of
the band is itself the signal for whether to spend engineering effort here.

**Validation rules — non-negotiable:**
* **Design-level holdout.** Never a random tile split. Neighbouring tiles are
  strongly correlated; a tile split gives a flattering number that says nothing
  about a new block. If anyone reports a tile-split R², we throw it away.
* **Headline metric = top-5% hit rate**, not MAE. What a PD team acts on is the
  *set of tiles they must go and look at*.
* **Always report the ablation:** physics baseline / learned-only / hybrid. The
  comparison is the evidence, not the absolute number.

### 6.4 · B3 — Risk engine [P0]
1. Convert droop → added delay per cell via the alpha-power law.
2. Sum along each timing path → **effective slack**.
3. Attribute each path's slack loss back to the tiles that caused it.
4. Roll up: **tile → block → power domain** (the deliverable spec asks for block
   and domain ranking, not just tiles).
5. Weight scenarios by mission-profile share → **expected risk** (N5).

Output: a ranked table a physical-design lead could act on Monday morning.

### 6.5 · B4 — Mitigation catalog and ROI [P0]

| Action | Physical effect | Cost model |
|---|---|---|
| Decap insertion | reduces effective peak current | area (µm²), leakage |
| Strap widening | raises local conductance | routing tracks consumed |
| Cell de-densification | spreads current | area, wirelength, timing |
| Extra bump / pad | new low-R supply node | package pins — discrete, expensive |
| Vt swap / downsize (non-critical only) | lowers current | timing margin |
| **Clock-skew staggering** | de-correlates simultaneous switching, cuts peak current 10–25% | CTS effort, hold buffers |

Clock-skew staggering is the cheap clever one nobody else will propose. Lead with
it in the presentation.

Each candidate is scored by **re-solving the grid**, then a 0/1 knapsack under an
area/routing/effort budget selects a package. Sweep the budget → Pareto curve.

### 6.6 · B6 — Telemetry extension [P1]
IR-drop risk becomes the **prior** for where TDET/PDETs go. Formulate placement
as submodular maximum-coverage (greedy has a provable 1−1/e bound), where a
sensor "covers" a region whose droop/thermal behaviour it predicts within ε
across scenarios. Report guardband reclaimed and the resulting power saving
(P ∝ V²f). One slide, ~90 minutes of work, and it ties us to the hackathon's own
title.

---

## 7. Parameterised RTL design

### Why we need it
Our biggest weakness is dataset size and realism. A parameterised design gives us
**many genuinely different physical implementations from one source**, each with
real netlists, real floorplans, real timing reports and real PDNSim IR maps. It
also turns "we tested on 3 designs" into "we swept a design space" — which is a
much stronger claim, and it is the input to §8.

### The design: `ssd_ctrl_top`
A plausible storage-controller block. Arithmetic-heavy enough to create real
switching activity, regular enough to synthesise cleanly.

```
ssd_ctrl_top #(...)
 ├── host_if          AXI4-Lite slave, command decode        [clk_host]
 ├── cdc_fifo × N     async FIFOs between domains            [clk_host↔clk_core]
 ├── dma_fabric       NUM_CH-way crossbar + arbiter          [clk_core]
 ├── ecc_engine       ECC_LANES × GF(2) syndrome/XOR tree    [clk_core]  ← hot
 ├── ch_ctrl[NUM_CH]  per-channel NAND sequencer + FIFO      [clk_nand]
 ├── sram_ctl         banked buffer controller               [clk_core]
 └── seq_core         small sequencer / register file        [clk_core]
```

### Parameter file — `rtl/params.svh`

```systemverilog
`ifndef PRISM_PARAMS_SVH
`define PRISM_PARAMS_SVH

package prism_params;
  // --- architectural knobs (swept by DSE) --------------------------------
  parameter int NUM_CH      = 4;    // NAND channels        {2, 4, 8}
  parameter int DATA_W      = 64;   // datapath width       {32, 64, 128}
  parameter int ECC_LANES   = 2;    // parallel ECC lanes   {1, 2, 4}
  parameter int ECC_DEGREE  = 12;   // GF(2^m) degree       {8, 12, 16}
  parameter int FIFO_DEPTH  = 32;   // per-channel FIFO     {16, 32, 64}
  parameter int PIPE_STAGES = 3;    // datapath pipe depth  {2, 3, 4}
  parameter int NUM_BANKS   = 4;    // SRAM buffer banks    {2, 4, 8}
  parameter bit CLK_GATE_EN = 1;    // fine-grain clock gating {0, 1}

  // --- derived ------------------------------------------------------------
  parameter int ADDR_W      = 32;
  parameter int STRB_W      = DATA_W/8;
  parameter int CH_ID_W     = (NUM_CH <= 2) ? 1 : $clog2(NUM_CH);
endpackage
`endif
```

**Rules for whoever writes the RTL:**
* Every module takes its sizes from `prism_params` or from its own
  `#(parameter ...)` list. **Zero hard-coded widths.** If you write `[31:0]`
  anywhere outside a leaf constant, you have broken the sweep.
* Use `generate ... for` loops for `NUM_CH` and `ECC_LANES` replication.
* Must be **Yosys-synthesisable**: no `initial` blocks outside testbenches, no
  delays, no `real`, no unsupported SystemVerilog constructs. Lint with Verilator
  before you hand it over.
* Every module gets a clock-gate enable path when `CLK_GATE_EN` is set — this is
  what creates the interesting activity *differences* between scenarios.
* Keep a `tb/tb_ssd_ctrl.sv` that drives the six scenarios (§ scenario list) so
  we can dump **real VCD toggle activity** instead of guessing toggle rates.
  If we get this working, our activity estimates stop being assumptions — say so
  in the presentation.

### Generating a configuration

```bash
# scripts/gen_config.py writes rtl/params.svh from a JSON config
python3 scripts/gen_config.py --config configs/cfg_0042.json --out rtl/params.svh
yosys -p "read_verilog -sv rtl/*.sv; synth -top ssd_ctrl_top; write_verilog out/netlist.v"
```

### Physical knobs (not RTL, but part of the same sweep)

| Knob | Values | Where it lives |
|---|---|---|
| `CORE_UTIL` | 0.55, 0.65, 0.75, 0.85 | ORFS `config.mk` |
| `ASPECT_RATIO` | 0.8, 1.0, 1.25 | ORFS `config.mk` |
| `PDN_STRAP_PITCH_UM` | 8, 12, 16, 24 | `pdn.tcl` |
| `PDN_STRAP_WIDTH_UM` | 0.4, 0.8, 1.6 | `pdn.tcl` |
| `PDN_LAYERS` | M4–M7, M5–M8 | `pdn.tcl` |
| `BUMP_PITCH_UM` | 150, 200, 300 | floorplan script |
| `DECAP_PCT` | 0, 2, 5, 10 | filler insertion |
| `CLK_PERIOD_NS` | 0.8, 1.0, 1.25 | `constraint.sdc` |

---

## 8. Design Space Exploration

### The problem DSE actually solves for us
Blocks B2–B4 answer *"this floorplan has a problem here, and here's the cheapest
patch"*. DSE answers the bigger question: **"which configuration should we have
built?"** That reframes our deliverable from a linter into a design tool, and it
is the strongest possible answer to the "quantify the trade-off between expected
improvement and implementation cost" objective.

### Why it is only possible because of the surrogate

| | Time per configuration | Configs in 8 hours |
|---|---|---|
| Real PnR + PDNSim | ~20 min | ~24 |
| **PRISM surrogate** | ~50 ms | **~500,000** |

State this ratio on a slide. It is the cleanest possible justification for why a
learned model belongs in this flow at all.

### Three-tier strategy [P0 = tier 1+2, P2 = tier 3]

**Tier 1 — Screening (Sobol sampling).**
`scipy.stats.qmc.Sobol` over the full 13-dimensional knob space, 512–2048
samples, evaluated on the surrogate. Compute variance-based sensitivity to find
which knobs actually matter. Expected finding, worth predicting out loud before
you run it: *strap pitch and bump pitch dominate; ECC_LANES matters only through
its effect on local current density.*

**Tier 2 — Multi-objective optimisation (NSGA-II via `pymoo`).**

*Minimise:*
1. peak IR drop (mV)
2. IR-aware TNS (ns) — total negative slack **after** droop derating
3. area (µm²)
4. routing resource consumed (%)
5. total power (mW)

*Subject to:* peak IR ≤ 5% Vdd, WNS ≥ 0.

Output: a **Pareto front**, reported with **hypervolume** so we can quantify how
much better it is than a random or full-factorial search. Two or three named
points on that front ("cheapest compliant", "best performance", "balanced") go
straight onto a slide.

**Tier 3 — Bayesian optimisation for the expensive runs.**
We can afford maybe 12–16 *real* ORFS runs. Choosing them at random wastes them.
Use a GP with Expected Improvement (sklearn `GaussianProcessRegressor`, hand-rolled
EI — ~40 lines) to pick which configurations get a real PnR run, using the
surrogate's uncertainty band to seed the prior. Then **feed the real results back
to retrain the surrogate**.

That closed loop — cheap model proposes, expensive tool verifies, model
retrains — is a genuinely strong story and it directly mirrors how a real
methodology team works.

### Files
```
configs/            one JSON per configuration
dse/space.py        knob definitions and encode/decode
dse/screen.py       Sobol sampling + sensitivity
dse/nsga.py         pymoo problem definition + run
dse/bayesopt.py     GP + EI selection of real PnR runs
out/dse/pareto.csv  the front
```

---

## 9. Software to install

Do this **now**, before anything else. Everyone installs everything — a broken
environment at hour 30 is how teams lose.

### 9.1 · Base (Windows)
```powershell
wsl --install -d Ubuntu-22.04          # reboot when it asks
# then install Docker Desktop, and enable "Use WSL2 based engine" in settings
```
Everything below runs **inside WSL Ubuntu**, not PowerShell.

### 9.2 · Python
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git build-essential
python3 -m venv ~/prismenv && source ~/prismenv/bin/activate
pip install numpy scipy pandas scikit-learn matplotlib plotly streamlit \
            pymoo networkx pyyaml joblib
# optional, for the scalability demo:
pip install pyamg
```
We deliberately avoid xgboost/lightgbm — sklearn's `HistGradientBoostingRegressor`
is as good here and is one less install to fail.

### 9.3 · Open-source EDA
```bash
# Yosys, Verilator, Icarus, GTKWave in one download:
#   github.com/YosysHQ/oss-cad-suite-build  -> grab the latest linux-x64 release
# or:
sudo apt install -y yosys verilator iverilog gtkwave

# OpenROAD flow scripts (this is the important one)
git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd OpenROAD-flow-scripts
```
For OpenROAD itself, the supported paths are the repo's own
`etc/DockerHelper.sh` (Docker) or `sudo ./etc/DependencyInstaller.sh` followed by
`./build_openroad.sh --local`. There are prebuilt images under the `openroad`
org on Docker Hub — **check which tags actually exist rather than trusting a
pasted command**, they change.

Smoke test — do this before you trust anything:
```bash
source ./env.sh
cd flow && make DESIGN_CONFIG=./designs/nangate45/gcd/config.mk
make DESIGN_CONFIG=./designs/nangate45/gcd/config.mk gui_final   # optional
```
`gcd` on nangate45 takes a few minutes. If that works, the toolchain is real.

**What we need out of ORFS:** the routed DEF, the OpenSTA timing reports, the
synthesised netlist, and the **PDNSim / `analyze_power_grid`** voltage map. That
last one is our real ground truth.

### 9.4 · PDKs
Nangate45 and Sky130 ship inside ORFS. Nothing extra to install.

### 9.5 · Cadence Virtuoso
**Not on the critical path.** It is a custom/analog environment — schematic
capture, layout, Spectre. Neither our netlist-graph work nor the PDN mesh solve
has anything to run in it. *Optional* use: one small Spectre simulation showing a
single logic path slowing down under a 50 mV supply droop, as an illustrative
slide that grounds the alpha-power law in a real simulator. Nice to have, worth
30 minutes, worth zero if we're behind.

---

## 10. What to learn, by role

Time budgets assume you are starting from a standard 3rd-year ECE background.

| Topic | Who | Budget | What "knowing it" means |
|---|---|---|---|
| IR drop, PDN, static vs dynamic droop | everyone | 45 min | Can explain why droop is worse far from a bump and under a macro |
| Alpha-power law | B, C | 20 min | Can derive `d(ln delay)/d(ln V)` and plug in numbers |
| STA: slack, WNS, TNS, setup/hold | C, D | 45 min | Can read an OpenSTA report and say which endpoint is worst |
| Graph Laplacian / KCL as a linear system | B | 40 min | Can explain why `L·1 = 0` and why that removes the supply term |
| Gradient boosting, over/underfitting, holdout design | B | 60 min | Can explain why a random tile split is cheating |
| Quantile regression | B | 20 min | Can explain what a 10–90 band means |
| Knapsack / greedy / submodularity | C | 45 min | Can state the 1−1/e bound and when it applies |
| Multi-objective optimisation, Pareto, hypervolume, NSGA-II | C | 60 min | Can read a Pareto front plot and defend a chosen point |
| Bayesian optimisation, GP, Expected Improvement | C | 45 min | Can explain why we don't pick real PnR runs at random |
| Parameterised SystemVerilog, `generate`, packages | A | 60 min | Can add a knob without touching ten files |
| Yosys + ORFS flow stages | A | 90 min | Can run gcd end to end and find the IR map on disk |
| Streamlit | D | 30 min | Can build a page with a selectbox and a plotly figure |

---

## 11. Work distribution

Written for **four people**. Collapse guidance at the end.

### Roles

**A — RTL & Flow owner**
Parameterised RTL, Verilator lint, testbench + VCD for real activity, Yosys,
ORFS runs, `gen_config.py`, extracting DEF / reports / IR maps into our schema.
*Owns:* `rtl/`, `scripts/`, `flow/`, `orfs_ingest.py`.

**B — Physics & Prediction owner**
PDN solver, two-fidelity setup, feature extraction, hybrid model, quantile band,
validation table and ablation, the leakage audit.
*Owns:* `prism/design.py`, `prism/solver.py`, `prism/features.py`,
`prism/model.py`, `prism/audit.py`.

**C — Risk, Mitigation & DSE owner**
Volts→ps conversion, effective slack, rollups, scenario economics, mitigation
catalog, cost model, knapsack, Pareto, and all of §8.
*Owns:* `prism/timing.py`, `prism/mitigation.py`, `dse/`.

**D — Product owner**
Dashboard, figures, slides, technical report, the demo script, rehearsals,
integration, and being the one person who runs the whole pipeline clean every
few hours to catch breakage early.
*Owns:* `app.py`, `docs/`, `slides/`, `run_all.py`.

### The interfaces that let this run in parallel

Agree these **in the first 30 minutes** and do not change them without telling
everyone. Each is a plain CSV or npz on disk, so nobody imports anyone else's
half-finished code.

```
out/<design>/artefacts.npz   A → B    grids, placement, macros, bumps, strap map
out/<design>/paths.csv       A → C    endpoint, slack_ns, delay_ns, inst_idx list
out/features.csv             B        one row per (design, scenario, tile)
out/predictions.csv          B → C,D  + pred_v, lo_v, hi_v, label_v
out/risk_tiles.csv           C → D    + slack_loss_ps, risk, block, domain
out/risk_blocks.csv          C → D    block/domain/scenario rollup
out/mitigations.csv          C → D    action, tile, benefit_ps, cost, roi
out/pareto.csv               C → D    budget sweep + DSE front
```

**Everything is precomputed to disk. The dashboard never trains or solves live.**
That is how we avoid the demo dying on stage.

### Timeline (T+0 = when you start; ~35 h to deadline)

| Window | A | B | C | D |
|---|---|---|---|---|
| T+0 → T+2 | **Start ORFS build/pull in background.** Agree interfaces. Everyone installs. | | | |
| T+2 → T+8 | RTL skeleton + lint + first Yosys run | Physics engine, generator, labels | Read up (alpha-law, knapsack, pymoo) | Repo scaffold, plotting helpers |
| T+8 → T+13 | gcd smoke test → first real ORFS run | Features + model + **validation table** | Volts→ps + effective slack | Dashboard skeleton against fake data |
| T+13 → T+17 | Parameter sweep configs, batch ORFS | Ablation + audit + importance | Rollups + scenario economics | Slide skeleton, figure list |
| T+17 → T+22 | Ingest real ORFS results | Retrain on real+synthetic | Mitigation catalog + knapsack + Pareto | Integrate B and C outputs |
| T+22 → T+28 | **Sleep. All of you.** Stagger if someone must babysit a batch run. | | | |
| T+28 → T+31 | DSE batch runs | Telemetry module | NSGA-II + BO | Dashboard final, figures final |
| T+31 → T+33 | | | | Slides + report + **one full rehearsal** |
| T+33 → T+35 | Buffer. Do not start anything new. | | | |

### If you are fewer than four

* **Three:** merge D into whoever is furthest ahead at T+20. Product work is
  compressible; physics is not.
* **Two:** one person takes A+B (data and prediction), the other C+D (risk,
  mitigation, product). Cut DSE tier 3 and the telemetry module. Keep the
  volts→ps conversion — it is the differentiator and it is only ~150 lines.

### Priorities if you fall behind

Cut in this order — **bottom of the list goes first**:

1. [P0] Physics engine + two fidelities
2. [P0] Hybrid predictor + validation table with ablation
3. [P0] Volts→ps ranking + block/domain rollup
4. [P0] Mitigation catalog + ROI Pareto
5. [P0] Slides + one rehearsal
6. [P1] Dashboard
7. [P1] Real ORFS data
8. [P1] Parameterised RTL sweep
9. [P1] DSE tiers 1–2
10. [P2] Telemetry module
11. [P2] DSE tier 3 (Bayesian optimisation)
12. [P2] Virtuoso/Spectre illustration

---

## 12. Definition of done

The demo, in the order we will show it, in under four minutes:

1. **The setup** — "this grid was signed off on the read-stream scenario and it
   passes at 42 mV against a 45 mV budget."
2. **The reveal** — switch scenario to GC compaction. 65 mV. Budget blown.
   *This is the moment that lands.*
3. **The prediction** — our floorplan-time map next to the signoff map, with the
   top-5% hit rate on screen.
4. **The ranking** — the millivolt ranking and the effective-slack ranking side
   by side, showing they disagree. Point at a high-mV tile that doesn't matter
   and a low-mV tile that does.
5. **The fix** — pick the top-ranked block, show six mitigations with measured
   benefit, show the knapsack picking three under budget.
6. **The curve** — the Pareto front. "40% of the droop for 12% of the cost."
7. **The bigger answer** — the DSE front. "Or don't build that floorplan at all."
8. **The close** — the telemetry map. "And here is where to put the TDETs that
   verify all of this on real silicon."

---

## 13. Questions we will be asked, and the answers

**"Isn't this just PowerNet / an ML IR-drop predictor?"**
No. PowerNet predicts the map. We predict only what physics *can't*, then convert
it to slack, then optimise mitigations against a budget. The prediction is the
*input* to our contribution, not the contribution.

**"Your ground truth is your own solver — isn't that circular?"**
Get there first, before they ask. Yes, for the synthetic corpus, which is why we
also run real OpenROAD/PDNSim data, and why every assumption in the generator is
on a slide. The residual formulation is also what limits the damage: the model
only learns the *gap*, so the physics — which is not learned — carries the
generalisation.

**"Would this survive on a real 5 nm SoC?"**
The mesh abstraction and the residual formulation transfer; the trained weights
do not, and would need retraining per node from a company's tapeout history.
Say this plainly. Overclaiming here is how you lose a judge who does this for a
living.

**"Why should I trust a 50 ms prediction over a 20 minute signoff run?"**
You shouldn't, and we don't ask you to. It is a *screening* tool — it tells you
where to spend the 20-minute runs. That's the point of the uncertainty band.

**"What's your biggest limitation?"**
Static IR only. Real droop is dynamic — di/dt against package inductance, with
decap as a charge reservoir, and that needs a transient solve. Our decap model is
therefore the weakest link in the mitigation catalog. Name it before they do.
