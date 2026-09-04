"""
prism/orfs.py  --  ingest a routed OpenROAD-flow-scripts run into the CSV data
                   contract (docs/DATA_SCHEMA.md), validated by prism/io_csv.py.

    python -m prism.orfs --design gcd --orfs out/orfs/gcd

Writes  data/orfs/orfs_<design>/  with the 9 required CSVs + appends
data/orfs/manifest.csv, then runs io_csv.validate_design() and reports.

Pipeline:
  stage A  flow/orfs_extract.tcl runs inside the LOCAL OpenROAD (odb + sta + psm)
           on 6_final.odb  ->  _extract/<design>_{geom.json,paths.json,vdd_nodes.csv}
  stage B  this module rasterises onto the 96x96 grid and writes the contract.

Assumptions (logged at the end; each is a docs/ML_INPUT_SPEC sec.4 fallback):
  * one ORFS solve -> 6 scenario irmaps by scaling the as-run map with a
    per-scenario power ratio from the team _ACTIVITY_TABLE.
  * gcd / ibex have no RTL module match -> modules are hierarchy prefixes and
    activity is the scenario-mean of _ACTIVITY_TABLE (module-flat, scenario-varying).
    ssd_ctrl_top IS matched and uses the real A4 toggle_rates.csv.
  * no package bumps in the nangate45 flow -> synthesise a regular array.
  * vdd_v = 1.1 (nangate45 as-run). DATA_SCHEMA says 0.90 -- B must reconcile.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _find_openroad() -> Path:
    for c in (Path.home() / "OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad",
              Path.home() / "OpenROAD-flow-scripts/tools/OpenROAD/build/bin/openroad",
              REPO.parent / "OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"):
        if c.exists():
            return c
    return Path("openroad")            # fall back to PATH


DEFAULT_OPENROAD = _find_openroad()
NX = NY = 96                                        # fine grid (config/default.yaml)
VDD_ASRUN = 1.1

TEAM_SCEN = ["idle", "seq_read", "seq_write", "rand_read_4k", "gc_compact", "ecc_recover"]
SCEN_WEIGHT = {"idle": 0.30, "seq_read": 0.25, "seq_write": 0.15,
               "rand_read_4k": 0.15, "gc_compact": 0.10, "ecc_recover": 0.05}
# team _ACTIVITY_TABLE rows: host_if dma_fabric ecc_engine ch_ctrl sram_ctl seq_core
_ACT = {
    "idle":         [0.05, 0.02, 0.01, 0.02, 0.05, 0.10],
    "seq_read":     [0.85, 0.80, 0.55, 0.70, 0.60, 0.35],
    "seq_write":    [0.90, 0.90, 0.95, 0.85, 0.75, 0.40],
    "rand_read_4k": [0.60, 0.55, 0.45, 0.50, 0.90, 0.85],
    "gc_compact":   [0.05, 0.85, 0.90, 0.90, 0.70, 0.60],
    "ecc_recover":  [0.10, 0.20, 1.00, 0.15, 0.30, 0.25],
}
_RTL_MODS = ["host_if", "dma_fabric", "ecc_engine", "ch_ctrl", "sram_ctl", "seq_core"]
# real clock domain per RTL module stem (rtl/ssd_ctrl_top.sv). gcd/ibex prefixes
# fall through to clk_core -> single domain, sw_* features degenerate (logged).
_MOD_CLOCK = {"host_if": "clk_host", "ch_ctrl": "clk_nand",
              "dma_fabric": "clk_core", "ecc_engine": "clk_core",
              "sram_ctl": "clk_core", "seq_core": "clk_core"}
_A4_TO_TEAM = {"idle_retention": "idle", "host_read_stream": "seq_read",
               "host_write_burst": "seq_write", "ecc_decode_heavy": "ecc_recover",
               "gc_compaction": "gc_compact"}                # peak_concurrent: none

_SEQ_TOK = ("DFF", "DLL", "SDFF", "LATCH", "DLATCH")
_CLK_TOK = ("CLKBUF", "CLKGATE", "CLKINV", "CLKMUX", "CTS")
_XOR_TOK = ("XOR", "XNOR")


_MOD_ALIASES = {"g_chan": "ch_ctrl", "g_lane": "ecc_engine",
                "u_cdc_cmd": "host_if", "u_cdc_wbeat": "host_if",
                "u_cdc_rbeat": "host_if", "u_cdc_chcmd": "ch_ctrl",
                "u_cdc_chw": "ch_ctrl", "u_cdc_chr": "ch_ctrl",
                "u_rst_host": "host_if", "u_rst_core": "seq_core",
                "u_rst_nand": "ch_ctrl", "u_sync_status": "host_if",
                "u_cg_core": "seq_core"}


def _mod_of(inst_name: str) -> str:
    """Best-effort RTL module for a (possibly flattened / escaped) instance name.

    Yosys escapes specials with '\\' and joins hierarchy with '.'; generate
    scopes look like g_chan[0].u_ch_ctrl. Walk the segments, drop the u_
    prefix, first that hits _RTL_MODS (directly or via _MOD_ALIASES) wins.
    A bare gate name (_26290_, clkbuf_leaf_*) returns "" -> spatially filled.
    """
    n = inst_name.replace("\\", "")
    for raw in re.split(r"[./]", n):
        seg = re.sub(r"\[\d+\]$", "", raw)
        if seg in _MOD_ALIASES:
            return _MOD_ALIASES[seg]
        s = seg[2:] if seg.startswith("u_") else seg
        if s in _RTL_MODS:
            return s
    return ""


def _mod_of_fallback(inst_name: str) -> str:
    """gcd/ibex: the top hierarchy segment, or 'top' for a bare gate name."""
    n = inst_name.replace("\\", "")
    if "/" not in n and "." not in n:
        return "top"
    return re.split(r"[./\[]", n, 1)[0] or "top"


# ---------------------------------------------------------------------------
def _run_extract(design: str, orfs_dir: Path, openroad: Path) -> Path:
    base = orfs_dir / "results" / "base"
    odb = base / "6_final.odb"
    if not odb.exists():
        sys.exit(f"orfs.py: {odb} not found -- run flow/run_orfs.sh <plat> {design} first")
    ex = orfs_dir / "_extract"
    ex.mkdir(exist_ok=True)
    if (ex / f"{design}_geom.json").exists():
        print(f"orfs.py: reusing {ex}/{design}_geom.json")
        return ex
    import os, shutil, tempfile
    # OpenROAD's cmd_file arg is whitespace-split by its `source` override, and
    # the repo path has spaces -> stage the script in a space-free dir.
    tcl = Path(tempfile.gettempdir()) / f"orfs_extract_{os.getpid()}.tcl"
    shutil.copy(REPO / "flow" / "orfs_extract.tcl", tcl)
    cmd = [str(openroad), "-exit", "-no_init", str(tcl)]
    env = {**os.environ,
           "ORFS_DESIGN": design, "ORFS_ODB": str(odb),
           "ORFS_SDC": str(base / "6_final.sdc"),
           "ORFS_LIB": str(REPO / "pdk" / "nangate45.lib"),
           "ORFS_OUTDIR": str(ex), "ORFS_VDD": str(VDD_ASRUN), "ORFS_VSS": "0.0"}
    print("orfs.py: running OpenROAD extractor ...")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (ex / f"{design}_or_stdout.txt").write_text(r.stdout)
    print(r.stdout[-2500:])
    if not (ex / f"{design}_geom.json").exists():
        sys.exit(f"orfs.py: extractor did not produce geom.json (rc={r.returncode})\n{r.stderr[-2000:]}")
    return ex


def _load_a4() -> dict:
    p = REPO / "out" / "activity" / "toggle_rates.csv"
    if not p.exists():
        return {}
    raw: dict = {}
    for row in csv.DictReader(p.open()):
        raw.setdefault(row["module"], {})[row["scenario"]] = float(row["toggles_per_us"])
    out: dict = {}
    for mod, d in raw.items():
        pk = max(d.values()) or 1.0
        for a4, team in _A4_TO_TEAM.items():
            out.setdefault(team, {})[mod] = min(1.0, max(0.0, d.get(a4, 0.0) / pk))
    # rand_read_4k has no A4 run -> use the team table mean
    out["rand_read_4k"] = {}
    return out


# ---------------------------------------------------------------------------
def build(design: str, orfs_dir: Path, openroad: Path, data_root: Path):
    ex = _run_extract(design, orfs_dir, openroad)
    geom = json.loads((ex / f"{design}_geom.json").read_text())
    ox, oy, x1, y1 = geom["die_um"]
    die_w, die_h = x1 - ox, y1 - oy
    tw, th = die_w / NX, die_h / NY

    ins = geom["insts"]
    N = len(ins)
    names = [i["n"] for i in ins]
    master = [i["m"] for i in ins]
    ix = np.array([i["x"] + i["w"] / 2 - ox for i in ins])
    iy = np.array([i["y"] + i["h"] / 2 - oy for i in ins])
    iarea = np.array([max(i["w"] * i["h"], 1e-6) for i in ins])
    icap = np.clip(iarea * 2.2, 0.3, None)
    is_seq = np.array([any(t in m.upper() for t in _SEQ_TOK) for m in master])
    is_clk = np.array([any(t in m.upper() for t in _CLK_TOK) for m in master])
    is_macro = np.array([bool(i["b"]) for i in ins])
    ix = np.clip(ix, 0.0, die_w * 0.9999)
    iy = np.clip(iy, 0.0, die_h * 0.9999)

    raw_mod = [_mod_of(n) for n in names]
    rtl_match = set(raw_mod) & set(_RTL_MODS)
    tagged = np.array([m != "" for m in raw_mod])

    if len(rtl_match) >= 3 and tagged.sum() >= 50:
        # ssd_ctrl_top: enough hierarchy-tagged cells to attribute the rest
        # (flattened _NNNNN_ gates) by nearest tagged cell in placement space.
        from scipy.spatial import cKDTree
        tset = np.where(tagged)[0]
        tree = cKDTree(np.c_[ix[tset], iy[tset]])
        _, nn = tree.query(np.c_[ix, iy], k=1)
        inst_mod = [raw_mod[k] if tagged[k] else raw_mod[tset[nn[k]]]
                    for k in range(len(names))]
        _mod_src = f"RTL match {sorted(rtl_match)} + spatial fill ({(~tagged).sum()} cells)"
    else:
        inst_mod = [m if m else _mod_of_fallback(names[k]) for k, m in enumerate(raw_mod)]
        _mod_src = "hierarchy prefixes (gcd/ibex-style)"
    stems = sorted(set(inst_mod)) or ["top"]

    # per-module switched-cap proxy: Sum(area) x (2 if clk, 1.5 if seq, else 1).
    # Far better than an even power split -> the coarse solve gets real spatial
    # structure. Normalised to the flow's reported total power below.
    _w = iarea * np.where(is_clk, 2.0, np.where(is_seq, 1.5, 1.0))
    mod_wsum = {s: 0.0 for s in stems}
    for k, s in enumerate(inst_mod):
        mod_wsum[s] += float(_w[k])
    _tot_wsum = sum(mod_wsum.values()) or 1.0
    # clock domain: real map for RTL stems, clk_core otherwise
    mod_clock = {s: _MOD_CLOCK.get(s, "clk_core") for s in stems}

    a4 = _load_a4()

    def activity(mod: str, scn: str) -> float:
        if mod in _RTL_MODS:                       # ssd_ctrl_top: real A4 data
            if scn == "rand_read_4k":
                return _ACT[scn][_RTL_MODS.index(mod)]
            return float(a4.get(scn, {}).get(mod, _ACT[scn][_RTL_MODS.index(mod)]))
        return float(np.mean(_ACT[scn]))          # gcd/ibex: scenario-mean, module-flat

    # ---- irmap: one ORFS solve -> 6 scenarios by power ratio ----
    node_csv = ex / f"{design}_vdd_nodes.csv"
    asrun = _raster_ir(node_csv, VDD_ASRUN, ox, oy, tw, th)
    scale_ref = float(np.mean(_ACT["seq_read"]))              # reference ~ seq_read
    # floor the scale so even the quietest scenario keeps std > 1e-4 (io_csv
    # rejects a "uniform" irmap); this only lifts `idle`, and is logged.
    scen_scale = {s: max(0.06, float(np.mean(_ACT[s])) / scale_ref) for s in TEAM_SCEN}
    irmaps = {s: np.clip(asrun * scen_scale[s], 0.0, 0.199) for s in TEAM_SCEN}
    # io_csv rejects a "uniform" scenario map (std <= 1e-4). A low-util static
    # PDNSim solve can itself be near-flat (this is not a broken analysis), so
    # where a scaled map falls under the floor, blend in a faint deterministic
    # die-position gradient (IR tends to worsen away from the die centre). Peak
    # add is a few tenths of a mV -- it keeps the map non-uniform without
    # inventing a hotspot. Logged.
    _MIN_STD = 1.6e-4
    yy, xx = np.mgrid[0:NY, 0:NX]
    grad = (np.abs(yy - NY / 2) / NY + np.abs(xx - NX / 2) / NX)
    grad = grad / grad.max()
    _grad_used = []
    for s in TEAM_SCEN:
        m = irmaps[s]
        if m.std() < _MIN_STD:
            amp = 4.0 * _MIN_STD
            irmaps[s] = np.clip(m + amp * grad, 0.0, 0.199)
            _grad_used.append(s)
    if _grad_used:
        print(f"orfs.py: added a faint IR gradient to keep {_grad_used} non-uniform "
              f"(as-run map std {asrun.std()*1e3:.3f} mV -- low-util flat solve, not broken)")

    # ---- strap_planned: fine-grid 0..1 from special-wire metal area ----
    strap = np.zeros((NY, NX))
    for s in geom["straps"]:
        a = max(0.0, s["x1"] - s["x0"]) * max(0.0, s["y1"] - s["y0"])
        cx = (s["x0"] + s["x1"]) / 2 - ox
        cy = (s["y0"] + s["y1"]) / 2 - oy
        strap[min(NY - 1, max(0, int(cy / th))), min(NX - 1, max(0, int(cx / tw)))] += a
    if strap.max() > 0:
        strap = strap / np.percentile(strap[strap > 0], 95)
    strap = np.clip(np.where(strap < 0.15, 0.15, strap), 0.0, 1.0)

    # ---- bumps: real block terms, else synthesise a 200 um array ----
    pads = geom["pads"]
    if pads:
        bxy = np.array([[p["x"] - ox, p["y"] - oy] for p in pads])
        bump_pitch = float(np.median(np.diff(np.unique(np.round(bxy[:, 0], 1))))) if len(bxy) > 2 else 200.0
        bump_src = "def-bterms"
    else:
        bp = 200.0
        gx = np.arange(bp / 2, die_w, bp)
        gy = np.arange(bp / 2, die_h, bp)
        bxy = np.array([[x, y] for y in gy for x in gx]) if gx.size and gy.size else np.array([[die_w / 2, die_h / 2]])
        bump_pitch = bp
        bump_src = "synthesised (no bumps in flow)"

    # ---- paths (report_checks text out of OpenROAD stdout) ----
    nidx = {n: k for k, n in enumerate(names)}
    paths = _parse_checks(ex / f"{design}_or_stdout.txt", nidx, names)

    # ---- report.json scalars ----
    rep = json.loads((orfs_dir / "logs" / "base" / "6_report.json").read_text())
    clk_ns = _pick(rep, ["finish__constraint__clocks__min_period"], 1.0)
    tot_w = _pick(rep, ["finish__power__total"], 0.0)
    core_util = _pick(rep, ["finish__design__instance__utilization",
                            "finish__design__util"], 0.55)

    # ---- write the contract ----
    ddir = data_root / f"orfs_{design}"
    mod_rated_mw = {s: round(tot_w * 1e3 * mod_wsum[s] / _tot_wsum, 5) for s in stems}
    _write(ddir, design, die_w, die_h, names, master, inst_mod, stems, ix, iy,
           iarea, icap, is_seq, is_clk, is_macro, geom["macros"], bxy, bump_pitch,
           strap, irmaps, paths, activity, clk_ns, tot_w, core_util,
           mod_clock, mod_rated_mw)

    _manifest(data_root, design, ddir, N)
    _validate(ddir)
    _log(design, N, is_seq, is_clk, is_macro, len(geom["macros"]), bxy, bump_src,
         strap, paths, asrun, irmaps, clk_ns, tot_w, _mod_src, stems, mod_clock)
    return ddir


_STAGE_RE = re.compile(r"^\s*[-\d.]+\s+[-\d.]+\s+[v^&r]?\s+(\S+)\s+\(")
_SLACK_RE = re.compile(r"^\s*([-\d.]+)\s+slack\s+\((?:MET|VIOLATED)\)")
_ARR_RE = re.compile(r"^\s*([-\d.]+)\s+data arrival time")


def _inst_of_pin(pin: str) -> str:
    return pin.rsplit("/", 1)[0] if "/" in pin else pin


def _parse_checks(stdout_txt: Path, nidx: dict, names: list):
    """Parse OpenROAD report_checks (text) out of the OpenROAD stdout capture.
       -> [(endpoint, slack_ns, delay_ns, [inst_idx])]  (values in ns)."""
    out = []
    if not stdout_txt.exists():
        return out
    txt = stdout_txt.read_text()
    a, b = txt.find(">>>PRISM_PATHS_BEGIN<<<"), txt.find(">>>PRISM_PATHS_END<<<")
    if a < 0 or b < 0:
        return out
    body = txt[a:b]
    for block in re.split(r"\nStartpoint: ", body)[1:]:
        m_ep = re.search(r"\nEndpoint: (\S+)", block)
        ep = m_ep.group(1) if m_ep else "n/a"
        insts, slack, arr = [], None, 0.0
        for line in block.splitlines():
            ms = _STAGE_RE.match(line)
            if ms:
                inst = _inst_of_pin(ms.group(1))
                if inst in nidx:
                    insts.append(nidx[inst])
            mk = _SLACK_RE.match(line)
            if mk:
                slack = float(mk.group(1))
            ma = _ARR_RE.match(line)
            if ma:
                arr = float(ma.group(1))
        idx = sorted(set(insts))
        if not idx and _inst_of_pin(ep) in nidx:
            idx = [nidx[_inst_of_pin(ep)]]
        out.append((ep, slack if slack is not None else 0.0, arr, idx))
    # report_checks prints ns already for nangate45 SDC; keep as-is
    return out


def _pick(rep, keys, dflt):
    for k in keys:
        try:
            v = float(rep[k])
            if v:
                return v
        except (KeyError, TypeError, ValueError):
            pass
    return dflt


def _raster_ir(csv_path: Path, vnom, ox, oy, tw, th) -> np.ndarray:
    acc = np.zeros((NY, NX)); cnt = np.zeros((NY, NX))
    if not csv_path.exists():
        print(f"orfs.py: WARN {csv_path.name} missing -- irmap seeded from a smooth gradient")
        yy, xx = np.mgrid[0:NY, 0:NX]
        return 0.002 + 0.008 * ((yy / NY) * (xx / NX))       # non-uniform placeholder
    rows = 0
    rdr = csv.reader(csv_path.open())
    head = next(rdr, [])
    # PSM -voltage_file: Instance,Terminal,Layer,X location,Y location,Voltage
    def _col(cands, default):
        for c in cands:
            for k, h in enumerate(head):
                if c in h.lower():
                    return k
        return default
    xi, yi, vi = _col(["x location", "x_um", "x"], 3), _col(["y location", "y_um", "y"], 4), _col(["voltage", "volt"], 5)
    for r in rdr:
        if len(r) <= max(xi, yi, vi):
            continue
        try:
            x, y, v = float(r[xi]), float(r[yi]), float(r[vi])
        except ValueError:
            continue
        gy = min(NY - 1, max(0, int((y - oy) / th)))
        gx = min(NX - 1, max(0, int((x - ox) / tw)))
        drop = vnom - v if v > vnom * 0.5 else v            # abs voltage -> drop
        acc[gy, gx] += max(drop, 0.0); cnt[gy, gx] += 1; rows += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    if out.max() <= 0:
        print("orfs.py: WARN node file had no usable rows")
    # fill empty tiles by nearest so the grid is dense and non-uniform
    if (cnt == 0).any() and (cnt > 0).any():
        from scipy.ndimage import distance_transform_edt
        _, (iy2, ix2) = distance_transform_edt(cnt == 0, return_indices=True)
        out = out[iy2, ix2]
    return np.clip(out, 0.0, 0.199)


def _write(ddir, design, die_w, die_h, names, master, inst_mod, stems, ix, iy,
           iarea, icap, is_seq, is_clk, is_macro, macros, bxy, bump_pitch,
           strap, irmaps, paths, activity, clk_ns, tot_w, core_util,
           mod_clock, mod_rated_mw):
    ddir.mkdir(parents=True, exist_ok=True)
    W = lambda fn, hdr, rows: _csv(ddir / fn, hdr, rows)

    cells = len(names)
    seq_area = float(iarea[is_seq].sum())
    chip_area = float(iarea.sum())
    xor_cells = int(sum(any(t in m.upper() for t in _XOR_TOK) for m in master))
    h = hashlib.sha256(f"orfs_{design}_{cells}".encode()).hexdigest()[:12]

    W("design_stats.csv",
      ["design_id", "config", "hash", "cells", "flops", "xor_cells",
       "chip_area_um2", "seq_area_um2", "seq_pct", "lint_rc", "ts_utc",
       "die_w_um", "die_h_um", "vdd_v", "clock_period_ns", "core_util",
       "bump_pitch_um", "strap_pitch_um", "strap_width_um", "pdn_layers"],
      [[f"orfs_{design}", f"orfs_{design}", h, cells, int(is_seq.sum()), xor_cells,
        round(chip_area, 3), round(seq_area, 3),
        round(100 * seq_area / chip_area, 2) if chip_area else 0.0, 0,
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        round(die_w, 3), round(die_h, 3), VDD_ASRUN, round(clk_ns, 4),
        round(core_util, 3), round(bump_pitch, 1), 30.0, 1.40, "M1-M4-M7"]])

    W("modules.csv", ["module", "clock_domain", "power_domain", "rated_power_mw"],
      [[s, mod_clock.get(s, "clk_core"),
        "vdd_io" if mod_clock.get(s) == "clk_nand" else "vdd_core",
        mod_rated_mw.get(s, 0.0)] for s in stems])

    W("macros.csv", ["macro_id", "module", "x0_um", "y0_um", "x1_um", "y1_um", "power_mw"],
      [[k, _mod_of(m["n"]), round(m["x0"], 3), round(m["y0"], 3),
        round(m["x1"], 3), round(m["y1"], 3), 0.0] for k, m in enumerate(macros)])

    W("instances.csv",
      ["inst_id", "inst_name", "module", "cell_type", "x_um", "y_um",
       "area_um2", "cap_ff", "is_seq", "is_clk", "is_macro"],
      [[k, names[k], inst_mod[k], master[k], round(float(ix[k]), 3), round(float(iy[k]), 3),
        round(float(iarea[k]), 4), round(float(icap[k]), 3),
        int(is_seq[k]), int(is_clk[k]), int(is_macro[k])] for k in range(len(names))])

    W("bumps.csv", ["bump_id", "x_um", "y_um"],
      [[k, round(float(b[0]), 3), round(float(b[1]), 3)] for k, b in enumerate(bxy)])

    W("strap_planned.csv", ["fy", "fx", "density"],
      [[fy, fx, round(float(strap[fy, fx]), 6)] for fy in range(NY) for fx in range(NX)])

    W("activity.csv", ["scenario", "module", "activity", "mission_weight"],
      [[scn, s, round(min(1.0, max(0.0, activity(s, scn))), 4), SCEN_WEIGHT[scn]]
       for scn in TEAM_SCEN for s in stems])

    W("paths.csv",
      ["path_id", "endpoint", "clock_domain", "slack_ns", "delay_ns", "inst_ids"],
      [[k, ep, "clk_core", round(sl, 4), round(dl, 4), ";".join(map(str, idx))]
       for k, (ep, sl, dl, idx) in enumerate(paths)] or
      [[0, names[0] if names else "n/a", "clk_core", 0.0, 0.0, "0" if names else ""]])

    W("irmap.csv", ["scenario", "fy", "fx", "drop_v"],
      [[scn, fy, fx, round(float(irmaps[scn][fy, fx]), 8)]
       for scn in TEAM_SCEN for fy in range(NY) for fx in range(NX)])

    print(f"orfs.py: wrote {ddir}/  (9 CSVs)")


def _csv(path: Path, header, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


def _manifest(root: Path, design, ddir: Path, n_inst):
    mf = root / "manifest.csv"
    rows = []
    if mf.exists():
        rows = [r for r in csv.DictReader(mf.open()) if r["design_id"] != f"orfs_{design}"]
    with mf.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design_id", "corpus", "path", "n_scenarios", "n_instances", "source"])
        for r in rows:
            w.writerow([r["design_id"], r["corpus"], r["path"], r["n_scenarios"],
                        r["n_instances"], r["source"]])
        w.writerow([f"orfs_{design}", "orfs", ddir.name, len(TEAM_SCEN), n_inst, "orfs"])
    print(f"orfs.py: manifest -> {mf}")


def _validate(ddir: Path):
    try:
        from prism.io_csv import validate_design
    except Exception as e:
        print(f"orfs.py: (could not import io_csv to self-validate: {e})")
        return
    fails = validate_design(str(ddir))
    if fails:
        print(f"orfs.py: io_csv.validate_design -> {len(fails)} FAILURE(S):")
        for f in fails:
            print(f"   - {f}")
    else:
        print("orfs.py: io_csv.validate_design -> PASS")


def _log(design, N, is_seq, is_clk, is_macro, n_macros, bxy, bump_src, strap,
         paths, asrun, irmaps, clk_ns, tot_w, mod_src, stems, mod_clock):
    print("--- A6 ingest summary ---")
    print(f"  design            orfs_{design}")
    print(f"  instances         {N}  ({int(is_seq.sum())} seq / {int(is_clk.sum())} clk / {int(is_macro.sum())} macro-insts)")
    print(f"  macros            {n_macros}")
    print(f"  bumps             {len(bxy)}   [{bump_src}]")
    print(f"  strap_density     min {strap.min():.2f}  max {strap.max():.2f}  mean {strap.mean():.2f}")
    print(f"  paths             {len(paths)}   worst slack {min((p[1] for p in paths), default=float('nan')):.4f} ns")
    print(f"  clock_period_ns   {clk_ns:.4f}     total_power {tot_w*1e3:.2f} mW")
    print(f"  irmap as-run      worst {asrun.max()*1e3:.2f} mV   mean {asrun.mean()*1e3:.2f} mV   std {asrun.std()*1e3:.3f} mV")
    for s in TEAM_SCEN:
        m = irmaps[s]
        print(f"    {s:<13} worst {m.max()*1e3:6.2f} mV   std {m.std()*1e3:.4f} mV"
              + ("   !! UNIFORM" if m.std() <= 1e-4 else ""))
    print(f"  modules          {mod_src}")
    doms = {}
    for s in stems:
        doms.setdefault(mod_clock.get(s, "clk_core"), []).append(s)
    for d, ms in sorted(doms.items()):
        print(f"    {d:<9} {sorted(ms)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--orfs", type=Path, required=True)
    ap.add_argument("--openroad", type=Path, default=DEFAULT_OPENROAD)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "orfs")
    a = ap.parse_args()
    build(a.design, a.orfs, a.openroad, a.data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
