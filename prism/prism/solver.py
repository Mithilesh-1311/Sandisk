"""solver.py — PDN mesh solver: the physics engine.

Builds and factorises A = L + D_bump on an ny x nx resistive grid.
This class signature is a FROZEN CONTRACT with role C — do not change it.

Properties preserved:
  1. U is exactly linear in I
  2. Scaling all conductances by s scales U by 1/s
  3. Factorise A once; every scenario is a back-substitution
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class PDNSolver:
    """Resistive-mesh IR-drop solver with cached LU factorisation.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (rows, columns).
    sheet_cond : float
        Sheet conductance between adjacent nodes [S].
    bump_cond : float
        Conductance from each bump node to the ideal supply [S].
    bump_mask : np.ndarray, shape (ny, nx), dtype bool
        True where a power bump exists.
    strap_density : np.ndarray or None, shape (ny, nx), float in [0,1]
        Local scaling of edge conductance.  None → uniform 1.0.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        sheet_cond: float,
        bump_cond: float,
        bump_mask: np.ndarray,
        strap_density: Optional[np.ndarray] = None,
    ):
        self.ny = ny
        self.nx = nx
        self.sheet_cond = sheet_cond
        self.bump_cond = bump_cond
        self.bump_mask = np.asarray(bump_mask, dtype=bool)
        self.strap_density = (
            np.asarray(strap_density, dtype=np.float64)
            if strap_density is not None
            else np.ones((ny, nx), dtype=np.float64)
        )
        self._n = ny * nx
        self._factor = None  # cached splu

        # Guard: at least one bump required or A is singular
        if self.bump_mask.sum() == 0:
            raise ValueError(
                "bump_mask has no True entries — A is singular.  "
                "At least one power bump is required."
            )

        # Build A and edge list
        self.edges: List[Tuple[int, int, float]] = []
        self.A: sp.csc_matrix = self._build_A()

    # ------------------------------------------------------------------
    # Node indexing: row-major  node_id = iy * nx + ix
    # ------------------------------------------------------------------

    def _idx(self, iy: int, ix: int) -> int:
        return iy * self.nx + ix

    # ------------------------------------------------------------------
    # Build A = L + D_bump
    # ------------------------------------------------------------------

    def _build_A(self) -> sp.csc_matrix:
        """Construct the system matrix A = L + D_bump."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        diag = np.zeros(self._n, dtype=np.float64)
        self.edges = []

        sd = self.strap_density

        # Horizontal edges
        for iy in range(self.ny):
            for ix in range(self.nx - 1):
                n0 = self._idx(iy, ix)
                n1 = self._idx(iy, ix + 1)
                g = self.sheet_cond * 0.5 * (sd[iy, ix] + sd[iy, ix + 1])
                rows.append(n0); cols.append(n1); vals.append(-g)
                rows.append(n1); cols.append(n0); vals.append(-g)
                diag[n0] += g
                diag[n1] += g
                self.edges.append((n0, n1, g))

        # Vertical edges
        for iy in range(self.ny - 1):
            for ix in range(self.nx):
                n0 = self._idx(iy, ix)
                n1 = self._idx(iy + 1, ix)
                g = self.sheet_cond * 0.5 * (sd[iy, ix] + sd[iy + 1, ix])
                rows.append(n0); cols.append(n1); vals.append(-g)
                rows.append(n1); cols.append(n0); vals.append(-g)
                diag[n0] += g
                diag[n1] += g
                self.edges.append((n0, n1, g))

        # Diagonal: bump conductance
        for iy in range(self.ny):
            for ix in range(self.nx):
                if self.bump_mask[iy, ix]:
                    n = self._idx(iy, ix)
                    diag[n] += self.bump_cond

        # Add diagonal entries
        for n in range(self._n):
            rows.append(n); cols.append(n); vals.append(diag[n])

        A = sp.csc_matrix(
            (np.array(vals), (np.array(rows, dtype=np.int32),
                              np.array(cols, dtype=np.int32))),
            shape=(self._n, self._n),
        )
        self._factor = None  # invalidate cached factorisation
        return A

    # ------------------------------------------------------------------
    # Factorise
    # ------------------------------------------------------------------

    def factorise(self) -> None:
        """LU-factorise A using SuperLU.  Cached until A changes."""
        self._factor = spla.splu(self.A)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve A * U = rhs via cached LU back-substitution.

        Parameters
        ----------
        rhs : np.ndarray, shape (n,) or (ny, nx)
            Current vector [A].  Positive current = current drawn from supply.

        Returns
        -------
        np.ndarray, same shape as rhs
            IR drop per node [V].
        """
        flat = rhs.ravel()
        if flat.shape[0] != self._n:
            raise ValueError(
                f"rhs has {flat.shape[0]} elements, expected {self._n} "
                f"({self.ny}x{self.nx})"
            )
        if self._factor is None:
            self.factorise()
        u = self._factor.solve(flat)
        return u.reshape(rhs.shape)

    # ------------------------------------------------------------------
    # Calibrate — closed-form
    # ------------------------------------------------------------------

    def calibrate(self, I_ref: np.ndarray, target_drop_v: float) -> None:
        """Closed-form calibration.

        Since scaling all conductances by `s` scales U by `1/s`:
          current_max_drop = max(solve(I_ref))
          s = current_max_drop / target_drop_v
          new_cond = old_cond * s

        No iterative search.
        """
        u = self.solve(I_ref)
        current_max = np.max(u)
        if current_max <= 0:
            raise ValueError("Current solve produced non-positive drop — cannot calibrate")

        s = current_max / target_drop_v
        self.sheet_cond *= s
        self.bump_cond *= s
        self.A = self._build_A()
        self.factorise()


# ---------------------------------------------------------------------------
# Standalone physics validation (run with: python -m prism.solver)
# ---------------------------------------------------------------------------

def _run_physics_validation() -> None:
    """Run the five analytic assertions from §6.S1 and print a PASS/FAIL table."""
    import time as _time

    N = 64
    results: list[tuple[str, bool, str]] = []

    # --- Build: uniform grid, single central bump, uniform current ---
    bump_mask = np.zeros((N, N), dtype=bool)
    bump_mask[N // 2, N // 2] = True

    solver = PDNSolver(N, N, sheet_cond=1.0, bump_cond=10.0, bump_mask=bump_mask)

    I_uniform = np.ones((N, N)) * 1e-3  # 1 mA per node

    # TEST 1: Monotonic radial profile
    U = solver.solve(I_uniform)
    cy, cx = N // 2, N // 2
    # Sample drop along a row from center outward
    center_drop = U[cy, cx]
    monotonic = True
    for dx in range(1, N // 2):
        if U[cy, cx + dx] < U[cy, cx + dx - 1] - 1e-15:
            monotonic = False
            break
    # Radial symmetry: compare 4 cardinal directions at distance 10
    d = 10
    drops = [U[cy, cx + d], U[cy, cx - d], U[cy + d, cx], U[cy - d, cx]]
    mean_d = np.mean(drops)
    max_dev = max(abs(v - mean_d) for v in drops) / mean_d if mean_d > 0 else 0
    sym_pass = max_dev < 0.02
    t1_pass = monotonic and sym_pass
    results.append((
        "1. Monotonic + radial symmetric (2%)",
        t1_pass,
        f"monotonic={monotonic}, max_radial_dev={max_dev:.6e}"
    ))

    # TEST 2: Linearity: solve(2*I) == 2*solve(I)
    U1 = solver.solve(I_uniform)
    U2 = solver.solve(2.0 * I_uniform)
    lin_err = np.max(np.abs(U2 - 2.0 * U1))
    t2_pass = lin_err < 1e-10
    results.append((
        "2. Linearity: solve(2I) == 2*solve(I)",
        t2_pass,
        f"max|err| = {lin_err:.2e}"
    ))

    # TEST 3: Conductance scaling: doubling cond halves U
    solver2 = PDNSolver(N, N, sheet_cond=2.0, bump_cond=20.0, bump_mask=bump_mask)
    U_double = solver2.solve(I_uniform)
    scale_err = np.max(np.abs(U_double - 0.5 * U1))
    t3_pass = scale_err < 1e-10
    results.append((
        "3. Conductance scaling: 2x cond => 0.5x U",
        t3_pass,
        f"max|err| = {scale_err:.2e}"
    ))

    # TEST 4: Superposition: solve(Ia + Ib) == solve(Ia) + solve(Ib)
    rng = np.random.RandomState(42)
    Ia = rng.rand(N, N) * 1e-3
    Ib = rng.rand(N, N) * 1e-3
    U_a = solver.solve(Ia)
    U_b = solver.solve(Ib)
    U_ab = solver.solve(Ia + Ib)
    sup_err = np.max(np.abs(U_ab - (U_a + U_b)))
    t4_pass = sup_err < 1e-10
    results.append((
        "4. Superposition: solve(Ia+Ib) == solve(Ia)+solve(Ib)",
        t4_pass,
        f"max|err| = {sup_err:.2e}"
    ))

    # TEST 5: Timing on 96x96
    N96 = 96
    bump96 = np.zeros((N96, N96), dtype=bool)
    bump96[N96 // 2, N96 // 2] = True
    bump96[0, 0] = True
    bump96[0, N96 - 1] = True
    bump96[N96 - 1, 0] = True
    bump96[N96 - 1, N96 - 1] = True

    solver96 = PDNSolver(N96, N96, sheet_cond=1.0, bump_cond=10.0, bump_mask=bump96)

    t0 = _time.perf_counter()
    solver96.factorise()
    t_fac = _time.perf_counter() - t0

    I96 = np.ones((N96, N96)) * 1e-3
    n_solves = 100
    t0 = _time.perf_counter()
    for _ in range(n_solves):
        solver96.solve(I96)
    t_solve = (_time.perf_counter() - t0) / n_solves

    t5_pass = t_solve < 0.005  # < 5 ms
    results.append((
        "5. Timing: per-solve < 5 ms (96x96)",
        t5_pass,
        f"factorise={t_fac*1000:.1f}ms, per-solve={t_solve*1000:.3f}ms"
    ))

    # --- Print table ---
    print()
    print(f"{'Test':<55} {'Result':<8} {'Detail'}")
    print("-" * 110)
    for name, passed, detail in results:
        tag = "PASS" if passed else "FAIL"
        print(f"{name:<55} {tag:<8} {detail}")
    print("-" * 110)

    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} passed")
    if n_pass < n_total:
        raise SystemExit(f"SOLVER VALIDATION FAILED: {n_total - n_pass} assertion(s) failed")


if __name__ == "__main__":
    _run_physics_validation()


# ---------------------------------------------------------------------------
# Floorplan-time coarse solver — built from PUBLISHED inputs only.
#
# This is the "early estimate" side of the two-fidelity split.  It reads only
# design_stats.csv, bumps.csv and strap_planned.csv.  It never touches the
# as-built strap map, the fine solve or the IR map, so it is safe to call from
# feature extraction under the audit.py leakage trap.
# ---------------------------------------------------------------------------

def design_conductances(stats_row, k_sheet: float, k_bump: float) -> Tuple[float, float]:
    """Physical mesh constants from a design's PUBLISHED PDN geometry.

    The strap ratio r = strap_width_um / strap_pitch_um is the fraction of each
    track pitch that is actually metal.  For a square tile the number of straps
    crossing it is tile/pitch and each carries width `strap_width`, so the
    conductance per tile edge is proportional to r and — crucially — independent
    of tile size.  That scale-invariance is what lets the coarse and fine meshes
    share one pair of constants.

    k_sheet, k_bump are the two technology constants.  On synthetic data they
    are calibrated once over the corpus (S2); on real data they are fitted so
    the mesh reproduces the signoff map (S8).
    """
    r = float(stats_row["strap_width_um"]) / float(stats_row["strap_pitch_um"])
    return k_sheet * r, k_bump * r


def coarse_solver_from_design(design, cfg: dict, k_sheet: float, k_bump: float) -> "PDNSolver":
    """Build the coarse (floorplan-stage) solver for a loaded Design.

    Uses the PLANNED strap map — the as-built map is unknown at this stage and
    is exactly what the residual model has to learn.
    """
    ny_c = cfg["grid"]["ny_coarse"]
    nx_c = cfg["grid"]["nx_coarse"]
    ny_f = cfg["grid"]["ny_fine"]
    nx_f = cfg["grid"]["nx_fine"]
    ratio = ny_f // ny_c

    stats = design.stats.iloc[0]
    die_w = float(stats["die_w_um"])
    die_h = float(stats["die_h_um"])
    sheet_cond, bump_cond = design_conductances(stats, k_sheet, k_bump)

    # Bump mask: a coarse tile is supplied once per bump that lands in it, so
    # multiplicity is accumulated rather than collapsed to a boolean.
    ctw, cth = die_w / nx_c, die_h / ny_c
    bx = np.clip((design.bumps["x_um"].values / ctw).astype(int), 0, nx_c - 1)
    by = np.clip((design.bumps["y_um"].values / cth).astype(int), 0, ny_c - 1)
    mult = np.zeros((ny_c, nx_c), dtype=np.float64)
    np.add.at(mult, (by, bx), 1.0)
    bump_mask = mult > 0

    # Planned strap density, averaged fine -> coarse
    sp = design.strap_planned
    planned_fine = np.zeros((ny_f, nx_f), dtype=np.float64)
    planned_fine[sp["fy"].values.astype(int), sp["fx"].values.astype(int)] = (
        sp["density"].values
    )
    planned_coarse = planned_fine.reshape(ny_c, ratio, nx_c, ratio).mean(axis=(1, 3))

    solver = PDNSolver(ny_c, nx_c, sheet_cond, bump_cond, bump_mask,
                       strap_density=planned_coarse)
    # Apply bump multiplicity: a tile holding m bumps has m times the
    # conductance to the supply.
    if mult.max() > 1:
        extra = (mult - 1.0).ravel() * bump_cond
        solver.A = (solver.A + sp.diags(np.maximum(extra, 0.0), format="csc")).tocsc()
        solver._factor = None
    return solver

