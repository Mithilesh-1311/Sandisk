"""test_gates.py — acceptance gates as pytest.

S0 gates:
  - Config loads and scenario weights sum to 1.0
  - Fine grid is an integer multiple of coarse grid
  - validate_design on a nonexistent directory returns failures
"""

import pathlib
import yaml
import pytest

# Resolve paths relative to the repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# S0 gates
# ---------------------------------------------------------------------------

class TestS0Config:
    """Configuration integrity checks."""

    def setup_method(self):
        with open(CONFIG_PATH, "r") as f:
            self.cfg = yaml.safe_load(f)

    def test_config_loads(self):
        """Config YAML parses without error."""
        assert self.cfg is not None
        assert "grid" in self.cfg
        assert "electrical" in self.cfg
        assert "scenarios" in self.cfg

    def test_scenario_weights_sum_to_one(self):
        """Scenario mission weights must sum to 1.0."""
        total = sum(s["weight"] for s in self.cfg["scenarios"].values())
        assert abs(total - 1.0) < 1e-9, f"Scenario weights sum to {total}, expected 1.0"

    def test_fine_grid_multiple_of_coarse(self):
        """Fine grid dimensions must be exact multiples of coarse."""
        g = self.cfg["grid"]
        assert g["nx_fine"] % g["nx_coarse"] == 0, (
            f"nx_fine={g['nx_fine']} not a multiple of nx_coarse={g['nx_coarse']}"
        )
        assert g["ny_fine"] % g["ny_coarse"] == 0, (
            f"ny_fine={g['ny_fine']} not a multiple of ny_coarse={g['ny_coarse']}"
        )

    def test_ir_budget_positive(self):
        """IR budget must be a positive fraction."""
        frac = self.cfg["electrical"]["ir_budget_frac"]
        vdd = self.cfg["electrical"]["vdd"]
        assert 0 < frac < 1
        assert vdd > 0

    def test_seeds_are_list(self):
        """Validation seeds must be a list of ints."""
        seeds = self.cfg["validation"]["seeds"]
        assert isinstance(seeds, list)
        assert len(seeds) == 5
        assert all(isinstance(s, int) for s in seeds)


class TestS0Validation:
    """io_csv.validate_design on nonexistent directory returns failures."""

    def test_nonexistent_directory_returns_failures(self):
        from prism.io_csv import validate_design
        failures = validate_design("data/real/nonexistent")
        assert len(failures) > 0, "validate_design must report failures for nonexistent dir"
        assert "does not exist" in failures[0].lower() or "not exist" in failures[0].lower()

    def test_nonexistent_returns_list(self):
        from prism.io_csv import validate_design
        result = validate_design("data/real/nonexistent")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# S1 gates — solver physics assertions
# ---------------------------------------------------------------------------

class TestS1Solver:
    """Five analytic assertions proving the solver is physically correct."""

    def setup_method(self):
        import numpy as np
        from prism.solver import PDNSolver

        self.np = np
        N = 64
        bump_mask = np.zeros((N, N), dtype=bool)
        bump_mask[N // 2, N // 2] = True
        self.N = N
        self.bump_mask = bump_mask
        self.solver = PDNSolver(N, N, sheet_cond=1.0, bump_cond=10.0,
                                bump_mask=bump_mask)
        self.I_uniform = np.ones((N, N)) * 1e-3

    def test_monotonic_radial_profile(self):
        """Drop monotonically increases with distance from bump, radially symmetric to 2%."""
        np = self.np
        N = self.N
        U = self.solver.solve(self.I_uniform)
        cy, cx = N // 2, N // 2

        # Monotonic along row from center outward
        for dx in range(1, N // 2):
            assert U[cy, cx + dx] >= U[cy, cx + dx - 1] - 1e-15, (
                f"Non-monotonic at offset {dx}"
            )

        # Radial symmetry at distance 10
        d = 10
        drops = [U[cy, cx + d], U[cy, cx - d], U[cy + d, cx], U[cy - d, cx]]
        mean_d = np.mean(drops)
        max_dev = max(abs(v - mean_d) for v in drops) / mean_d
        assert max_dev < 0.02, f"Radial deviation {max_dev:.6e} >= 2%"

    def test_linearity(self):
        """solve(3.7*I) == 3.7*solve(I) to within 1e-10."""
        np = self.np
        U1 = self.solver.solve(self.I_uniform)
        U2 = self.solver.solve(3.7 * self.I_uniform)
        err = np.max(np.abs(U2 - 3.7 * U1))
        assert err < 1e-10, f"Linearity error {err:.2e} >= 1e-10"

    def test_conductance_scaling(self):
        """Scaling all conductances by 3.7 scales U by 1/3.7 to within 1e-10."""
        np = self.np
        from prism.solver import PDNSolver
        solver2 = PDNSolver(self.N, self.N, sheet_cond=3.7, bump_cond=37.0,
                            bump_mask=self.bump_mask)
        U1 = self.solver.solve(self.I_uniform)
        U2 = solver2.solve(self.I_uniform)
        err = np.max(np.abs(U2 - U1 / 3.7))
        assert err < 1e-10, f"Scaling error {err:.2e} >= 1e-10"

    def test_superposition(self):
        """solve(Ia + Ib) == solve(Ia) + solve(Ib) to within 1e-10."""
        np = self.np
        rng = np.random.RandomState(42)
        Ia = rng.rand(self.N, self.N) * 1e-3
        Ib = rng.rand(self.N, self.N) * 1e-3
        U_a = self.solver.solve(Ia)
        U_b = self.solver.solve(Ib)
        U_ab = self.solver.solve(Ia + Ib)
        err = np.max(np.abs(U_ab - (U_a + U_b)))
        assert err < 1e-10, f"Superposition error {err:.2e} >= 1e-10"

    def test_timing_96x96(self):
        """Per-solve time on 96x96 must be < 5 ms."""
        import time
        np = self.np
        from prism.solver import PDNSolver

        N96 = 96
        bump96 = np.zeros((N96, N96), dtype=bool)
        bump96[N96 // 2, N96 // 2] = True
        bump96[0, 0] = True
        bump96[0, N96 - 1] = True
        bump96[N96 - 1, 0] = True
        bump96[N96 - 1, N96 - 1] = True

        solver96 = PDNSolver(N96, N96, sheet_cond=1.0, bump_cond=10.0,
                             bump_mask=bump96)
        solver96.factorise()

        I96 = np.ones((N96, N96)) * 1e-3
        n_solves = 100
        t0 = time.perf_counter()
        for _ in range(n_solves):
            solver96.solve(I96)
        t_solve = (time.perf_counter() - t0) / n_solves
        assert t_solve < 0.005, f"Per-solve {t_solve*1000:.3f}ms >= 5ms"


# ---------------------------------------------------------------------------
# S2 — synthetic corpus: the two fidelities must actually diverge
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pathlib.Path("data/synthetic/manifest.csv").exists(),
    reason="corpus not generated; run `python run_all.py gen`",
)
class TestS2Corpus:
    """Guards on the generated corpus.  These are the S2 gate, as pytest."""

    @staticmethod
    def _corpus():
        import numpy as np
        import pandas as pd
        from prism.io_csv import load_config, load_design
        from prism.design import _SCENARIOS, scenario_currents, load_calibration
        from prism.solver import coarse_solver_from_design

        cfg = load_config()
        k_sheet, k_bump = load_calibration()
        ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
        ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
        ratio = ny_f // ny_c

        manifest = pd.read_csv("data/synthetic/manifest.csv")
        labels, phys = [], []
        for _, row in manifest.iterrows():
            d = load_design(f"data/synthetic/{row['design_id']}")
            solver = coarse_solver_from_design(d, cfg, k_sheet, k_bump)
            for scn in _SCENARIOS:
                _, I_c = scenario_currents(d, scn, cfg)
                phys.append(solver.solve(I_c).ravel())
                s = d.irmap[d.irmap["scenario"] == scn]
                fine = np.zeros((ny_f, nx_f))
                fine[s["fy"].values.astype(int),
                     s["fx"].values.astype(int)] = s["drop_v"].values
                labels.append(
                    fine.reshape(ny_c, ratio, nx_c, ratio).max(axis=(1, 3)).ravel()
                )
        return np.concatenate(labels), np.concatenate(phys), manifest

    def test_manifest_row_count(self):
        import pandas as pd
        from prism.io_csv import load_config
        cfg = load_config()
        manifest = pd.read_csv("data/synthetic/manifest.csv")
        assert len(manifest) == cfg["corpus"]["n_designs"]

    def test_every_design_validates(self):
        import pandas as pd
        from prism.io_csv import validate_design
        manifest = pd.read_csv("data/synthetic/manifest.csv")
        for _, row in manifest.iterrows():
            failures = validate_design(f"data/synthetic/{row['design_id']}")
            assert not failures, f"{row['design_id']}: {failures}"

    def test_corpus_has_violations(self):
        """Nothing over budget means the demo has no story."""
        L, _, _ = self._corpus()
        assert L.max() > 0.045, f"corpus max label {L.max()*1000:.2f} mV <= 45 mV"

    def test_physics_baseline_is_biased_low(self):
        """The whole project rests on this gap being real and positive."""
        import numpy as np
        L, P, _ = self._corpus()
        bias_mv = float(np.mean(L - P)) * 1000
        assert 5.0 <= bias_mv <= 11.0, (
            f"bias {bias_mv:+.3f} mV outside [+5, +11]. Near zero means the two "
            f"fidelities have collapsed into one and there is nothing to learn."
        )

    def test_baseline_correlates_but_not_perfectly(self):
        from scipy.stats import pearsonr
        L, P, _ = self._corpus()
        r = float(pearsonr(P, L)[0])
        assert 0.90 <= r <= 0.98, f"pearson {r:.4f} outside [0.90, 0.98]"

    def test_no_irmap_is_uniform(self):
        import pandas as pd
        from prism.io_csv import load_design
        from prism.design import _SCENARIOS
        manifest = pd.read_csv("data/synthetic/manifest.csv")
        for _, row in manifest.iterrows():
            d = load_design(f"data/synthetic/{row['design_id']}")
            for scn in _SCENARIOS:
                std = d.irmap[d.irmap["scenario"] == scn]["drop_v"].std()
                assert std > 1e-4, f"{row['design_id']}/{scn} irmap is uniform"

    def test_round_trip_is_lossless(self):
        """Protects the real-data path: the CSV schema must lose nothing."""
        import numpy as np
        import pandas as pd
        from prism.io_csv import load_design
        manifest = pd.read_csv("data/synthetic/manifest.csv")
        for _, row in manifest.iterrows():
            ddir = f"data/synthetic/{row['design_id']}"
            d = load_design(ddir)
            raw = pd.read_csv(f"{ddir}/irmap.csv")
            err = float(np.max(np.abs(raw["drop_v"].values - d.irmap["drop_v"].values)))
            assert err < 1e-12, f"{row['design_id']}: irmap round-trip err {err:.2e}"

    def test_modules_are_spatially_clustered(self):
        """A floorplan, not a uniform scatter.  Without this the cur_/conc_/sw_
        feature families carry no signal."""
        import numpy as np
        from prism.io_csv import load_design
        d = load_design("data/synthetic/syn_000")
        nx_c = 24
        die_w = float(d.stats["die_w_um"].iloc[0])
        tx = np.clip((d.instances["x_um"].values / (die_w / nx_c)).astype(int),
                     0, nx_c - 1)
        uniform_std = np.sqrt((nx_c ** 2 - 1) / 12.0)
        mods = d.instances["module"].values
        spreads = [tx[mods == m].std() for m in np.unique(mods)]
        assert max(spreads) < 0.75 * uniform_std, (
            f"module placement looks uniform (max std {max(spreads):.2f} vs "
            f"uniform {uniform_std:.2f}) — there is no floorplan"
        )

    def test_clock_cells_exist(self):
        """top_clkden is a feature; is_clk all-zero makes it dead."""
        from prism.io_csv import load_design
        d = load_design("data/synthetic/syn_000")
        frac = float(d.instances["is_clk"].mean())
        assert 0.01 < frac < 0.25, f"is_clk fraction {frac:.4f} implausible"


# ---------------------------------------------------------------------------
# S3 — features and the leakage trap
# ---------------------------------------------------------------------------

class TestS3FeatureContract:
    """Contract tests that need no generated data."""

    def test_exactly_32_features(self):
        from prism.features import feature_columns
        assert len(feature_columns()) == 32

    def test_feature_names_are_unique(self):
        from prism.features import feature_columns
        cols = feature_columns()
        assert len(set(cols)) == len(cols)

    def test_group_sizes_match_spec(self):
        from prism.features import FEATURE_GROUPS
        expected = {"phys": 4, "grid": 6, "cur": 7, "conc": 3,
                    "top": 8, "sw": 2, "scn": 2}
        assert {k: len(v) for k, v in FEATURE_GROUPS.items()} == expected

    def test_banned_columns_are_absent(self):
        """§5B.5: a design-constant column lets the model fingerprint the design.
        `hash` in particular is a perfect fingerprint."""
        from prism.features import feature_columns, BANNED_COLUMNS
        cols = set(feature_columns())
        for banned in ["hash", "config", "design_id", "ts_utc", "lint_rc"]:
            assert banned in BANNED_COLUMNS, f"{banned} missing from BANNED_COLUMNS"
            assert banned not in cols, f"BANNED column {banned} is in feature_columns()"

    def test_identifier_columns_are_excluded(self):
        from prism.features import feature_columns, ID_COLUMNS
        cols = set(feature_columns())
        for ident in ID_COLUMNS:
            assert ident not in cols, f"identifier {ident} must not be a feature"


class TestS3LeakageTrap:
    """The trap must actually fire — an untested trap is decoration."""

    def test_irmap_read_raises(self):
        from prism import design as design_mod
        from prism.audit import LeakageError, leakage_trap
        from prism.io_csv import Design
        with leakage_trap():
            d = Design.__new__(Design)
            with pytest.raises(LeakageError, match="Design.irmap"):
                _ = d.irmap

    @pytest.mark.parametrize("fname", ["_build_layout", "_ground_truth_current",
                                       "_reference_drop"])
    def test_label_side_functions_raise(self, fname):
        from prism import design as design_mod
        from prism.audit import LeakageError, leakage_trap
        with leakage_trap():
            with pytest.raises(LeakageError, match=fname):
                getattr(design_mod, fname)()

    def test_trap_restores_state(self):
        from prism import design as design_mod
        from prism.audit import leakage_trap
        before = design_mod._build_layout
        with leakage_trap():
            pass
        assert design_mod._build_layout is before


@pytest.mark.skipif(
    not pathlib.Path("out/features.csv").exists(),
    reason="feature table not built; run `python run_all.py features`",
)
class TestS3FeatureTable:
    """The S3 gate, as pytest."""

    @staticmethod
    def _table():
        import pandas as pd
        return pd.read_csv("out/features.csv")

    def test_row_count(self):
        from prism.io_csv import load_config
        cfg = load_config()
        expected = (cfg["corpus"]["n_designs"] * len(cfg["scenarios"])
                    * cfg["grid"]["ny_coarse"] * cfg["grid"]["nx_coarse"])
        assert len(self._table()) == expected == 48384

    def test_no_nan_or_inf(self):
        """Any NaN is a hard fail — fix the feature, do not impute."""
        import numpy as np
        from prism.features import feature_columns
        t = self._table()
        X = t[feature_columns()].to_numpy(dtype=float)
        bad = ~np.isfinite(X)
        assert not bad.any(), (
            f"non-finite values in "
            f"{[feature_columns()[i] for i in np.unique(np.where(bad)[1])]}"
        )

    def test_all_32_columns_present(self):
        from prism.features import feature_columns
        t = self._table()
        missing = [c for c in feature_columns() if c not in t.columns]
        assert not missing, f"missing feature columns: {missing}"

    def test_no_banned_column_in_table(self):
        t = self._table()
        for banned in ["hash", "config", "design_id", "ts_utc", "lint_rc"]:
            assert banned not in t.columns

    def test_label_is_the_tile_max_not_the_mean(self):
        """The label must be the MAX fine drop in the tile. If someone swaps in
        the mean, label_v drops below the tile max and this fails."""
        import numpy as np
        from prism.io_csv import load_config, load_design
        cfg = load_config()
        t = self._table()
        d = load_design("data/synthetic/syn_000")
        ny_f, nx_f = cfg["grid"]["ny_fine"], cfg["grid"]["nx_fine"]
        ny_c, nx_c = cfg["grid"]["ny_coarse"], cfg["grid"]["nx_coarse"]
        ratio = ny_f // ny_c
        s = d.irmap[d.irmap["scenario"] == "seq_read"]
        fine = np.zeros((ny_f, nx_f))
        fine[s["fy"].values.astype(int), s["fx"].values.astype(int)] = s["drop_v"].values
        blocks = fine.reshape(ny_c, ratio, nx_c, ratio)
        want_max = blocks.max(axis=(1, 3)).ravel()
        want_mean = blocks.mean(axis=(1, 3)).ravel()
        got = (t[(t.design == "syn_000") & (t.scenario == "seq_read")]
               .sort_values(["ty", "tx"])["label_v"].to_numpy())
        assert np.allclose(got, want_max, atol=1e-9)
        assert not np.allclose(got, want_mean)

    def test_every_design_and_scenario_present(self):
        from prism.io_csv import load_config
        cfg = load_config()
        t = self._table()
        assert t["design"].nunique() == cfg["corpus"]["n_designs"]
        assert t["scenario"].nunique() == len(cfg["scenarios"])
        assert (t.groupby(["design", "scenario"]).size() == 576).all()

    def test_grid_weak_is_activity_independent(self):
        """grid_weak is a property of the planned grid and bump array only, so
        it must not vary with the scenario."""
        t = self._table()
        spread = (t.groupby(["design", "ty", "tx"])["grid_weak"]
                   .nunique().max())
        assert spread == 1, "grid_weak varies across scenarios — it must not"

    def test_phys_base_rank_is_within_design(self):
        """A corpus-global rank would encode absolute design scale."""
        t = self._table()
        for _, sub in t.groupby(["design", "scenario"]):
            assert abs(sub["phys_base_rank"].max() - 1.0) < 1e-9
            assert sub["phys_base_rank"].min() > 0


# ---------------------------------------------------------------------------
# S4 — model, partitioning and conformal interval
# ---------------------------------------------------------------------------

class TestS4Partition:
    """The split protocol. R1: split by design, never by tile."""

    @staticmethod
    def _pool():
        from prism.io_csv import load_config
        from prism.model import make_partition
        cfg = load_config()
        designs = [f"syn_{i:03d}" for i in range(cfg["corpus"]["n_designs"])]
        return cfg, make_partition(designs, cfg)

    def test_holdout_and_pool_are_disjoint(self):
        _, (holdout, pool) = self._pool()
        assert set(holdout).isdisjoint(pool)
        assert len(holdout) == 3 and len(pool) == 11

    def test_25_distinct_partitions(self):
        """GroupKFold would give 5 distinct partitions repeated 5 times, and the
        reported std would then measure early-stopping jitter, not split
        variance."""
        from prism.model import iter_splits
        cfg, (_, pool) = self._pool()
        splits = list(iter_splits(pool, cfg))
        assert len(splits) == 25
        sigs = {(tuple(tr), tuple(ca)) for _, _, tr, ca in splits}
        assert len(sigs) == 25, f"only {len(sigs)} distinct partitions"

    def test_holdout_never_appears_in_train_or_calib(self):
        from prism.model import iter_splits
        cfg, (holdout, pool) = self._pool()
        for _, _, train_ids, calib_ids in iter_splits(pool, cfg):
            assert set(holdout).isdisjoint(train_ids)
            assert set(holdout).isdisjoint(calib_ids)

    def test_train_and_calib_are_disjoint_and_sized(self):
        from prism.model import iter_splits
        cfg, (_, pool) = self._pool()
        for _, _, train_ids, calib_ids in iter_splits(pool, cfg):
            assert set(train_ids).isdisjoint(calib_ids)
            assert len(calib_ids) == cfg["validation"]["n_calib_designs"]
            assert len(train_ids) + len(calib_ids) == len(pool)


class TestS4Variants:
    def test_hybrid_uses_all_32_features(self):
        from prism.features import feature_columns
        from prism.model import variant_columns
        assert variant_columns("hybrid") == feature_columns()
        assert len(variant_columns("hybrid")) == 32

    def test_learned_only_drops_the_physics_prior(self):
        from prism.model import variant_columns
        cols = variant_columns("learned_only")
        assert len(cols) == 28
        assert not [c for c in cols if c.startswith("phys_")]

    def test_four_variants(self):
        from prism.model import VARIANTS
        assert VARIANTS == ["physics_only", "physics_affine",
                            "learned_only", "hybrid"]


class TestS4Conformal:
    def test_conformal_band_is_additive_only(self):
        import numpy as np
        from prism.model import apply_conformal
        q10 = np.array([0.0, 1.0, 2.0])
        q90 = np.array([1.0, 3.0, 10.0])
        lo, hi = apply_conformal(q10, q90, Q_add=0.5, Q_ratio=0.5)
        assert np.allclose(lo, q10 - 0.5)
        assert np.allclose(hi, q90 + 0.5)

    def test_additive_correction_covers_calibration_set(self):
        """Split conformal guarantees >= 1-alpha coverage on exchangeable data;
        on the calibration set itself that must hold by construction."""
        import numpy as np
        from prism.model import conformalise
        rng = np.random.RandomState(0)
        y = rng.normal(size=2000)
        q10, q90 = np.full(2000, -0.2), np.full(2000, 0.2)
        Q_add, _ = conformalise(q10, q90, y, alpha=0.20)
        cov = np.mean((y >= q10 - Q_add) & (y <= q90 + Q_add))
        assert cov >= 0.80

    def test_conformal_widens_a_broken_band(self):
        import numpy as np
        from prism.model import conformalise
        rng = np.random.RandomState(1)
        y = rng.normal(size=1000)
        q10, q90 = np.full(1000, -0.05), np.full(1000, 0.05)
        Q_add, Q_ratio = conformalise(q10, q90, y, alpha=0.20)
        assert Q_add > 0, "a band covering ~4% must be widened"


class TestS4Metrics:
    def test_f1_is_defined_when_a_slice_has_no_positives(self):
        """`idle` has a 0.00% violation rate. Pooled F1 must not blow up, and
        this is why F1 is computed on pooled rows rather than averaged over
        per-scenario F1."""
        import numpy as np
        import pandas as pd
        from prism.model import compute_metrics
        df = pd.DataFrame({"label_v": np.full(100, 0.010)})
        m = compute_metrics(df, np.full(100, 0.011), None, None, 0.045)
        assert m["violation_f1"] == 0.0
        assert np.isfinite(m["violation_f1"])

    def test_affine_baseline_preserves_rank_order(self):
        """physics_affine is a monotone transform of physics_only, so Spearman
        must be identical. If it is not, the affine fit has a negative slope
        and something is wrong."""
        import numpy as np
        import pandas as pd
        from prism.model import compute_metrics
        rng = np.random.RandomState(0)
        phys = rng.uniform(0.01, 0.06, 500)
        df = pd.DataFrame({"label_v": phys * 1.3 + rng.normal(0, 0.002, 500)})
        a = compute_metrics(df, phys, None, None, 0.045)["spearman"]
        b = compute_metrics(df, 0.003 + 1.38 * phys, None, None, 0.045)["spearman"]
        assert abs(a - b) < 1e-12


@pytest.mark.skipif(
    not pathlib.Path("models/manifest.json").exists(),
    reason="models not trained; run `python run_all.py train`",
)
class TestS4Artefacts:
    def test_manifest_records_provenance(self):
        import json
        m = json.loads(pathlib.Path("models/manifest.json").read_text())
        for key in ["config_hash", "sklearn_version", "seeds", "partitions",
                    "split_scheme", "variants"]:
            assert key in m, f"manifest missing {key}"
        for part in ["holdout", "canonical_train", "canonical_calib"]:
            assert m["partitions"][part]

    def test_manifest_partitions_are_disjoint(self):
        import json
        m = json.loads(pathlib.Path("models/manifest.json").read_text())
        p = m["partitions"]
        assert set(p["holdout"]).isdisjoint(p["canonical_train"])
        assert set(p["holdout"]).isdisjoint(p["canonical_calib"])
        assert set(p["canonical_train"]).isdisjoint(p["canonical_calib"])

    def test_joblib_artefacts_exist(self):
        for v in ["hybrid", "learned_only"]:
            assert pathlib.Path(f"models/{v}.joblib").exists()

    def test_run_table_has_25_runs_per_variant(self):
        import pandas as pd
        from prism.model import VARIANTS
        res = pd.read_csv("out/model_runs.csv")
        for v in VARIANTS:
            assert (res["variant"] == v).sum() == 25

    def test_required_ablation_ordering(self):
        """BUILD_SPEC 6.S5: hybrid > learned_only > physics_only on violation F1."""
        import pandas as pd
        res = pd.read_csv("out/model_runs.csv")
        f1 = res.groupby("variant")["violation_f1"].mean()
        assert f1["hybrid"] > f1["learned_only"] > f1["physics_only"]

    def test_physics_baseline_is_biased_low(self):
        import pandas as pd
        res = pd.read_csv("out/model_runs.csv")
        bias = res[res["variant"] == "physics_only"]["bias_mv"].mean()
        assert bias < -3.0, f"physics_only bias {bias:+.3f} mV is not clearly negative"


# ---------------------------------------------------------------------------
# S5 gates — evaluation, validation table, PR-AUC, F-beta crossover
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pathlib.Path("out/validation.csv").exists(),
    reason="evaluation not run; run `python run_all.py eval`",
)
class TestS5Validation:
    def test_validation_csv_has_required_columns(self):
        import pandas as pd
        df = pd.read_csv("out/validation.csv")
        req = ["variant", "metric", "split", "mean", "std", "ci_lo", "ci_hi"]
        for c in req:
            assert c in df.columns, f"validation.csv missing {c}"

    def test_all_four_variants_present(self):
        import pandas as pd
        from prism.model import VARIANTS
        df = pd.read_csv("out/validation.csv")
        variants = df["variant"].unique()
        for v in VARIANTS:
            assert v in variants, f"validation.csv missing variant {v}"

    def test_pr_auc_measured_and_beats_learned(self):
        import pandas as pd
        df = pd.read_csv("out/validation.csv")
        pra = df[df["metric"] == "pr_auc"].set_index("variant")["mean"]
        assert pra["hybrid"] > pra["learned_only"], "hybrid must beat learned_only on PR-AUC"
        assert pra["hybrid"] > 0.90

    def test_pr_curves_file_valid(self):
        import pandas as pd
        from prism.model import VARIANTS
        df = pd.read_csv("out/pr_curves.csv")
        assert len(df) > 0
        for col in ["variant", "precision", "recall", "threshold_v"]:
            assert col in df.columns
        for v in VARIANTS:
            assert v in df["variant"].unique()

    def test_fbeta_crossover_in_range(self):
        import pandas as pd
        df = pd.read_csv("out/validation.csv")
        sub = df[(df["variant"] == "hybrid_vs_physics_affine") & (df["metric"] == "fbeta_crossover")]
        assert len(sub) == 1
        beta = float(sub.iloc[0]["mean"])
        assert 1.20 <= beta <= 1.50, f"crossover beta {beta:.4f} outside expected [1.20, 1.50]"

    def test_headline_findings_exists(self):
        p = pathlib.Path("out/headline_findings.md")
        assert p.exists()
        assert p.stat().st_size > 200


# ---------------------------------------------------------------------------
# S6 gates — figure generation and integrity
# ---------------------------------------------------------------------------

class TestS6Figures:
    EXPECTED_FIGURES = [
        "fig1_two_fidelity.png",
        "fig2_ablation.png",
        "fig3_pred_vs_label.png",
        "fig4_scenario_grid.png",
        "fig5_calibration.png",
        "fig6_importance.png",
        "fig7_residual.png",
        "fig8_error_map.png",
        "fig9_scaling.png",
        "fig10_pr_curves.png",
        "fig11_calibration_transfer.png",
    ]

    def test_all_eleven_figures_exist_and_over_50kb(self):
        fig_dir = pathlib.Path("figures")
        assert fig_dir.exists(), "figures/ directory must exist"
        for name in self.EXPECTED_FIGURES:
            path = fig_dir / name
            assert path.exists(), f"Missing figure {name}"
            size_kb = path.stat().st_size / 1024.0
            assert size_kb > 50, f"Figure {name} size {size_kb:.1f} kB is <= 50 kB"

    def test_viz_contains_no_hardcoded_metric_annotations(self):
        """Assert viz.py reads metrics at runtime and contains no hardcoded metric literals."""
        import ast
        import pandas as pd
        viz_path = pathlib.Path("prism/viz.py")
        content = viz_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        val_df = pd.read_csv("out/validation.csv")
        metrics_to_check = [
            "pr_auc", "violation_f1", "violation_precision", "violation_recall",
            "mae_mv", "rmse_mv", "r2", "spearman", "bias_mv"
        ]
        measured_values = set()
        for _, row in val_df[val_df["metric"].isin(metrics_to_check)].iterrows():
            m_val = round(float(row["mean"]), 4)
            if m_val not in (0.0, 1.0):
                measured_values.add(m_val)

        numeric_literals = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                numeric_literals.add(round(float(node.value), 4))

        overlap = measured_values.intersection(numeric_literals)
        assert len(overlap) == 0, f"Found hardcoded metric literals in viz.py: {overlap}"

    def test_physics_pr_curves_are_identical(self):
        """Assert physics_only and physics_affine share identical PR curves."""
        import numpy as np
        import pandas as pd
        pr_df = pd.read_csv("out/pr_curves.csv")
        p1 = pr_df[pr_df["variant"] == "physics_only"][["precision", "recall"]].reset_index(drop=True)
        p2 = pr_df[pr_df["variant"] == "physics_affine"][["precision", "recall"]].reset_index(drop=True)
        assert len(p1) == len(p2)
        assert np.allclose(p1.to_numpy(), p2.to_numpy()), "Physics curves must be identical"


# ---------------------------------------------------------------------------
# S7 gates — Streamlit dashboard integrity & Role C delivery contract
# ---------------------------------------------------------------------------

class TestS7Dashboard:
    """Gates for Session S7: app.py and out/predictions.csv contract."""

    def test_predictions_contract(self):
        """Assert out/predictions.csv matches role C contract exactly."""
        import numpy as np
        import pandas as pd

        p_path = pathlib.Path("out/predictions.csv")
        assert p_path.exists(), "out/predictions.csv must exist"

        # Check header comment
        first_line = p_path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#"), "out/predictions.csv must start with header comment"
        assert "tile_id = ty * 24 + tx" in first_line, "Formula must be documented in header"
        assert "inversion: ty = tile_id // 24, tx = tile_id % 24" in first_line

        df = pd.read_csv(p_path, comment="#")
        expected_cols = ["design", "scenario", "partition", "tile_id", "pred_v", "lo_v", "hi_v", "label_v", "coarse_v"]
        assert list(df.columns) == expected_cols, f"Columns must be exact: {list(df.columns)}"

        # Row count == 48,384
        assert len(df) == 48384, f"Row count {len(df)} != 48384"

        # Partition counts: train=27648, calib=10368, holdout=10368
        p_counts = df["partition"].value_counts().to_dict()
        assert p_counts.get("train") == 27648, f"Train count {p_counts.get('train')} != 27648"
        assert p_counts.get("calib") == 10368, f"Calib count {p_counts.get('calib')} != 10368"
        assert p_counts.get("holdout") == 10368, f"Holdout count {p_counts.get('holdout')} != 10368"

        # Zero NaN
        assert df.isna().sum().sum() == 0, f"Found {df.isna().sum().sum()} NaNs in predictions.csv"

        # Volts not millivolts
        assert df["pred_v"].max() < 0.20, f"Values must be Volts, found max {df['pred_v'].max()}"
        assert df["label_v"].max() < 0.20, f"Values must be Volts, found max {df['label_v'].max()}"

        # tile_id range
        assert df["tile_id"].min() == 0 and df["tile_id"].max() == 575

        # lo_v <= pred_v <= hi_v everywhere
        assert np.all(df["lo_v"] <= df["pred_v"]), "lo_v <= pred_v violated"
        assert np.all(df["pred_v"] <= df["hi_v"]), "pred_v <= hi_v violated"

        # Coverage of label_v by [lo_v, hi_v] strictly on holdout partition within [0.78, 0.90]
        holdout_df = df[df["partition"] == "holdout"]
        holdout_cov = float(np.mean((holdout_df["label_v"] >= holdout_df["lo_v"]) & (holdout_df["label_v"] <= holdout_df["hi_v"])))
        assert 0.78 <= holdout_cov <= 0.90, f"Holdout coverage {holdout_cov:.4f} outside [0.78, 0.90]"
        assert np.isclose(holdout_cov, 0.8205, atol=0.005), f"Holdout coverage {holdout_cov:.4f} not matching 0.8205"

    def test_app_never_computes_no_ml_imports(self):
        """Assert rule R7: app.py never imports joblib, sklearn, or prism.model
        at *module scope*.  Function-local imports inside declared compute
        helpers (load_hybrid_model, compute_design_bundle) are permitted
        because they only execute on the Upload Custom CSV path."""
        import ast

        app_path = pathlib.Path("app.py")
        assert app_path.exists(), "app.py must exist"

        content = app_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        banned_modules = {"joblib", "sklearn", "prism.model", "prism.solver"}

        # Allowed enclosing functions — these run only on user-triggered upload
        ALLOWED_FUNCTIONS = {"load_hybrid_model", "compute_design_bundle"}

        # Collect line ranges of allowed function bodies
        allowed_lines: set[int] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ALLOWED_FUNCTIONS:
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        allowed_lines.add(child.lineno)

        # Check all imports; ban only those NOT inside an allowed function
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in banned_modules and node.lineno not in allowed_lines:
                        raise AssertionError(
                            f"Banned module-scope import at line {node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                is_banned = mod in banned_modules or any(mod.startswith(b) for b in banned_modules)
                if is_banned and node.lineno not in allowed_lines:
                    raise AssertionError(
                        f"Banned module-scope import at line {node.lineno}: from {mod}"
                    )
            elif isinstance(node, ast.Call):
                # .predict() is banned everywhere — even inside helpers the
                # dashboard must not call model.predict() directly.
                if isinstance(node.func, ast.Attribute) and node.func.attr == "predict":
                    raise AssertionError(
                        f"Disallowed .predict() call found at line {node.lineno}"
                    )

    def test_app_uses_st_cache_data(self):
        """Assert that data loaders in app.py use @st.cache_data."""
        app_path = pathlib.Path("app.py")
        content = app_path.read_text(encoding="utf-8")
        assert "@st.cache_data" in content, "Data loaders must be decorated with @st.cache_data"

    def test_app_all_four_pages_render_without_exception(self):
        """Assert all 4 pages in app.py render cleanly without errors."""
        from streamlit.testing.v1 import AppTest

        app_file = str(pathlib.Path("app.py").resolve())
        at = AppTest.from_file(app_file, default_timeout=10)
        at.run()
        assert not at.exception, f"Exception on Page 1 (Predict): {at.exception}"

        at.sidebar.radio[0].set_value("Validate").run()
        assert not at.exception, f"Exception on Page 2 (Validate): {at.exception}"

        at.sidebar.radio[0].set_value("Scenarios").run()
        assert not at.exception, f"Exception on Page 3 (Scenarios): {at.exception}"

        at.sidebar.radio[0].set_value("Findings").run()
        assert not at.exception, f"Exception on Page 4 (Findings): {at.exception}"
