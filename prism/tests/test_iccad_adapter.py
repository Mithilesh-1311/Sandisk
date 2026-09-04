"""test_iccad_adapter.py — Acceptance tests for ICCAD 2023 benchmark integration."""

import pathlib
import numpy as np
import pytest

from prism.adapters.iccad import (
    find_benchmark_root,
    list_real_benchmarks,
    load_testcase,
    downsample_to_grid,
    compute_testcase_stats,
)
from prism.features import compute_effective_distance, compute_pdn_density


class TestICCADAdapter:
    """Test suite for ICCAD 2023 real-circuit benchmark ingestion."""

    def test_benchmark_root_discovery(self):
        """Locate ICCAD 2023 benchmark root directory."""
        root = find_benchmark_root()
        assert root is not None, "Failed to locate ICCAD 2023 benchmark directory"
        assert root.exists(), f"Benchmark root {root} does not exist"

    def test_list_real_benchmarks(self):
        """Discover real-circuit benchmarks."""
        benches = list_real_benchmarks()
        assert len(benches) >= 5, f"Expected at least 5 real testcases, found {len(benches)}"
        assert "testcase1" in benches, "testcase1 must be present"

    def test_load_testcase1_matrices(self):
        """Load testcase1 and assert matrix integrity."""
        data = load_testcase("testcase1")
        assert data["name"] == "testcase1"
        assert data["shape"] == (298, 298)

        # Non-empty, finite, zero NaNs
        for key in ["current", "eff_dist", "pdn_density", "ir_drop"]:
            mat = data[key]
            assert isinstance(mat, np.ndarray)
            assert mat.shape == (298, 298)
            assert np.all(np.isfinite(mat)), f"Non-finite values found in {key}"
            assert not np.any(np.isnan(mat)), f"NaN values found in {key}"

        # Physical value bounds
        assert data["current"].min() >= 0.0, "Current cannot be negative"
        assert data["ir_drop"].min() >= 0.0, "IR drop cannot be negative"
        assert data["pdn_density"].min() >= 0.0, "PDN density cannot be negative"
        assert data["pdn_density"].max() <= 4.0, "PDN density exceeds maximum routing layers"

    def test_compute_testcase_stats(self):
        """Compute electrical KPIs on real circuit."""
        data = load_testcase("testcase1")
        stats = compute_testcase_stats(data)

        assert stats["grid_resolution"] == 298.0
        assert stats["total_current_ma"] > 0.0
        assert stats["max_ir_drop_mv"] > 0.0
        assert stats["mean_eff_dist"] > 0.0
        assert stats["mean_pdn_density"] > 0.0

    def test_downsample_to_grid(self):
        """Area-average downsampling to coarse PRISM grid."""
        data = load_testcase("testcase1")
        coarse = downsample_to_grid(data["ir_drop"], (24, 24))
        assert coarse.shape == (24, 24)
        assert np.all(np.isfinite(coarse))
        # Mean drop must be approximately conserved
        assert np.isclose(coarse.mean(), data["ir_drop"].mean(), rtol=0.05)

    def test_features_effective_distance(self):
        """Test anisotropic effective distance extractor."""
        bumps = np.zeros((24, 24))
        bumps[6, 6] = 1.0
        bumps[18, 18] = 1.0

        ed = compute_effective_distance(bumps, x_pitch_ratio=1.0, y_pitch_ratio=2.0)
        assert ed.shape == (24, 24)
        assert ed[6, 6] == 0.0
        assert ed[18, 18] == 0.0
        # Check anisotropy: Y step costs 2x X step
        assert ed[7, 6] == 2.0
        assert ed[6, 7] == 1.0

    def test_features_pdn_density(self):
        """Test local PDN density extractor."""
        macros = np.zeros((24, 24))
        macros[8:16, 8:16] = 1.0

        pdn = compute_pdn_density(macros, base_layers=3, halo_tiles=1)
        assert pdn.shape == (24, 24)
        # Macro interior is blocked
        assert pdn[12, 12] == 0.0
        # Open core has full layers
        assert pdn[0, 0] == 3.0
        # Halo boundary is degraded
        assert pdn[7, 12] == 1.5
