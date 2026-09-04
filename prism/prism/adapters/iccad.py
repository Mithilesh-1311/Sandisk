"""ICCAD 2023 Contest Problem C Benchmark Ingestion Adapter for PRISM.

This module provides clean, zero-overhead ingestion of real-circuit benchmarks
from the ICCAD 2023 'ML-for-IR-drop' repository (Nangate45 / ASAP7 layouts).

It extracts:
  - current_map: cell current distributions I(x,y)
  - eff_dist_map: anisotropic resistive distance to VDD sources D_eff(x,y)
  - pdn_density: local metal routing and strap layer density rho_pdn(x,y)
  - ir_drop_map: signoff ground-truth static IR drop U(x,y)
"""

import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


_DEFAULT_BENCHMARK_ROOTS = [
    pathlib.Path("C:/Users/Admin/OneDrive/Desktop/Gitu/ML-for-IR-drop/benchmarks"),
    pathlib.Path("../Gitu/ML-for-IR-drop/benchmarks"),
    pathlib.Path("./data/iccad2023"),
]


def find_benchmark_root(custom_path: Optional[Union[str, pathlib.Path]] = None) -> Optional[pathlib.Path]:
    """Locate the ICCAD 2023 benchmarks root directory."""
    if custom_path:
        p = pathlib.Path(custom_path)
        if p.exists():
            return p

    for root in _DEFAULT_BENCHMARK_ROOTS:
        if root.exists():
            return root
    return None


def list_real_benchmarks(root_path: Optional[Union[str, pathlib.Path]] = None) -> List[str]:
    """Return list of available real-circuit benchmark testcase names."""
    root = find_benchmark_root(root_path)
    if not root:
        return []

    real_dir = root / "real-circuit-data"
    if not real_dir.exists():
        return []

    testcases = []
    for d in sorted(real_dir.iterdir()):
        if d.is_dir() and (d / "current_map.csv").exists():
            testcases.append(d.name)
    return testcases


def load_testcase(testcase_name: str,
                  root_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, np.ndarray]:
    """Load the 4 synchronized spatial matrices for an ICCAD testcase.

    Returns:
        dict with keys:
            'current': Current map in Amperes (shape N x N)
            'eff_dist': Effective distance map (shape N x N)
            'pdn_density': PDN layer density map (shape N x N)
            'ir_drop': Ground truth IR drop in Volts (shape N x N)
            'name': Testcase identifier
            'shape': (N, N)
    """
    root = find_benchmark_root(root_path)
    if not root:
        raise FileNotFoundError("ICCAD 2023 benchmarks root directory not found.")

    tc_dir = root / "real-circuit-data" / testcase_name
    if not tc_dir.exists():
        raise FileNotFoundError(f"Testcase '{testcase_name}' not found at {tc_dir}")

    current = pd.read_csv(tc_dir / "current_map.csv", header=None).to_numpy(dtype=float)
    eff_dist = pd.read_csv(tc_dir / "eff_dist_map.csv", header=None).to_numpy(dtype=float)
    pdn_density = pd.read_csv(tc_dir / "pdn_density.csv", header=None).to_numpy(dtype=float)
    ir_drop = pd.read_csv(tc_dir / "ir_drop_map.csv", header=None).to_numpy(dtype=float)

    assert current.shape == eff_dist.shape == pdn_density.shape == ir_drop.shape, \
        f"Matrix shape mismatch in {testcase_name}"

    return {
        "name": testcase_name,
        "shape": current.shape,
        "current": current,
        "eff_dist": eff_dist,
        "pdn_density": pdn_density,
        "ir_drop": ir_drop,
    }


def downsample_to_grid(matrix: np.ndarray, target_shape: Tuple[int, int] = (24, 24)) -> np.ndarray:
    """Downsample high-resolution matrix to target coarse grid using area-averaging."""
    ny, nx = matrix.shape
    ty, tx = target_shape

    y_edges = np.linspace(0, ny, ty + 1).astype(int)
    x_edges = np.linspace(0, nx, tx + 1).astype(int)

    coarse = np.zeros(target_shape, dtype=float)
    for i in range(ty):
        for j in range(tx):
            coarse[i, j] = np.mean(matrix[y_edges[i]:y_edges[i+1], x_edges[j]:x_edges[j+1]])
    return coarse


def compute_testcase_stats(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute key electrical and layout metrics for an ICCAD testcase."""
    current = data["current"]
    ir_drop = data["ir_drop"]
    eff_dist = data["eff_dist"]
    pdn_density = data["pdn_density"]

    return {
        "grid_resolution": float(data["shape"][0]),
        "total_current_ma": float(np.sum(current)) * 1000.0,
        "peak_tile_current_ua": float(np.max(current)) * 1e6,
        "max_ir_drop_mv": float(np.max(ir_drop)) * 1000.0,
        "mean_ir_drop_mv": float(np.mean(ir_drop)) * 1000.0,
        "mean_eff_dist": float(np.mean(eff_dist)),
        "max_eff_dist": float(np.max(eff_dist)),
        "mean_pdn_density": float(np.mean(pdn_density)),
        "min_pdn_density": float(np.min(pdn_density)),
    }
