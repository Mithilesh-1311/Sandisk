#!/usr/bin/env python
"""run_all.py — CLI dispatcher for the PRISM pipeline.

Usage:
    python run_all.py <stage> [--data-dir DATA_DIR]

Stages: gen | features | train | eval | figures | all
"""

import argparse
import sys
import time


def stage_gen(data_dir: str) -> None:
    """S2: Generate the synthetic corpus and run its acceptance gate."""
    from prism.io_csv import load_config
    from prism.design import build_corpus, _run_sanity_checks
    cfg = load_config()
    build_corpus(cfg)
    _run_sanity_checks(cfg)


def stage_features(data_dir: str) -> None:
    """S3: Build the feature table under the leakage trap."""
    from prism.io_csv import load_config
    from prism.features import build_feature_table, _run_feature_gate
    cfg = load_config()
    table = build_feature_table(cfg, data_dir=data_dir)
    _run_feature_gate(cfg, table)


def stage_train(data_dir: str) -> None:
    """S4: Train the hybrid predictor and run the 25-run ablation."""
    from prism.io_csv import load_config
    from prism.model import train_all, _run_model_gate, export_predictions_csv
    cfg = load_config()
    out = train_all(cfg)
    _run_model_gate(cfg, out)
    export_predictions_csv()


def stage_eval(data_dir: str) -> None:
    """S5: Evaluate and produce validation metrics."""
    from prism.io_csv import load_config
    from prism.evaluate import run_evaluation_pipeline
    from prism.model import export_predictions_csv
    cfg = load_config()
    run_evaluation_pipeline(cfg)
    export_predictions_csv()


def stage_figures(data_dir: str) -> None:
    """S6: Generate all matplotlib figures."""
    from prism.viz import run_figures_gate
    run_figures_gate()


_STAGES = {
    "gen": stage_gen,
    "features": stage_features,
    "train": stage_train,
    "eval": stage_eval,
    "figures": stage_figures,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=list(_STAGES.keys()) + ["all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
        help="Path to corpus directory (default: data/synthetic)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for name, fn in _STAGES.items():
            print(f"\n{'='*60}")
            print(f"  STAGE: {name}")
            print(f"{'='*60}")
            t0 = time.time()
            fn(args.data_dir)
            print(f"  [{name}] done in {time.time()-t0:.1f}s")
    else:
        _STAGES[args.stage](args.data_dir)


if __name__ == "__main__":
    main()
