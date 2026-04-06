#!/usr/bin/env python3
"""Tune ensemble or SegRank selector heuristics against a validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypseg.tuning import load_tuning_config, run_selector_tuning


def main() -> None:
    """Parse arguments, run selector tuning, and print the best trial summary."""
    parser = argparse.ArgumentParser(description="Tune ensemble or SegRank selectors on validation data.")
    parser.add_argument("--config", type=str, default="configs/tuning/selector_val_search.yaml")
    args = parser.parse_args()

    tuning_config = load_tuning_config(args.config)
    result = run_selector_tuning(tuning_config=tuning_config, project_root=ROOT)
    payload = {
        "mode": result["mode"],
        "metric": result["metric"],
        "num_trials_evaluated": result["num_trials_evaluated"],
        "best_trial": {
            "trial_index": result["best_trial"]["trial_index"],
            "metric_name": result["best_trial"]["metric_name"],
            "metric_value": result["best_trial"]["metric_value"],
            "parameters": result["best_trial"]["parameters"],
            "summary": result["best_trial"]["summary"]["overall"],
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
