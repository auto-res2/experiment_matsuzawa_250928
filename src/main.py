"""src/main.py
Entry-point that glues everything together.  It complies with the CLI contract

    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment

and can also take an explicit --config path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import yaml

from src.preprocess import DataPreprocessor
from src.train import SurrogateTrainer
from src.evaluate import evaluate, plot_training_curves

# default directories mandated by the spec
RESULT_DIR = Path(".research/iteration1/")
IMAGE_DIR = RESULT_DIR / "images/"


# --------------------------------------------------------------------------------------
#                                  CLI
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--smoke-test", action="store_true", help="Run quick sanity check with synthetic data")
    g.add_argument("--full-experiment", action="store_true", help="Run full experiment as per YAML config")

    p.add_argument("--config", type=str, default=None, help="Path to YAML config (overrides flags)")
    return p.parse_args()


# --------------------------------------------------------------------------------------
#                                  MAIN
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------- configuration resolution -------------------
    if args.config is not None:
        cfg_path = Path(args.config)
    elif args.smoke_test:
        cfg_path = Path("config/smoke_test.yaml")
    else:
        cfg_path = Path("config/full_experiment.yaml")

    with cfg_path.open() as fp:
        cfg: Dict = yaml.safe_load(fp)

    # experiment descriptor (printed to stdout)
    exp_name = cfg["misc"].get("experiment_name", cfg_path.stem)
    print("====================  Experiment Description  ====================")
    print(json.dumps({"experiment_name": exp_name, "config_path": str(cfg_path),
                      "dataset": cfg["data"]["name"], "lambda": cfg.get("monotonicity_lambda", 0.1)}, indent=2))
    print("==================================================================")

    # ------------------------- data pipeline -------------------------
    data_proc = DataPreprocessor(cfg)
    train_tuple, val_tuple = data_proc.load_dataset()

    # ------------------------- training ------------------------------
    trainer = SurrogateTrainer(cfg)
    model, likelihood, logs = trainer.fit(train_tuple, val_tuple)

    # ------------------------- evaluation ----------------------------
    final_metrics = evaluate(model, likelihood, val_tuple)

    # ------------------------- persistence ---------------------------
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RESULT_DIR / f"{exp_name}_results.json"
    with res_path.open("w") as fp:
        json.dump({"experiment": exp_name, "metrics": final_metrics}, fp, indent=2)

    # save training curves for later repro
    trainer.save_logs(logs, RESULT_DIR / f"{exp_name}_logs.json")

    # ------------------------- figures -------------------------------
    plot_training_curves(logs, IMAGE_DIR, exp_name)

    # ------------------------- std output ----------------------------
    print("\n====================  Numerical Results  ====================")
    print(json.dumps(final_metrics, indent=2))
    print("Results JSON saved to", res_path)


if __name__ == "__main__":
    main()
