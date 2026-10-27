#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from syntheticshield.auditor import canary_exposure_audit, membership_inference_audit, threshold_sweep
from syntheticshield.data import make_private_dataset
from syntheticshield.generators import build_generator
from syntheticshield.metrics import synthetic_utility


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_once(cfg: dict, defense: str, repeat: int, out_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(cfg["seed"]) + repeat * 100
    ds = make_private_dataset(
        seed=seed,
        n_train=int(cfg["n_train"]),
        n_test=int(cfg["n_test"]),
        n_features=int(cfg["n_features"]),
        n_canaries_per_rarity=int(cfg["n_canaries_per_rarity"]),
        rarity_levels=cfg["rarity_levels"],
    )

    gen = build_generator(defense, seed=seed + 13)
    gen.fit(ds.train, ds.feature_columns, ds.label_column)
    synthetic = gen.sample(int(cfg["synthetic_size"]))

    utility = synthetic_utility(synthetic, ds.test, ds.feature_columns, ds.label_column)
    audit = membership_inference_audit(
        train_members=ds.train,
        test_nonmembers=ds.test,
        synthetic=synthetic,
        feature_columns=ds.feature_columns,
        fpr_target=float(cfg["attack_fpr"]),
    )
    row = {
        "repeat": repeat,
        "defense": defense,
        "n_train_with_canaries": len(ds.train),
        "n_test": len(ds.test),
        "n_synthetic": len(synthetic),
        **utility,
        **audit,
    }

    canary = canary_exposure_audit(
        canaries=ds.canaries,
        synthetic=synthetic,
        feature_columns=ds.feature_columns,
        epsilon=float(cfg["canary_epsilon"]),
    )
    canary["repeat"] = repeat
    canary["defense"] = defense

    sweep = threshold_sweep(ds.train, ds.test, synthetic, ds.feature_columns)
    sweep["repeat"] = repeat
    sweep["defense"] = defense

    if repeat == 0:
        synthetic.head(100).to_csv(out_dir / f"synthetic_preview_{defense}.csv", index=False)

    return row, canary, sweep, synthetic


def main():
    parser = argparse.ArgumentParser(description="Run SyntheticShield privacy leakage experiments.")
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON config.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    canaries = []
    sweeps = []
    for repeat in range(int(cfg["repeats"])):
        for defense in cfg["defenses"]:
            row, canary, sweep, _ = run_once(cfg, defense, repeat, out_dir)
            rows.append(row)
            canaries.append(canary)
            sweeps.append(sweep)
            print(
                f"repeat={repeat} defense={defense} "
                f"utility_auc={row['utility_auc']:.3f} "
                f"leakage@5fpr={row['leakage_rate_at_5pct_fpr']:.3f} "
                f"mia_auc={row['membership_auc']:.3f}"
            )

    pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.concat(canaries, ignore_index=True).to_csv(out_dir / "canary_exposure.csv", index=False)
    pd.concat(sweeps, ignore_index=True).to_csv(out_dir / "threshold_sweep.csv", index=False)

    print(f"Wrote results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
