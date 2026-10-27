#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from syntheticshield.plotting import plot_main_pipeline, plot_result_panels


def main():
    parser = argparse.ArgumentParser(description="Generate SyntheticShield paper figures.")
    parser.add_argument("--results", default="outputs/metrics_summary.csv")
    parser.add_argument("--canaries", default="outputs/canary_exposure.csv")
    parser.add_argument("--out", default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_path = plot_main_pipeline(out_dir)
    panel_paths = plot_result_panels(args.results, args.canaries, out_dir)
    print("Generated figures:")
    print(main_path)
    for p in panel_paths:
        print(p)


if __name__ == "__main__":
    main()
