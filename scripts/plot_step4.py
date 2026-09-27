#!/usr/bin/env python3
"""p99 latency and average batch size vs rps, one line per (instances, delay) config.

Usage:
    python scripts/plot_step4.py results/step4_i1_d0.csv results/step4_i2_d0.csv ... \
        --labels "i=1 d=0" "i=2 d=0" ... -o results/step4.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette, assigned in fixed order (never cycled).
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#F0E442"]

MAX_BATCH_SIZE = 8  # mirrors main.cpp's MAX_BATCH_SIZE, for the reference line


def load_rows(path: Path):
    with path.open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    rows.sort(key=lambda r: r["rps"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", type=Path, nargs="+", help="results/step4_i*_d*.csv files")
    parser.add_argument("--labels", nargs="+", help="one label per csv (default: filename stem)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    labels = args.labels or [p.stem for p in args.csv_paths]
    if len(labels) != len(args.csv_paths):
        parser.error("--labels must have one entry per csv file")
    if len(args.csv_paths) > len(COLORS):
        parser.error(f"only {len(COLORS)} distinct series colors defined")

    fig, (ax_lat, ax_batch) = plt.subplots(1, 2, figsize=(13, 5))

    for path, label, color in zip(args.csv_paths, labels, COLORS):
        rows = load_rows(path)
        rps = [r["rps"] for r in rows]
        p99 = [r["p99"] for r in rows]
        avg_batch = [r["avg_batch"] for r in rows]
        ax_lat.plot(rps, p99, marker="o", markersize=6, linewidth=2, color=color, label=label)
        ax_batch.plot(rps, avg_batch, marker="o", markersize=6, linewidth=2, color=color, label=label)

    ax_lat.set_xlabel("requests / second")
    ax_lat.set_ylabel("p99 latency (ms, log scale)")
    ax_lat.set_yscale("log")
    ax_lat.set_title("p99 latency vs load")
    ax_lat.grid(True, axis="y", alpha=0.3, which="both")
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)
    ax_lat.legend(frameon=False, fontsize=8)

    ax_batch.set_xlabel("requests / second")
    ax_batch.set_ylabel("average batch size")
    ax_batch.set_title("average batch size vs load")
    ax_batch.grid(True, axis="y", alpha=0.3)
    ax_batch.spines["top"].set_visible(False)
    ax_batch.spines["right"].set_visible(False)
    ax_batch.axhline(MAX_BATCH_SIZE, color="#999999", linewidth=1, linestyle=":")
    ax_batch.text(1.0, MAX_BATCH_SIZE, " MAX_BATCH_SIZE ", transform=ax_batch.get_yaxis_transform(),
                  va="bottom", ha="right", fontsize=8, color="#999999")
    ax_batch.legend(frameon=False, fontsize=8)

    fig.suptitle("Step 4: batcher + N instance threads")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
