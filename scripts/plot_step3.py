#!/usr/bin/env python3
"""Overlay p50/p99 latency and average batch size vs rps across delay configs.

Usage:
    python scripts/plot_step3.py results/step3_delay0.csv results/step3_delay5000.csv \
        --labels "delay=0us" "delay=5000us" -o results/step3.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe pair, one hue per config (never per metric).
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

MAX_BATCH_SIZE = 8  # mirrors main.cpp's MAX_BATCH_SIZE, for the reference line


def load_rows(path: Path):
    with path.open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    rows.sort(key=lambda r: r["rps"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", type=Path, nargs="+", help="results/step3_*.csv files to overlay")
    parser.add_argument("--labels", nargs="+", help="one label per csv (default: filename stem)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    labels = args.labels or [p.stem for p in args.csv_paths]
    if len(labels) != len(args.csv_paths):
        parser.error("--labels must have one entry per csv file")

    fig, (ax_lat, ax_batch) = plt.subplots(1, 2, figsize=(13, 5))

    for path, label, color in zip(args.csv_paths, labels, COLORS):
        rows = load_rows(path)
        rps = [r["rps"] for r in rows]
        p50 = [r["p50"] for r in rows]
        p99 = [r["p99"] for r in rows]
        avg_batch = [r["avg_batch"] for r in rows]

        ax_lat.plot(rps, p50, marker="o", markersize=7, linewidth=2, color=color,
                    linestyle="-", label=f"{label} p50")
        ax_lat.plot(rps, p99, marker="o", markersize=7, linewidth=2, color=color,
                    linestyle="--", label=f"{label} p99")
        ax_batch.plot(rps, avg_batch, marker="o", markersize=7, linewidth=2, color=color,
                      label=label)

    ax_lat.set_xlabel("requests / second")
    ax_lat.set_ylabel("latency (ms)")
    ax_lat.set_title("p50 / p99 latency vs load")
    ax_lat.grid(True, axis="y", alpha=0.3)
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)
    ax_lat.legend(frameon=False, fontsize=9)

    ax_batch.set_xlabel("requests / second")
    ax_batch.set_ylabel("average batch size")
    ax_batch.set_title("average batch size vs load")
    ax_batch.grid(True, axis="y", alpha=0.3)
    ax_batch.spines["top"].set_visible(False)
    ax_batch.spines["right"].set_visible(False)
    ax_batch.axhline(MAX_BATCH_SIZE, color="#999999", linewidth=1, linestyle=":")
    ax_batch.text(1.0, MAX_BATCH_SIZE, " MAX_BATCH_SIZE ", transform=ax_batch.get_yaxis_transform(),
                  va="bottom", ha="right", fontsize=8, color="#999999")
    ax_batch.legend(frameon=False, fontsize=9)

    fig.suptitle("Step 3: max queue delay")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
