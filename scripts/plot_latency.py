#!/usr/bin/env python3
"""Plot p50/p99 latency vs rps from a mini_batcher benchmark CSV.

Usage:
    python scripts/plot_latency.py results/step1.csv
    python scripts/plot_latency.py results/step1.csv -o results/step1.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe pair: blue / vermillion.
P50_COLOR = "#0072B2"
P99_COLOR = "#D55E00"


def load_rows(path: Path):
    with path.open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    rows.sort(key=lambda r: r["rps"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="results/stepN.csv")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="output PNG path (default: same name as the CSV, .png)",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    rps = [r["rps"] for r in rows]
    p50 = [r["p50"] for r in rows]
    p99 = [r["p99"] for r in rows]

    out_path = args.out or args.csv_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rps, p50, marker="o", markersize=8, linewidth=2, color=P50_COLOR, label="p50")
    ax.plot(rps, p99, marker="o", markersize=8, linewidth=2, color=P99_COLOR, label="p99")
    ax.set_xlabel("requests / second")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Latency vs load — {args.csv_path.stem}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
