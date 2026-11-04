"""Verify short pre-ectopic coupling and longer post-ectopic pauses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_dataset import CLASS_CONFIGS, _generate_beat_train


def collect_rr_pairs(samples: int, duration: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    config = CLASS_CONFIGS["arrhythmia"]
    rows: list[dict[str, float | int]] = []
    for sample_id in range(samples):
        heart_rate = float(rng.uniform(*config.hr_range))
        hrv = float(rng.uniform(*config.hrv_range))
        times, kinds = _generate_beat_train(duration, heart_rate, hrv, True, rng)
        rr = np.diff(times)
        in_record_rr = rr[(np.asarray(times[:-1]) >= 0) & (np.asarray(times[1:]) < duration)]
        median_rr = float(np.median(in_record_rr)) if in_record_rr.size else np.nan
        for idx, kind in enumerate(kinds):
            if kind != "ectopic" or idx == 0 or idx >= len(times) - 1:
                continue
            if not (0 <= times[idx] < duration):
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "ectopic_index": idx,
                    "median_rr_seconds": median_rr,
                    "rr_before_seconds": times[idx] - times[idx - 1],
                    "rr_after_seconds": times[idx + 1] - times[idx],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pairs = collect_rr_pairs(args.samples, args.duration, args.seed)
    if pairs.empty:
        raise RuntimeError("No complete ectopic RR pairs were generated.")

    summary = pd.DataFrame(
        {
            "measure": ["class_median_rr", "rr_before_ectopic", "rr_after_ectopic"],
            "n": [len(pairs)] * 3,
            "mean_seconds": [
                pairs.median_rr_seconds.mean(),
                pairs.rr_before_seconds.mean(),
                pairs.rr_after_seconds.mean(),
            ],
            "std_seconds": [
                pairs.median_rr_seconds.std(),
                pairs.rr_before_seconds.std(),
                pairs.rr_after_seconds.std(),
            ],
        }
    )
    pairs.to_csv(results_dir / "ectopic_rr_pairs.csv", index=False)
    summary.to_csv(results_dir / "ectopic_rr_summary.csv", index=False)

    before_mean = float(pairs.rr_before_seconds.mean())
    after_mean = float(pairs.rr_after_seconds.mean())
    median_mean = float(pairs.median_rr_seconds.mean())
    print(summary.to_string(index=False))
    print(f"Pre-ectopic mean < mean within-record median: {before_mean:.4f} < {median_mean:.4f}")
    print(f"Post-ectopic mean > mean within-record median: {after_mean:.4f} > {median_mean:.4f}")
    if not (before_mean < median_mean < after_mean):
        raise AssertionError("Expected short-before/long-after ectopic RR ordering was not observed.")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    values = [pairs.rr_before_seconds, pairs.rr_after_seconds]
    means = [float(v.mean()) for v in values]
    stds = [float(v.std()) for v in values]
    ax.bar(["Before ectopic", "After ectopic"], means, yerr=stds, capsize=6, color=["#2f6f9f", "#c65353"])
    ax.axhline(median_mean, color="#333333", linestyle="--", linewidth=1.4, label="Mean within-record median RR")
    ax.set_ylabel("RR interval (s)")
    ax.set_title("Ectopic coupling interval and compensatory pause")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "ectopic_rr_before_after.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
