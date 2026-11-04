"""Compare synthetic and mapped MIT-BIH RR-interval distributions."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import wfdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_dataset import CLASS_CONFIGS, _generate_beat_train


CLASSES = ["normal_sinus_rhythm", "bradycardia", "tachycardia", "arrhythmia"]
NON_BEAT_SYMBOLS = {"+", "~", "|", "[", "]", "!", "x", "(", ")", "p", "t", "u", "`", "'", "^"}


def synthetic_rr(seed: int = 5151, records_per_class: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        config = CLASS_CONFIGS[class_name]
        for record_id in range(records_per_class):
            heart_rate = float(rng.uniform(*config.hr_range))
            hrv = float(rng.uniform(*config.hrv_range))
            times, _ = _generate_beat_train(10.0, heart_rate, hrv, config.irregular, rng)
            in_window = np.asarray([t for t in times if 0 <= t < 10.0])
            for rr in np.diff(in_window):
                rows.append({"class_name": class_name, "rr_interval_s": float(rr)})
    return pd.DataFrame(rows)


def real_rr() -> pd.DataFrame:
    provenance = pd.read_csv(ROOT / "data" / "mitdb_subset" / "mitdb_subset_provenance.csv")
    annotations: dict[str, tuple[np.ndarray, list[str], float]] = {}
    rows: list[dict[str, object]] = []
    for record_name in sorted(provenance.record.astype(str).unique()):
        ann = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
        record = wfdb.rdheader(record_name, pn_dir="mitdb")
        annotations[record_name] = (ann.sample, ann.symbol, float(record.fs))

    for segment in provenance.itertuples(index=False):
        record_name = str(segment.record)
        samples, symbols, fs = annotations[record_name]
        start = float(segment.start_second) * fs
        stop = start + 10.0 * fs
        peaks = np.asarray(
            [sample for sample, symbol in zip(samples, symbols) if start <= sample < stop and symbol not in NON_BEAT_SYMBOLS],
            dtype=float,
        )
        for rr in np.diff(peaks) / fs:
            rows.append({"class_name": segment.mapped_class, "rr_interval_s": float(rr)})
    return pd.DataFrame(rows)


def stats_row(class_name: str, synthetic: np.ndarray, real: np.ndarray) -> dict[str, object]:
    syn_q1, syn_q3 = np.percentile(synthetic, [25, 75])
    real_q1, real_q3 = np.percentile(real, [25, 75])
    ks = ks_2samp(synthetic, real, alternative="two-sided", method="auto")
    return {
        "class_name": class_name,
        "synthetic_n": len(synthetic),
        "synthetic_mean_s": synthetic.mean(),
        "synthetic_std_s": synthetic.std(ddof=1),
        "synthetic_median_s": np.median(synthetic),
        "synthetic_iqr_s": syn_q3 - syn_q1,
        "real_n": len(real),
        "real_mean_s": real.mean(),
        "real_std_s": real.std(ddof=1),
        "real_median_s": np.median(real),
        "real_iqr_s": real_q3 - real_q1,
        "ks_statistic": ks.statistic,
        "ks_p_value": ks.pvalue,
        "wasserstein_distance_s": wasserstein_distance(synthetic, real),
    }


def main() -> None:
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    synthetic = synthetic_rr()
    real = real_rr()
    synthetic.to_csv(results_dir / "synthetic_rr_distributions.csv", index=False)
    real.to_csv(results_dir / "real_mitdb_rr_distributions.csv", index=False)

    comparison_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=False, sharey=False)
    for ax, class_name in zip(axes.flat, CLASSES):
        syn = synthetic.loc[synthetic.class_name == class_name, "rr_interval_s"].to_numpy()
        rea = real.loc[real.class_name == class_name, "rr_interval_s"].to_numpy()
        row = stats_row(class_name, syn, rea)
        comparison_rows.append(row)
        upper = max(np.percentile(syn, 99.5), np.percentile(rea, 99.5))
        lower = max(0.2, min(np.percentile(syn, 0.5), np.percentile(rea, 0.5)))
        bins = np.linspace(lower, upper, 35)
        ax.hist(syn, bins=bins, density=True, histtype="step", linewidth=1.8, label="Synthetic", color="#2f6f9f")
        ax.hist(rea, bins=bins, density=True, histtype="step", linewidth=1.8, label="MIT-BIH", color="#c65353")
        ax.set_title(class_name.replace("_", " "), fontsize=10)
        ax.text(
            0.98,
            0.95,
            f"KS p={row['ks_p_value']:.2e}\nW={row['wasserstein_distance_s']:.3f} s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_xlabel("RR interval (s)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "rr_distribution_comparison.png", dpi=300)
    plt.close(fig)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(results_dir / "rr_distribution_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
