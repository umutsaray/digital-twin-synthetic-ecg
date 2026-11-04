"""Compare this repository's generator and ECGSYN against shared real references."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_pqrst_timing import (  # noqa: E402
    INTERVALS,
    extract_ludb,
    extract_synthetic,
    ludb_root,
    map_records,
)


CLASSES = ["normal_sinus_rhythm", "bradycardia", "tachycardia"]
SEED = 7373
RECORDS_PER_CLASS = 500
FS = 250
DURATION_S = 10
HR_RANGES = {
    "normal_sinus_rhythm": (62.0, 88.0),
    "bradycardia": (36.0, 55.0),
    "tachycardia": (105.0, 150.0),
}


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def delineated_intervals(waves: dict[str, list[float]]) -> list[dict[str, float]]:
    """Return complete positive P/PR/QRS/QT/ST/T measurements by beat index."""
    keys = {
        "p_on": "ECG_P_Onsets",
        "p_off": "ECG_P_Offsets",
        "q_on": "ECG_R_Onsets",
        "q_off": "ECG_R_Offsets",
        "t_on": "ECG_T_Onsets",
        "t_off": "ECG_T_Offsets",
    }
    count = min(len(waves.get(key, [])) for key in keys.values())
    rows: list[dict[str, float]] = []
    for index in range(count):
        points = {name: finite(waves[key][index]) for name, key in keys.items()}
        if any(value is None for value in points.values()):
            continue
        p_on, p_off = points["p_on"], points["p_off"]
        q_on, q_off = points["q_on"], points["q_off"]
        t_on, t_off = points["t_on"], points["t_off"]
        values = {
            "p_duration": (p_off - p_on) / FS,
            "pr_interval": (q_on - p_on) / FS,
            "qrs_duration": (q_off - q_on) / FS,
            "qt_interval": (t_off - q_on) / FS,
            "st_segment": (t_on - q_off) / FS,
            "t_duration": (t_off - t_on) / FS,
        }
        if all(value > 0 for value in values.values()):
            rows.append(values)
    return rows


def generate_ecgsyn() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rr_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        for record_id in range(RECORDS_PER_CLASS):
            heart_rate = float(rng.uniform(*HR_RANGES[class_name]))
            record_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            signal = nk.ecg_simulate(
                duration=DURATION_S,
                sampling_rate=FS,
                heart_rate=heart_rate,
                heart_rate_std=1,
                method="ecgsyn",
                noise=0.01,
                random_state=record_seed,
            )
            cleaned = nk.ecg_clean(signal, sampling_rate=FS, method="neurokit")
            _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=FS, method="neurokit")
            peaks = np.asarray(peak_info["ECG_R_Peaks"], dtype=float)
            for rr in np.diff(peaks) / FS:
                rr_rows.append({"class_name": class_name, "rr_interval_s": float(rr)})
            complete = []
            if len(peaks) >= 3:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, waves = nk.ecg_delineate(
                            cleaned,
                            rpeaks=peaks.astype(int),
                            sampling_rate=FS,
                            method="dwt",
                            check=False,
                        )
                    complete = delineated_intervals(waves)
                except (ValueError, IndexError, TypeError):
                    complete = []
            for beat_index, values in enumerate(complete):
                for interval_name, value in values.items():
                    timing_rows.append(
                        {
                            "class_name": class_name,
                            "record_id": record_id,
                            "beat_index": beat_index,
                            "interval_name": interval_name,
                            "value_s": value,
                        }
                    )
            quality_rows.append(
                {
                    "class_name": class_name,
                    "record_id": record_id,
                    "sampled_heart_rate_bpm": heart_rate,
                    "detected_r_peaks": len(peaks),
                    "complete_delineated_beats": len(complete),
                }
            )
    return pd.DataFrame(rr_rows), pd.DataFrame(timing_rows), pd.DataFrame(quality_rows)


def pair_stats(left: np.ndarray, right: np.ndarray, prefix: str) -> dict[str, float | int]:
    ks = ks_2samp(left, right, alternative="two-sided", method="auto")
    return {
        f"{prefix}_ks_statistic": float(ks.statistic),
        f"{prefix}_ks_p_value": float(ks.pvalue),
        f"{prefix}_wasserstein_s": float(wasserstein_distance(left, right)),
    }


def describe(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    q1, q3 = np.percentile(values, [25, 75])
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean_s": float(np.mean(values)),
        f"{prefix}_std_s": float(np.std(values, ddof=1)),
        f"{prefix}_median_s": float(np.median(values)),
        f"{prefix}_iqr_s": float(q3 - q1),
    }


def rr_comparison(ecgsyn: pd.DataFrame) -> pd.DataFrame:
    ours = pd.read_csv(ROOT / "results" / "synthetic_rr_distributions.csv")
    real = pd.read_csv(ROOT / "results" / "real_mitdb_rr_distributions.csv")
    rows = []
    for class_name in CLASSES:
        own = ours.loc[ours.class_name == class_name, "rr_interval_s"].to_numpy()
        ecg = ecgsyn.loc[ecgsyn.class_name == class_name, "rr_interval_s"].to_numpy()
        rea = real.loc[real.class_name == class_name, "rr_interval_s"].to_numpy()
        rows.append(
            {"class_name": class_name, **describe(own, "our_generator"),
             **describe(ecg, "ecgsyn"), **describe(rea, "real"),
             **pair_stats(own, rea, "our_vs_real"),
             **pair_stats(ecg, rea, "ecgsyn_vs_real"),
             **pair_stats(own, ecg, "our_vs_ecgsyn")}
        )
    return pd.DataFrame(rows)


def timing_references() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate value arrays with the unchanged prior extractor and verify its CSV summary."""
    metadata = pd.read_csv(ludb_root() / "ludb.csv")
    mapping, _ = map_records(metadata)
    real = extract_ludb(ludb_root(), mapping)
    ours = extract_synthetic()
    prior = pd.read_csv(ROOT / "results" / "pqrst_timing_comparison.csv")
    for row in prior.itertuples(index=False):
        rea = real.loc[(real.class_name == row.class_name) & (real.interval_name == row.interval_name), "value_s"]
        own = ours.loc[(ours.class_name == row.class_name) & (ours.interval_name == row.interval_name), "value_s"]
        if len(rea) != row.real_n or len(own) != row.synthetic_n:
            raise RuntimeError("Recreated timing reference counts do not match the immutable prior CSV.")
        if not np.isclose(rea.mean(), row.real_mean_s) or not np.isclose(own.mean(), row.synthetic_mean_s):
            raise RuntimeError("Recreated timing references do not match the immutable prior CSV.")
    return ours, real


def timing_comparison(ecgsyn: pd.DataFrame) -> pd.DataFrame:
    ours, real = timing_references()
    rows = []
    for class_name in CLASSES:
        for interval_name in INTERVALS:
            own = ours.loc[(ours.class_name == class_name) & (ours.interval_name == interval_name), "value_s"].to_numpy()
            ecg = ecgsyn.loc[(ecgsyn.class_name == class_name) & (ecgsyn.interval_name == interval_name), "value_s"].to_numpy()
            rea = real.loc[(real.class_name == class_name) & (real.interval_name == interval_name), "value_s"].to_numpy()
            rows.append(
                {"class_name": class_name, "interval_name": interval_name,
                 **describe(own, "our_generator"), **describe(ecg, "ecgsyn"), **describe(rea, "real"),
                 **pair_stats(own, rea, "our_vs_real"),
                 **pair_stats(ecg, rea, "ecgsyn_vs_real"),
                 **pair_stats(own, ecg, "our_vs_ecgsyn")}
            )
    return pd.DataFrame(rows)


def summary_figure(rr: pd.DataFrame, timing: pd.DataFrame, output: Path) -> None:
    colors = ["#2f6f9f", "#d28b37"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), gridspec_kw={"height_ratios": [1, 2.2]})
    x = np.arange(len(rr))
    width = 0.35
    axes[0].bar(x - width / 2, rr.our_vs_real_wasserstein_s, width, label="Our generator", color=colors[0])
    axes[0].bar(x + width / 2, rr.ecgsyn_vs_real_wasserstein_s, width, label="ECGSYN", color=colors[1])
    axes[0].set_xticks(x, [name.replace("_", " ") for name in rr.class_name])
    axes[0].set_ylabel("Wasserstein distance (s)")
    axes[0].set_title("RR intervals vs. MIT-BIH")
    axes[0].legend(frameon=False, ncol=2)
    timing = timing.copy()
    timing["label"] = timing.class_name.str.replace("_", " ") + "\n" + timing.interval_name.str.replace("_", " ")
    x = np.arange(len(timing))
    axes[1].bar(x - width / 2, timing.our_vs_real_wasserstein_s, width, color=colors[0])
    axes[1].bar(x + width / 2, timing.ecgsyn_vs_real_wasserstein_s, width, color=colors[1])
    axes[1].set_xticks(x, timing.label, rotation=55, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Wasserstein distance (s)")
    axes[1].set_title("P/QRS/T timing vs. LUDB")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    rr_values, timing_values, quality = generate_ecgsyn()
    rr = rr_comparison(rr_values)
    timing = timing_comparison(timing_values)
    rr.to_csv(results / "ecgsyn_rr_comparison.csv", index=False)
    timing.to_csv(results / "ecgsyn_pqrst_comparison.csv", index=False)
    quality.to_csv(results / "ecgsyn_generation_quality.csv", index=False)
    summary_figure(rr, timing, figures / "ecgsyn_comparison_summary.png")
    print("Generation quality by class:")
    print(quality.groupby("class_name")[["detected_r_peaks", "complete_delineated_beats"]].sum())
    print("\nRR comparison:")
    print(rr[["class_name", "our_vs_real_wasserstein_s", "ecgsyn_vs_real_wasserstein_s", "our_vs_ecgsyn_wasserstein_s"]].to_string(index=False))
    print("\nTiming comparison:")
    print(timing[["class_name", "interval_name", "our_vs_real_wasserstein_s", "ecgsyn_vs_real_wasserstein_s", "our_vs_ecgsyn_wasserstein_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
