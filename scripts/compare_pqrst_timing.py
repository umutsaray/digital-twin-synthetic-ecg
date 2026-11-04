"""Compare synthetic Gaussian-template timing with manual LUDB Lead-II timing."""

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

from src.generate_dataset import CLASS_CONFIGS, _beat_morphology, _generate_beat_train


CLASSES = ["normal_sinus_rhythm", "bradycardia", "tachycardia", "arrhythmia"]
INTERVALS = ["p_duration", "pr_interval", "qrs_duration", "qt_interval", "st_segment", "t_duration"]
PEAK_TO_WAVE = {"p": "p", "N": "qrs", "t": "t"}


def ludb_root() -> Path:
    candidates = list((ROOT / "data" / "ludb").glob("*/ludb.csv"))
    if not candidates:
        raise FileNotFoundError("Extract LUDB 1.0.1 under data/ludb/ before running this script.")
    return candidates[0].parent


def map_records(metadata: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    mapping: dict[int, str] = {}
    excluded = {"atrial_fibrillation_or_flutter": 0, "unmapped_rhythm": 0}
    for row in metadata.itertuples(index=False):
        record_id = int(row.ID)
        rhythm = str(row.Rhythms).strip().lower()
        extras = "" if pd.isna(row.Extrasystolies) else str(row.Extrasystolies).strip()
        if rhythm.startswith("atrial fibrillation") or rhythm.startswith("atrial flutter"):
            excluded["atrial_fibrillation_or_flutter"] += 1
        elif extras:
            mapping[record_id] = "arrhythmia"
        elif rhythm.startswith("sinus bradycardia"):
            mapping[record_id] = "bradycardia"
        elif rhythm.startswith("sinus tachycardia"):
            mapping[record_id] = "tachycardia"
        elif rhythm.startswith("sinus rhythm"):
            mapping[record_id] = "normal_sinus_rhythm"
        else:
            excluded["unmapped_rhythm"] += 1
    return mapping, excluded


def wave_triplets(annotation: wfdb.Annotation) -> list[dict[str, float | str]]:
    waves: list[dict[str, float | str]] = []
    for idx in range(len(annotation.symbol) - 2):
        symbols = annotation.symbol[idx : idx + 3]
        if symbols[0] == "(" and symbols[1] in PEAK_TO_WAVE and symbols[2] == ")":
            waves.append(
                {
                    "wave": PEAK_TO_WAVE[symbols[1]],
                    "onset": float(annotation.sample[idx]),
                    "peak": float(annotation.sample[idx + 1]),
                    "offset": float(annotation.sample[idx + 2]),
                }
            )
    return waves


def complete_cycles(waves: list[dict[str, float | str]], fs: float) -> list[dict[str, float]]:
    qrs_waves = [wave for wave in waves if wave["wave"] == "qrs"]
    p_waves = [wave for wave in waves if wave["wave"] == "p"]
    t_waves = [wave for wave in waves if wave["wave"] == "t"]
    rows: list[dict[str, float]] = []
    for idx, qrs in enumerate(qrs_waves):
        previous_qrs_offset = float(qrs_waves[idx - 1]["offset"]) if idx else -np.inf
        next_qrs_onset = float(qrs_waves[idx + 1]["onset"]) if idx + 1 < len(qrs_waves) else np.inf
        preceding_p = [
            wave
            for wave in p_waves
            if previous_qrs_offset < float(wave["onset"])
            and float(wave["offset"]) < float(qrs["onset"])
        ]
        following_t = [
            wave
            for wave in t_waves
            if float(qrs["offset"]) < float(wave["onset"])
            and float(wave["offset"]) < next_qrs_onset
        ]
        if not preceding_p or not following_t:
            continue
        p_wave = max(preceding_p, key=lambda wave: float(wave["onset"]))
        t_wave = min(following_t, key=lambda wave: float(wave["onset"]))
        p_on, p_off = float(p_wave["onset"]) / fs, float(p_wave["offset"]) / fs
        q_on, q_off = float(qrs["onset"]) / fs, float(qrs["offset"]) / fs
        t_on, t_off = float(t_wave["onset"]) / fs, float(t_wave["offset"]) / fs
        values = {
            "p_duration": p_off - p_on,
            "pr_interval": q_on - p_on,
            "qrs_duration": q_off - q_on,
            "qt_interval": t_off - q_on,
            "st_segment": t_on - q_off,
            "t_duration": t_off - t_on,
        }
        if all(value > 0 for value in values.values()):
            rows.append(values)
    return rows


def extract_ludb(root: Path, mapping: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    data_dir = root / "data"
    for record_id, class_name in sorted(mapping.items()):
        base = data_dir / str(record_id)
        header = wfdb.rdheader(str(base))
        lead_names = [name.lower() for name in header.sig_name]
        if "ii" not in lead_names:
            continue
        lead_ii_index = lead_names.index("ii")
        signal = wfdb.rdrecord(str(base), channels=[lead_ii_index])
        annotation = wfdb.rdann(str(base), "ii")
        if signal.sig_len <= 0 or annotation.sample.size == 0:
            continue
        for beat_index, values in enumerate(complete_cycles(wave_triplets(annotation), float(signal.fs))):
            for interval_name, value in values.items():
                rows.append(
                    {
                        "source": "ludb",
                        "class_name": class_name,
                        "record_id": record_id,
                        "beat_index": beat_index,
                        "interval_name": interval_name,
                        "value_s": value,
                    }
                )
    return pd.DataFrame(rows)


def gaussian_boundaries(morphology: list[tuple[float, float, float]]) -> dict[str, float]:
    """Use a declared ±2 sigma envelope because Gaussians have no finite boundary."""
    p_center, p_width, _ = morphology[0]
    t_center, t_width, _ = morphology[4]
    qrs = morphology[1:4]
    p_on, p_off = p_center - 2 * p_width, p_center + 2 * p_width
    q_on = min(center - 2 * width for center, width, _ in qrs)
    q_off = max(center + 2 * width for center, width, _ in qrs)
    t_on, t_off = t_center - 2 * t_width, t_center + 2 * t_width
    return {
        "p_duration": p_off - p_on,
        "pr_interval": q_on - p_on,
        "qrs_duration": q_off - q_on,
        "qt_interval": t_off - q_on,
        "st_segment": t_on - q_off,
        "t_duration": t_off - t_on,
    }


def extract_synthetic(seed: int = 6262, records_per_class: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        config = CLASS_CONFIGS[class_name]
        for record_id in range(records_per_class):
            heart_rate = float(rng.uniform(*config.hr_range))
            hrv = float(rng.uniform(*config.hrv_range))
            beat_times, beat_kinds = _generate_beat_train(
                10.0, heart_rate, hrv, config.irregular, rng
            )
            for beat_index, (beat_time, beat_kind) in enumerate(zip(beat_times, beat_kinds)):
                morphology = _beat_morphology(class_name, beat_kind, rng)
                if not 0 <= beat_time < 10.0:
                    continue
                values = gaussian_boundaries(morphology)
                for interval_name, value in values.items():
                    rows.append(
                        {
                            "source": "synthetic",
                            "class_name": class_name,
                            "record_id": record_id,
                            "beat_index": beat_index,
                            "interval_name": interval_name,
                            "value_s": value,
                        }
                    )
    return pd.DataFrame(rows)


def comparison_row(class_name: str, interval_name: str, syn: np.ndarray, real: np.ndarray) -> dict[str, object]:
    syn_q1, syn_q3 = np.percentile(syn, [25, 75])
    real_q1, real_q3 = np.percentile(real, [25, 75])
    ks = ks_2samp(syn, real, alternative="two-sided", method="auto")
    return {
        "class_name": class_name,
        "interval_name": interval_name,
        "synthetic_n": len(syn),
        "synthetic_mean_s": syn.mean(),
        "synthetic_std_s": syn.std(ddof=1),
        "synthetic_median_s": np.median(syn),
        "synthetic_iqr_s": syn_q3 - syn_q1,
        "real_n": len(real),
        "real_mean_s": real.mean(),
        "real_std_s": real.std(ddof=1),
        "real_median_s": np.median(real),
        "real_iqr_s": real_q3 - real_q1,
        "ks_statistic": ks.statistic,
        "ks_p_value": ks.pvalue,
        "wasserstein_distance_s": wasserstein_distance(syn, real),
        "underpowered_real_n_lt_20": len(real) < 20,
    }


def plot_selected(comparison: pd.DataFrame, synthetic: pd.DataFrame, real: pd.DataFrame, output: Path) -> None:
    selected = ["qrs_duration", "qt_interval"]
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0))
    colors = {"synthetic": "#2f6f9f", "ludb": "#c65353"}
    for row_index, class_name in enumerate(CLASSES):
        for col_index, interval_name in enumerate(selected):
            ax = axes[row_index, col_index]
            syn = synthetic.loc[
                (synthetic.class_name == class_name) & (synthetic.interval_name == interval_name),
                "value_s",
            ].to_numpy()
            rea = real.loc[
                (real.class_name == class_name) & (real.interval_name == interval_name),
                "value_s",
            ].to_numpy()
            low = min(np.percentile(syn, 0.5), np.percentile(rea, 0.5))
            high = max(np.percentile(syn, 99.5), np.percentile(rea, 99.5))
            bins = np.linspace(max(0, low), high, 30)
            ax.hist(syn, bins=bins, density=True, histtype="step", linewidth=1.6, color=colors["synthetic"], label="Synthetic")
            ax.hist(rea, bins=bins, density=True, histtype="step", linewidth=1.6, color=colors["ludb"], label="LUDB")
            stats = comparison.loc[
                (comparison.class_name == class_name) & (comparison.interval_name == interval_name)
            ].iloc[0]
            p_label = "<1e-300" if stats.ks_p_value < 1e-300 else f"{stats.ks_p_value:.1e}"
            ax.text(
                0.98,
                0.95,
                f"KS p={p_label}\nW={stats.wasserstein_distance_s:.3f} s\nLUDB n={int(stats.real_n)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            if row_index == 0:
                ax.set_title(interval_name.replace("_", " "), fontsize=10)
            if col_index == 0:
                ax.set_ylabel(class_name.replace("_", " ") + "\nDensity", fontsize=8)
            ax.set_xlabel("Duration (s)", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    root = ludb_root()
    metadata = pd.read_csv(root / "ludb.csv")
    mapping, excluded = map_records(metadata)
    data_dir = root / "data"
    example = wfdb.rdann(str(data_dir / "1"), "ii")
    print("Lead-II annotation confirmation for record 1:")
    print("samples:", example.sample[:18].tolist())
    print("symbols:", example.symbol[:18])
    print("aux_note:", example.aux_note[:18])
    print("mapped records:", pd.Series(mapping).value_counts().to_dict())
    print("excluded:", excluded)

    real = extract_ludb(root, mapping)
    synthetic = extract_synthetic()
    rows = []
    for class_name in CLASSES:
        for interval_name in INTERVALS:
            syn = synthetic.loc[
                (synthetic.class_name == class_name) & (synthetic.interval_name == interval_name),
                "value_s",
            ].to_numpy()
            rea = real.loc[
                (real.class_name == class_name) & (real.interval_name == interval_name),
                "value_s",
            ].to_numpy()
            if len(syn) and len(rea):
                rows.append(comparison_row(class_name, interval_name, syn, rea))
    comparison = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "pqrst_timing_comparison.csv", index=False)
    plot_selected(comparison, synthetic, real, figures_dir / "pqrst_timing_comparison.png")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
