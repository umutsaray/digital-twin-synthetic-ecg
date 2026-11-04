"""Synthetic ECG dataset generation using mathematical morphology.

The generator creates six rhythm / artifact classes with realistic segment-level
variation while keeping the process deterministic under a fixed random seed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .signal_quality import compute_signal_quality_metrics, save_signal_quality_outputs
except ImportError:
    from signal_quality import compute_signal_quality_metrics, save_signal_quality_outputs  # type: ignore


CLASS_NAMES = [
    "normal_sinus_rhythm",
    "bradycardia",
    "tachycardia",
    "arrhythmia",
    "noisy_ecg",
    "sensor_fault_ecg",
]


@dataclass(frozen=True)
class ClassConfig:
    hr_range: Tuple[float, float]
    hrv_range: Tuple[float, float]
    gaussian_noise: Tuple[float, float]
    baseline_wander: Tuple[float, float]
    powerline_noise: Tuple[float, float]
    motion_probability: float
    clipping_probability: float
    missing_probability: float
    amplitude_range: Tuple[float, float] = (0.85, 1.20)
    irregular: bool = False
    sensor_fault: bool = False


CLASS_CONFIGS: Dict[str, ClassConfig] = {
    "normal_sinus_rhythm": ClassConfig(
        hr_range=(62, 88),
        hrv_range=(0.015, 0.055),
        gaussian_noise=(0.004, 0.018),
        baseline_wander=(0.015, 0.065),
        powerline_noise=(0.000, 0.012),
        motion_probability=0.10,
        clipping_probability=0.01,
        missing_probability=0.01,
    ),
    "bradycardia": ClassConfig(
        hr_range=(36, 55),
        hrv_range=(0.015, 0.060),
        gaussian_noise=(0.004, 0.020),
        baseline_wander=(0.015, 0.070),
        powerline_noise=(0.000, 0.014),
        motion_probability=0.12,
        clipping_probability=0.01,
        missing_probability=0.01,
    ),
    "tachycardia": ClassConfig(
        hr_range=(105, 150),
        hrv_range=(0.015, 0.070),
        gaussian_noise=(0.005, 0.024),
        baseline_wander=(0.020, 0.080),
        powerline_noise=(0.000, 0.018),
        motion_probability=0.18,
        clipping_probability=0.02,
        missing_probability=0.01,
    ),
    "arrhythmia": ClassConfig(
        hr_range=(58, 104),
        hrv_range=(0.110, 0.260),
        gaussian_noise=(0.006, 0.026),
        baseline_wander=(0.020, 0.090),
        powerline_noise=(0.000, 0.018),
        motion_probability=0.22,
        clipping_probability=0.02,
        missing_probability=0.02,
        irregular=True,
    ),
    "noisy_ecg": ClassConfig(
        hr_range=(60, 96),
        hrv_range=(0.030, 0.095),
        gaussian_noise=(0.060, 0.160),
        baseline_wander=(0.070, 0.220),
        powerline_noise=(0.020, 0.080),
        motion_probability=0.82,
        clipping_probability=0.08,
        missing_probability=0.07,
        amplitude_range=(0.70, 1.35),
    ),
    "sensor_fault_ecg": ClassConfig(
        hr_range=(58, 98),
        hrv_range=(0.025, 0.080),
        gaussian_noise=(0.006, 0.030),
        baseline_wander=(0.020, 0.100),
        powerline_noise=(0.000, 0.025),
        motion_probability=0.28,
        clipping_probability=0.65,
        missing_probability=0.70,
        amplitude_range=(0.60, 1.45),
        sensor_fault=True,
    ),
}


def seed_everything(seed: int) -> np.random.Generator:
    """Return a deterministic NumPy random generator."""
    return np.random.default_rng(seed)


def _gaussian(rel_t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((rel_t - center) / width) ** 2)


def _beat_morphology(class_name: str, beat_kind: str, rng: np.random.Generator) -> List[Tuple[float, float, float]]:
    """Return Gaussian PQRST components as (center seconds, width seconds, amplitude)."""
    if beat_kind == "ectopic":
        return [
            (-0.10, 0.050, 0.02 * rng.uniform(0.5, 1.2)),
            (-0.030, 0.028, -0.10 * rng.uniform(0.7, 1.2)),
            (0.000, 0.035, 0.72 * rng.uniform(0.75, 1.25)),
            (0.065, 0.045, -0.38 * rng.uniform(0.8, 1.4)),
            (0.310, 0.110, -0.18 * rng.uniform(0.6, 1.4)),
        ]

    p_amp = 0.11
    r_amp = 1.00
    t_amp = 0.34
    qrs_width = 1.0

    if class_name == "tachycardia":
        p_amp *= 0.82
        t_amp *= 0.78
    elif class_name == "bradycardia":
        p_amp *= 1.05
        t_amp *= 1.10
    elif class_name == "arrhythmia":
        p_amp *= rng.uniform(0.70, 1.25)
        r_amp *= rng.uniform(0.82, 1.15)
        t_amp *= rng.uniform(0.70, 1.25)
        qrs_width *= rng.uniform(0.85, 1.30)

    return [
        (-0.205 + rng.normal(0.0, 0.006), 0.040, p_amp * rng.uniform(0.85, 1.15)),
        (-0.045 + rng.normal(0.0, 0.004), 0.014 * qrs_width, -0.15 * rng.uniform(0.80, 1.20)),
        (0.000 + rng.normal(0.0, 0.003), 0.012 * qrs_width, r_amp * rng.uniform(0.88, 1.12)),
        (0.042 + rng.normal(0.0, 0.004), 0.020 * qrs_width, -0.25 * rng.uniform(0.80, 1.20)),
        (0.285 + rng.normal(0.0, 0.012), 0.082, t_amp * rng.uniform(0.85, 1.15)),
    ]


def _generate_beat_train(
    duration: float,
    heart_rate: float,
    hrv: float,
    irregular: bool,
    rng: np.random.Generator,
) -> Tuple[List[float], List[str]]:
    base_rr = 60.0 / heart_rate
    t = rng.uniform(0.35, min(1.05, base_rr + 0.35))
    beat_times: List[float] = []
    beat_kinds: List[str] = []
    next_beat_kind = "normal"
    pause_pending = False

    while t < duration + 0.7:
        beat_kind = next_beat_kind
        next_beat_kind = "normal"
        beat_times.append(float(t))
        beat_kinds.append(beat_kind)

        rr_scale = 1.0 + rng.normal(0.0, hrv)

        if irregular:
            if pause_pending:
                # This is the interval *after* an ectopic beat.  Consume the
                # pause exactly one iteration after the short coupling
                # interval that led into the ectopic beat.
                rr_scale *= rng.uniform(1.20, 1.65)
                pause_pending = False
            else:
                # Decide the type of the upcoming beat.  Any short coupling
                # interval therefore ends at the beat labelled "ectopic",
                # rather than incorrectly starting at it.
                event = rng.random()
                if event < 0.14:
                    rr_scale *= rng.uniform(0.45, 0.72)
                    next_beat_kind = "ectopic"
                    pause_pending = True
                elif event < 0.22:
                    rr_scale *= rng.uniform(1.35, 1.85)
                elif event < 0.30:
                    rr_scale *= rng.uniform(0.72, 1.35)

        rr = np.clip(base_rr * rr_scale, 0.30, 2.20)
        t += float(rr)

    return beat_times, beat_kinds


def _synthesize_clean_ecg(
    class_name: str,
    duration: float,
    sampling_rate: int,
    heart_rate: float,
    hrv: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n_samples = int(duration * sampling_rate)
    signal = np.zeros(n_samples, dtype=np.float32)
    beat_times, beat_kinds = _generate_beat_train(
        duration=duration,
        heart_rate=heart_rate,
        hrv=hrv,
        irregular=CLASS_CONFIGS[class_name].irregular,
        rng=rng,
    )

    ectopic_beats = 0
    for beat_time, beat_kind in zip(beat_times, beat_kinds):
        if beat_kind == "ectopic":
            ectopic_beats += 1
        start = max(0, int((beat_time - 0.45) * sampling_rate))
        end = min(n_samples, int((beat_time + 0.62) * sampling_rate))
        if end <= start:
            continue
        rel_t = np.arange(start, end, dtype=np.float32) / sampling_rate - beat_time
        beat = np.zeros_like(rel_t, dtype=np.float32)
        beat_scale = rng.uniform(0.92, 1.08)
        for center, width, amplitude in _beat_morphology(class_name, beat_kind, rng):
            beat += _gaussian(rel_t, center, width, amplitude * beat_scale).astype(np.float32)
        signal[start:end] += beat

    return signal, {
        "num_beats": len([b for b in beat_times if 0 <= b < duration]),
        "ectopic_beats": ectopic_beats,
    }


def _add_baseline_wander(
    signal: np.ndarray,
    sampling_rate: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(signal.size, dtype=np.float32) / sampling_rate
    f1 = rng.uniform(0.12, 0.34)
    f2 = rng.uniform(0.035, 0.085)
    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    wander = amplitude * np.sin(2.0 * np.pi * f1 * t + phase1)
    wander += 0.45 * amplitude * np.sin(2.0 * np.pi * f2 * t + phase2)
    return signal + wander.astype(np.float32)


def _add_powerline_noise(
    signal: np.ndarray,
    sampling_rate: int,
    amplitude: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    frequency = float(rng.choice([50, 60]))
    t = np.arange(signal.size, dtype=np.float32) / sampling_rate
    phase = rng.uniform(0.0, 2.0 * np.pi)
    interference = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
    return signal + interference.astype(np.float32), frequency


def _add_motion_artifact(
    signal: np.ndarray,
    sampling_rate: int,
    rng: np.random.Generator,
    intensity: float,
) -> Tuple[np.ndarray, int]:
    n_samples = signal.size
    n_bursts = int(rng.integers(1, 4))
    output = signal.copy()

    for _ in range(n_bursts):
        burst_len = int(rng.uniform(0.35, 1.6) * sampling_rate)
        if burst_len >= n_samples:
            burst_len = n_samples - 1
        start = int(rng.integers(0, max(1, n_samples - burst_len)))
        end = start + burst_len
        local_t = np.arange(burst_len, dtype=np.float32) / sampling_rate
        envelope = np.hanning(burst_len).astype(np.float32)
        low_freq = rng.uniform(0.8, 4.5)
        artifact = np.sin(2.0 * np.pi * low_freq * local_t + rng.uniform(0.0, 2.0 * np.pi))
        artifact += 0.45 * rng.standard_normal(burst_len)
        output[start:end] += (intensity * envelope * artifact).astype(np.float32)

    return output, n_bursts


def _apply_missing_segments(
    signal: np.ndarray,
    sampling_rate: int,
    rng: np.random.Generator,
    severe: bool = False,
) -> Tuple[np.ndarray, int]:
    output = signal.copy()
    n_segments = int(rng.integers(1, 4 if severe else 3))
    n_samples = signal.size

    for _ in range(n_segments):
        length_sec = rng.uniform(0.06, 0.45 if severe else 0.22)
        segment_len = max(2, int(length_sec * sampling_rate))
        start = int(rng.integers(0, max(1, n_samples - segment_len)))
        end = start + segment_len
        fill_mode = rng.choice(["zero", "hold", "nan_safe"])
        if fill_mode == "hold" and start > 0:
            output[start:end] = output[start - 1]
        elif fill_mode == "nan_safe":
            output[start:end] = np.median(output[max(0, start - 50) : start + 1])
        else:
            output[start:end] = 0.0

    return output, n_segments


def _apply_clipping(
    signal: np.ndarray,
    rng: np.random.Generator,
    severe: bool = False,
) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(signal)) + 1e-6)
    threshold = max_abs * rng.uniform(0.42, 0.75 if severe else 0.92)
    return np.clip(signal, -threshold, threshold).astype(np.float32), threshold


def _apply_sensor_faults(
    signal: np.ndarray,
    sampling_rate: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    output = signal.copy()
    fault_count = 0
    flatline_segments = 0
    spike_count = 0

    if rng.random() < 0.72:
        output, missing_segments = _apply_missing_segments(output, sampling_rate, rng, severe=True)
        fault_count += missing_segments
    else:
        missing_segments = 0

    if rng.random() < 0.72:
        output, clip_threshold = _apply_clipping(output, rng, severe=True)
        fault_count += 1
    else:
        clip_threshold = np.nan

    if rng.random() < 0.42:
        n_samples = output.size
        segment_len = int(rng.uniform(0.30, 1.20) * sampling_rate)
        start = int(rng.integers(0, max(1, n_samples - segment_len)))
        end = start + segment_len
        output[start:end] = output[start] if start < output.size else 0.0
        flatline_segments = 1
        fault_count += 1

    if rng.random() < 0.55:
        spike_count = int(rng.integers(2, 9))
        positions = rng.integers(0, output.size, size=spike_count)
        signs = rng.choice([-1.0, 1.0], size=spike_count)
        output[positions] += signs * rng.uniform(1.2, 2.6, size=spike_count)
        fault_count += spike_count

    if rng.random() < 0.32:
        start = int(rng.integers(0, output.size))
        output[start:] += rng.uniform(-0.55, 0.55)
        fault_count += 1

    return output.astype(np.float32), {
        "sensor_fault_count": fault_count,
        "sensor_missing_segments": missing_segments,
        "flatline_segments": flatline_segments,
        "spike_count": spike_count,
        "sensor_clip_threshold": clip_threshold,
    }


def generate_ecg_segment(
    class_name: str,
    duration: float,
    sampling_rate: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    config = CLASS_CONFIGS[class_name]
    heart_rate = float(rng.uniform(*config.hr_range))
    hrv = float(rng.uniform(*config.hrv_range))
    signal, morphology_meta = _synthesize_clean_ecg(
        class_name=class_name,
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        hrv=hrv,
        rng=rng,
    )

    amplitude_scale = float(rng.uniform(*config.amplitude_range))
    signal *= amplitude_scale
    clean_reference = signal.copy()

    baseline_amp = float(rng.uniform(*config.baseline_wander))
    signal = _add_baseline_wander(signal, sampling_rate, baseline_amp, rng)

    gaussian_std = float(rng.uniform(*config.gaussian_noise))
    signal += rng.normal(0.0, gaussian_std, size=signal.size).astype(np.float32)

    powerline_amp = float(rng.uniform(*config.powerline_noise))
    signal, powerline_frequency = _add_powerline_noise(signal, sampling_rate, powerline_amp, rng)

    motion_bursts = 0
    if rng.random() < config.motion_probability:
        motion_intensity = float(rng.uniform(0.05, 0.22 if class_name != "noisy_ecg" else 0.55))
        signal, motion_bursts = _add_motion_artifact(signal, sampling_rate, rng, motion_intensity)
    else:
        motion_intensity = 0.0

    missing_segments = 0
    if rng.random() < config.missing_probability:
        signal, missing_segments = _apply_missing_segments(
            signal,
            sampling_rate,
            rng,
            severe=class_name in {"noisy_ecg", "sensor_fault_ecg"},
        )

    clipped = 0
    clip_threshold = np.nan
    if rng.random() < config.clipping_probability:
        signal, clip_threshold = _apply_clipping(
            signal,
            rng,
            severe=class_name == "sensor_fault_ecg",
        )
        clipped = 1

    sensor_meta = {
        "sensor_fault_count": 0,
        "sensor_missing_segments": 0,
        "flatline_segments": 0,
        "spike_count": 0,
        "sensor_clip_threshold": np.nan,
    }
    if config.sensor_fault:
        signal, sensor_meta = _apply_sensor_faults(signal, sampling_rate, rng)
        missing_segments += int(sensor_meta["sensor_missing_segments"])
        if not np.isnan(sensor_meta["sensor_clip_threshold"]):
            clipped = 1

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    quality_metrics = compute_signal_quality_metrics(clean_reference, signal)
    meta = {
        "heart_rate_bpm": heart_rate,
        "hrv_fraction": hrv,
        "amplitude_scale": amplitude_scale,
        "baseline_wander_amplitude": baseline_amp,
        "gaussian_noise_std": gaussian_std,
        "powerline_amplitude": powerline_amp,
        "powerline_frequency_hz": powerline_frequency,
        "motion_artifact_bursts": motion_bursts,
        "motion_artifact_intensity": motion_intensity,
        "missing_segments": missing_segments,
        "clipped": clipped,
        "clip_threshold": clip_threshold,
        **morphology_meta,
        **sensor_meta,
    }
    return signal, meta, quality_metrics


def _plot_example_signals(
    data: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    sampling_rate: int,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(data.shape[1]) / sampling_rate
    fig, axes = plt.subplots(len(class_names), 1, figsize=(14, 11), sharex=True)
    for class_id, class_name in enumerate(class_names):
        idx = int(np.flatnonzero(labels == class_id)[0])
        axes[class_id].plot(t, data[idx], linewidth=0.9, color="#1f77b4")
        axes[class_id].set_ylabel(class_name.replace("_", "\n"), fontsize=10)
        axes[class_id].grid(alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Synthetic ECG examples by class", y=0.995)
    fig.tight_layout()
    fig.savefig(figures_dir / "example_signals.png", dpi=300)
    plt.close(fig)


def _plot_class_distribution(labels: np.ndarray, class_names: List[str], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2f6f9f", "#559e83", "#d6a01d", "#c65353", "#7b5aa6", "#5c677d"]
    ax.bar([name.replace("_", "\n") for name in class_names], counts, color=colors)
    ax.set_ylabel("Segments")
    ax.set_title("Class distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=300)
    plt.close(fig)


def generate_dataset(
    data_dir: str | Path = "data",
    figures_dir: str | Path = "figures",
    results_dir: str | Path = "results",
    samples_per_class: int = 2000,
    duration: float = 10.0,
    sampling_rate: int = 250,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, Path]:
    """Generate and save the complete synthetic ECG dataset."""
    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    data_path = data_dir / "synthetic_ecg_dataset.npy"
    labels_path = data_dir / "labels.csv"
    metadata_path = data_dir / "metadata.csv"
    quality_path = results_dir / "signal_quality_metrics.csv"
    quality_summary_path = results_dir / "signal_quality_summary.csv"

    if (
        not force
        and data_path.exists()
        and labels_path.exists()
        and metadata_path.exists()
        and quality_path.exists()
        and quality_summary_path.exists()
    ):
        print(f"Dataset already exists at {data_path}. Use force=True to regenerate.")
        return {
            "data": data_path,
            "labels": labels_path,
            "metadata": metadata_path,
            "signal_quality_metrics": quality_path,
            "signal_quality_summary": quality_summary_path,
            "example_signals": figures_dir / "example_signals.png",
            "class_distribution": figures_dir / "class_distribution.png",
            "snr_by_class": figures_dir / "snr_by_class.png",
            "prd_by_class": figures_dir / "prd_by_class.png",
            "rmse_by_class": figures_dir / "rmse_by_class.png",
        }
    if not force and data_path.exists() and labels_path.exists() and metadata_path.exists():
        print(
            "Existing dataset found, but clean-reference signal quality metrics are missing. "
            "Regenerating dataset to compute SNR, PRD, and RMSE."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(seed)

    n_classes = len(CLASS_NAMES)
    n_samples = n_classes * samples_per_class
    segment_length = int(duration * sampling_rate)
    data = np.empty((n_samples, segment_length), dtype=np.float32)
    labels = np.empty(n_samples, dtype=np.int64)
    label_rows = []
    metadata_rows = []
    quality_rows = []

    row = 0
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"Generating {samples_per_class} segments for {class_name}...")
        for local_idx in range(samples_per_class):
            segment, meta, quality_metrics = generate_ecg_segment(class_name, duration, sampling_rate, rng)
            sample_id = f"{class_name}_{local_idx:05d}"
            data[row] = segment
            labels[row] = class_id
            label_rows.append(
                {
                    "sample_index": row,
                    "sample_id": sample_id,
                    "label_id": class_id,
                    "label": class_name,
                }
            )
            metadata_rows.append(
                {
                    "sample_index": row,
                    "sample_id": sample_id,
                    "label_id": class_id,
                    "label": class_name,
                    "duration_seconds": duration,
                    "sampling_rate_hz": sampling_rate,
                    **meta,
                }
            )
            quality_rows.append(
                {
                    "sample_index": row,
                    "sample_id": sample_id,
                    "label_id": class_id,
                    "label": class_name,
                    **quality_metrics,
                }
            )
            row += 1

    permutation = rng.permutation(n_samples)
    data = data[permutation]
    labels = labels[permutation]

    labels_df = pd.DataFrame(label_rows).iloc[permutation].reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata_rows).iloc[permutation].reset_index(drop=True)
    quality_df = pd.DataFrame(quality_rows).iloc[permutation].reset_index(drop=True)
    labels_df["sample_index"] = np.arange(n_samples)
    metadata_df["sample_index"] = np.arange(n_samples)
    quality_df["sample_index"] = np.arange(n_samples)

    np.save(data_path, data)
    labels_df.to_csv(labels_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)
    _plot_example_signals(data, labels, CLASS_NAMES, sampling_rate, figures_dir)
    _plot_class_distribution(labels, CLASS_NAMES, figures_dir)
    quality_paths = save_signal_quality_outputs(quality_df, results_dir, figures_dir, CLASS_NAMES)

    print(f"Saved dataset: {data_path} with shape {data.shape}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved signal quality metrics: {quality_paths['signal_quality_metrics']}")
    return {
        "data": data_path,
        "labels": labels_path,
        "metadata": metadata_path,
        **quality_paths,
        "example_signals": figures_dir / "example_signals.png",
        "class_distribution": figures_dir / "class_distribution.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ECG dataset.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        data_dir=args.data_dir,
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        samples_per_class=args.samples_per_class,
        duration=args.duration,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
        force=args.force,
    )
