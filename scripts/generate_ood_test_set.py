"""Generate a disjoint OOD ECG test set with unseen parameter combinations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_dataset import (
    CLASS_CONFIGS,
    CLASS_NAMES,
    _add_baseline_wander,
    _add_motion_artifact,
    _add_powerline_noise,
    _apply_clipping,
    _apply_missing_segments,
    _apply_sensor_faults,
    _synthesize_clean_ecg,
)


ARTIFACT_DONOR = {
    "normal_sinus_rhythm": "noisy_ecg",
    "bradycardia": "sensor_fault_ecg",
    "tachycardia": "noisy_ecg",
    "arrhythmia": "sensor_fault_ecg",
    "noisy_ecg": "tachycardia",
    "sensor_fault_ecg": "normal_sinus_rhythm",
}


def tail_range(class_name: str, upper: bool) -> tuple[float, float]:
    low, high = CLASS_CONFIGS[class_name].hr_range
    extension = 0.15 * (high - low)
    return (high, high + extension) if upper else (low - extension, low)


def generate_ood_segment(
    class_name: str,
    artifact_profile: str,
    hr_range: tuple[float, float],
    duration: float,
    sampling_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep target rhythm/morphology but use the donor's full artifact path."""
    target = CLASS_CONFIGS[class_name]
    donor = CLASS_CONFIGS[artifact_profile]
    heart_rate = float(rng.uniform(*hr_range))
    hrv = float(rng.uniform(*target.hrv_range))
    signal, morphology_meta = _synthesize_clean_ecg(
        class_name, duration, sampling_rate, heart_rate, hrv, rng
    )
    amplitude_scale = float(rng.uniform(*donor.amplitude_range))
    signal *= amplitude_scale
    baseline_amp = float(rng.uniform(*donor.baseline_wander))
    signal = _add_baseline_wander(signal, sampling_rate, baseline_amp, rng)
    gaussian_std = float(rng.uniform(*donor.gaussian_noise))
    signal += rng.normal(0.0, gaussian_std, size=signal.size).astype(np.float32)
    powerline_amp = float(rng.uniform(*donor.powerline_noise))
    signal, powerline_frequency = _add_powerline_noise(
        signal, sampling_rate, powerline_amp, rng
    )

    motion_bursts = 0
    motion_intensity = 0.0
    if rng.random() < donor.motion_probability:
        motion_intensity = float(
            rng.uniform(0.05, 0.55 if artifact_profile == "noisy_ecg" else 0.22)
        )
        signal, motion_bursts = _add_motion_artifact(
            signal, sampling_rate, rng, motion_intensity
        )
    missing_segments = 0
    if rng.random() < donor.missing_probability:
        signal, missing_segments = _apply_missing_segments(
            signal,
            sampling_rate,
            rng,
            severe=artifact_profile in {"noisy_ecg", "sensor_fault_ecg"},
        )
    clipped = 0
    clip_threshold = np.nan
    if rng.random() < donor.clipping_probability:
        signal, clip_threshold = _apply_clipping(
            signal, rng, severe=artifact_profile == "sensor_fault_ecg"
        )
        clipped = 1
    sensor_meta = {
        "sensor_fault_count": 0,
        "sensor_missing_segments": 0,
        "flatline_segments": 0,
        "spike_count": 0,
        "sensor_clip_threshold": np.nan,
    }
    if donor.sensor_fault:
        signal, sensor_meta = _apply_sensor_faults(signal, sampling_rate, rng)
        missing_segments += int(sensor_meta["sensor_missing_segments"])
        if not np.isnan(sensor_meta["sensor_clip_threshold"]):
            clipped = 1
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return signal, {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output-dir", default="data/ood")
    args = parser.parse_args()

    if args.samples_per_class % 2:
        raise ValueError("samples-per-class must be even for balanced lower/upper tails.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    signals: list[np.ndarray] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        donor_name = ARTIFACT_DONOR[class_name]
        for local_idx in range(args.samples_per_class):
            upper_tail = local_idx % 2 == 1
            cross_applied = local_idx >= args.samples_per_class // 2
            profile_name = donor_name if cross_applied else class_name
            ood_hr_range = tail_range(class_name, upper_tail)
            signal, metadata = generate_ood_segment(
                class_name,
                profile_name,
                ood_hr_range,
                args.duration,
                args.sampling_rate,
                rng,
            )
            signals.append(signal)
            rows.append(
                {
                    "sample_index": len(signals) - 1,
                    "sample_id": f"ood_{class_name}_{local_idx:03d}",
                    "label_id": class_id,
                    "label": class_name,
                    "seed": args.seed,
                    "hr_tail": "upper" if upper_tail else "lower",
                    "ood_hr_min_bpm": ood_hr_range[0],
                    "ood_hr_max_bpm": ood_hr_range[1],
                    "artifact_profile": profile_name,
                    "artifact_cross_applied": cross_applied,
                    **metadata,
                }
            )

    data = np.stack(signals).astype(np.float32)
    metadata = pd.DataFrame(rows)
    np.save(output_dir / "ood_ecg_test_600.npy", data)
    metadata[["sample_index", "sample_id", "label_id", "label"]].to_csv(
        output_dir / "ood_labels.csv", index=False
    )
    metadata.to_csv(output_dir / "ood_metadata.csv", index=False)
    print(f"Saved OOD data {data.shape} to {output_dir}")
    print(metadata.groupby(["label", "artifact_cross_applied"]).size().to_string())


if __name__ == "__main__":
    main()
