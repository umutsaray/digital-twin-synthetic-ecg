"""Regenerate only arrhythmia samples while preserving all other rows."""

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
    CLASS_NAMES,
    _plot_class_distribution,
    _plot_example_signals,
    generate_ecg_segment,
)
from src.signal_quality import save_signal_quality_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    data_path = data_dir / "synthetic_ecg_dataset.npy"
    labels_path = data_dir / "labels.csv"
    metadata_path = data_dir / "metadata.csv"
    quality_path = results_dir / "signal_quality_metrics.csv"

    data = np.load(data_path)
    labels = pd.read_csv(labels_path)
    metadata = pd.read_csv(metadata_path)
    quality = pd.read_csv(quality_path)
    target_name = "arrhythmia"
    target_id = CLASS_NAMES.index(target_name)
    target_rows = labels.index[labels.label_id == target_id].to_numpy()
    if len(target_rows) != args.samples_per_class:
        raise ValueError(f"Expected {args.samples_per_class} arrhythmia rows, found {len(target_rows)}")

    # Replay the original global RNG through preceding classes.  This reaches
    # the exact state at which arrhythmia generation began, without modifying
    # any existing samples from those classes.
    rng = np.random.default_rng(args.seed)
    for class_name in CLASS_NAMES[:target_id]:
        for _ in range(args.samples_per_class):
            generate_ecg_segment(class_name, args.duration, args.sampling_rate, rng)

    by_local_index: dict[int, tuple[np.ndarray, dict[str, float], dict[str, float]]] = {}
    for local_idx in range(args.samples_per_class):
        by_local_index[local_idx] = generate_ecg_segment(
            target_name, args.duration, args.sampling_rate, rng
        )

    for row in target_rows:
        sample_id = str(labels.at[row, "sample_id"])
        local_idx = int(sample_id.rsplit("_", 1)[1])
        segment, meta, quality_metrics = by_local_index[local_idx]
        data[row] = segment
        for key, value in meta.items():
            metadata.at[row, key] = value
        for key, value in quality_metrics.items():
            quality.at[row, key] = value

    np.save(data_path, data.astype(np.float32, copy=False))
    metadata.to_csv(metadata_path, index=False)
    quality.to_csv(quality_path, index=False)
    label_ids = labels.label_id.to_numpy(np.int64)
    _plot_example_signals(data, label_ids, CLASS_NAMES, args.sampling_rate, figures_dir)
    _plot_class_distribution(label_ids, CLASS_NAMES, figures_dir)
    save_signal_quality_outputs(quality, results_dir, figures_dir, CLASS_NAMES)
    print(f"Regenerated {len(target_rows)} arrhythmia rows; preserved {len(data) - len(target_rows)} other rows.")


if __name__ == "__main__":
    main()
