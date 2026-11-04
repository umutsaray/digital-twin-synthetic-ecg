"""Refresh the published balanced sample from the corrected full dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DATA = ROOT / "github_release" / "digital-twin-synthetic-ecg" / "data"


def main() -> None:
    full = np.load(ROOT / "data" / "synthetic_ecg_dataset.npy", mmap_mode="r")
    labels = pd.read_csv(ROOT / "data" / "labels.csv")
    metadata = pd.read_csv(ROOT / "data" / "metadata.csv")
    sample_labels = pd.read_csv(RELEASE_DATA / "sample_labels.csv")

    index_by_id = dict(zip(labels.sample_id.astype(str), labels.index))
    selected = np.asarray([index_by_id[str(sample_id)] for sample_id in sample_labels.sample_id])
    sample = np.asarray(full[selected], dtype=np.float32)
    np.save(RELEASE_DATA / "synthetic_ecg_sample_1200.npy", sample)

    refreshed_labels = labels.iloc[selected].reset_index(drop=True)
    refreshed_metadata = metadata.iloc[selected].reset_index(drop=True)
    refreshed_labels["sample_index"] = np.arange(len(selected))
    refreshed_metadata["sample_index"] = np.arange(len(selected))
    refreshed_labels.to_csv(RELEASE_DATA / "sample_labels.csv", index=False)
    refreshed_metadata.to_csv(RELEASE_DATA / "sample_metadata.csv", index=False)
    counts = refreshed_labels.label.value_counts().sort_index()
    if len(refreshed_labels) != 1200 or not (counts == 200).all():
        raise AssertionError("Release sample is not balanced at 200 records per class.")
    print(counts.to_string())


if __name__ == "__main__":
    main()
