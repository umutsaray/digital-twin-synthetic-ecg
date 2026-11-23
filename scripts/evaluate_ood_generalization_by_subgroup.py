"""Evaluate fixed OOD predictions by heart-rate/artifact-swap subgroup."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import compute_classification_metrics
from src.generate_dataset import CLASS_NAMES
from src.models import MODEL_REGISTRY, get_model
from src.train_models import ECGDataset, make_loader, predict_probabilities, resolve_device


MODELS = ["cnn1d", "lstm", "cnn_lstm", "transformer"]
SUBGROUPS = [
    ("aggregate", None),
    ("heart-rate-alone", False),
    ("heart-rate + artifact-swap", True),
]
ALIGNMENT_COLUMNS = ["sample_index", "sample_id", "label_id", "label"]


def load_and_verify_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both tables and fail loudly unless their identifying columns align."""
    labels = pd.read_csv(ROOT / "data" / "ood" / "ood_labels.csv")
    metadata = pd.read_csv(ROOT / "data" / "ood" / "ood_metadata.csv")
    if len(labels) != 600 or len(metadata) != 600:
        raise AssertionError(
            f"Expected 600 aligned OOD rows, got labels={len(labels)}, metadata={len(metadata)}."
        )
    missing = [
        f"{table_name}.{column}"
        for table_name, table in (("labels", labels), ("metadata", metadata))
        for column in ALIGNMENT_COLUMNS
        if column not in table.columns
    ]
    if missing:
        raise AssertionError(f"Missing alignment columns: {', '.join(missing)}")
    if "artifact_cross_applied" not in metadata.columns:
        raise AssertionError("metadata.artifact_cross_applied is required.")
    for column in ALIGNMENT_COLUMNS:
        equal = labels[column].equals(metadata[column])
        if not equal:
            mismatch = np.flatnonzero(labels[column].to_numpy() != metadata[column].to_numpy())
            preview = mismatch[:5].tolist()
            raise AssertionError(
                f"OOD row alignment failed for {column!r}; first mismatched rows: {preview}."
            )
    if not np.array_equal(metadata.sample_index.to_numpy(), np.arange(600)):
        raise AssertionError("metadata.sample_index must be exactly 0..599 in prediction order.")
    counts = metadata.groupby(["artifact_cross_applied", "label"], sort=False).size()
    if set(counts.to_numpy()) != {50} or len(counts) != 12:
        raise AssertionError(
            "Expected 50 records per class in each artifact subgroup; "
            f"observed {counts.to_dict()}."
        )
    return labels, metadata


def per_class_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_pred = np.argmax(probabilities, axis=1)
    labels = np.arange(len(CLASS_NAMES))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_values = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    values: dict[str, float] = {}
    for class_name, recall, f1 in zip(CLASS_NAMES, recalls, f1_values):
        values[f"recall_{class_name}"] = float(recall)
        values[f"f1_{class_name}"] = float(f1)
    return values


def subgroup_row(
    model_name: str,
    subgroup: str,
    mask: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    subgroup_y = y_true[mask]
    subgroup_probabilities = probabilities[mask]
    metrics = compute_classification_metrics(
        subgroup_y, subgroup_probabilities, CLASS_NAMES
    )
    return {
        "model": model_name,
        "display_name": MODEL_REGISTRY[model_name],
        "subgroup": subgroup,
        "n": int(mask.sum()),
        **metrics,
        **per_class_metrics(subgroup_y, subgroup_probabilities),
    }


def main() -> None:
    labels_table, metadata = load_and_verify_labels()
    labels = labels_table.label_id.to_numpy(np.int64)
    dataset = ECGDataset(
        ROOT / "data" / "ood" / "ood_ecg_test_600.npy",
        labels,
        np.arange(len(labels)),
        downsample_factor=5,
    )
    loader = make_loader(dataset, batch_size=128, shuffle=False, seed=4242)
    device = resolve_device("auto")
    artifact_swap = metadata.artifact_cross_applied.to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []

    for name in MODELS:
        checkpoint = torch.load(
            ROOT / "results" / "best_models" / f"{name}.pt",
            map_location=device,
            weights_only=False,
        )
        model = get_model(name, num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        y_true, probabilities = predict_probabilities(model, loader, device)
        if not np.array_equal(y_true, labels):
            raise AssertionError(
                f"Prediction order for {name} does not match verified OOD label order."
            )
        for subgroup, flag in SUBGROUPS:
            mask = np.ones(len(labels), dtype=bool) if flag is None else artifact_swap == flag
            rows.append(subgroup_row(name, subgroup, mask, y_true, probabilities))

    output = pd.DataFrame(rows)
    prior = pd.read_csv(ROOT / "results" / "ood_generalization_metrics.csv").set_index("model")
    aggregate = output.loc[output.subgroup == "aggregate"].set_index("model")
    metric_columns = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_ovr_macro",
    ]
    for model_name in MODELS:
        if not np.allclose(
            aggregate.loc[model_name, metric_columns].to_numpy(dtype=float),
            prior.loc[model_name, metric_columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        ):
            raise AssertionError(
                f"Aggregate metrics for {model_name} do not reproduce the immutable prior CSV."
            )

    output_path = ROOT / "results" / "ood_generalization_by_subgroup.csv"
    output.to_csv(output_path, index=False)
    print("OOD row alignment: PASSED (600/600 rows; sample_index, sample_id, label_id, label).")
    print("Subgroup balance: PASSED (300 rows/subgroup; 50 rows/class/subgroup).")
    print("Aggregate reproduction: PASSED for all four models.")
    print()
    print(
        output[["model", "subgroup", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
