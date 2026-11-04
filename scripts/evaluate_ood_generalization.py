"""Evaluate existing publication checkpoints zero-shot on the OOD test set."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import compute_classification_metrics, save_confusion_matrix
from src.generate_dataset import CLASS_NAMES
from src.models import MODEL_REGISTRY, get_model
from src.train_models import ECGDataset, make_loader, predict_probabilities, resolve_device


MODELS = ["cnn1d", "lstm", "cnn_lstm", "transformer"]


def main() -> None:
    data_path = ROOT / "data" / "ood" / "ood_ecg_test_600.npy"
    labels_path = ROOT / "data" / "ood" / "ood_labels.csv"
    results_dir = ROOT / "results"
    labels = pd.read_csv(labels_path).label_id.to_numpy(np.int64)
    dataset = ECGDataset(data_path, labels, np.arange(len(labels)), downsample_factor=5)
    loader = make_loader(dataset, batch_size=128, shuffle=False, seed=4242)
    device = resolve_device("auto")
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
        metrics = compute_classification_metrics(y_true, probabilities, CLASS_NAMES)
        rows.append({"model": name, "display_name": MODEL_REGISTRY[name], **metrics})
        save_confusion_matrix(
            y_true,
            probabilities,
            CLASS_NAMES,
            results_dir / "confusion_matrices",
            f"{name}_ood",
        )

    output = pd.DataFrame(rows)
    output.to_csv(results_dir / "ood_generalization_metrics.csv", index=False)
    print(output[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
