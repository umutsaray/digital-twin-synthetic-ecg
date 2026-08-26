"""Training and cross-validation pipeline for ECG classification models."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .config import DEFAULT_CV_MODELS
    from .evaluate import (
        compute_classification_metrics,
        plot_training_curves,
        save_confusion_matrix,
        write_experiment_summary,
    )
    from .explainability import save_cnn_explainability_examples
    from .generate_dataset import CLASS_NAMES
    from .models import MODEL_REGISTRY, count_parameters, get_model
except ImportError:
    from config import DEFAULT_CV_MODELS  # type: ignore
    from evaluate import (  # type: ignore
        compute_classification_metrics,
        plot_training_curves,
        save_confusion_matrix,
        write_experiment_summary,
    )
    from explainability import save_cnn_explainability_examples  # type: ignore
    from generate_dataset import CLASS_NAMES  # type: ignore
    from models import MODEL_REGISTRY, count_parameters, get_model  # type: ignore


METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_ovr_macro",
]


class ECGDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        labels: np.ndarray,
        indices: Iterable[int],
        downsample_factor: int = 5,
        normalize: bool = True,
    ):
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = labels.astype(np.int64)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.downsample_factor = max(1, int(downsample_factor))
        self.normalize = normalize

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        x = np.asarray(self.data[index], dtype=np.float32)
        if self.downsample_factor > 1:
            usable = (x.size // self.downsample_factor) * self.downsample_factor
            x = x[:usable].reshape(-1, self.downsample_factor).mean(axis=1)
        if self.normalize:
            x = (x - float(x.mean())) / (float(x.std()) + 1e-6)
        x_tensor = torch.from_numpy(np.ascontiguousarray(x[None, :], dtype=np.float32))
        y_tensor = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x_tensor, y_tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_labels(labels_path: str | Path) -> np.ndarray:
    labels_df = pd.read_csv(labels_path)
    if "label_id" not in labels_df.columns:
        raise ValueError("labels.csv must include a label_id column.")
    return labels_df["label_id"].to_numpy(dtype=np.int64)


def make_splits(
    labels: np.ndarray,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    indices = np.arange(labels.size)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        stratify=labels,
    )
    relative_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_fraction,
        random_state=seed,
        stratify=labels[temp_idx],
    )
    return train_idx, val_idx, test_idx


def make_validation_split(
    trainval_idx: np.ndarray,
    labels: np.ndarray,
    seed: int,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    n_classes = int(np.unique(labels).size)
    val_count = max(n_classes, int(round(len(trainval_idx) * val_fraction)))
    val_count = min(val_count, len(trainval_idx) - n_classes)
    if val_count < n_classes:
        raise ValueError("Not enough samples to create a stratified validation split.")
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_count,
        random_state=seed,
        stratify=labels[trainval_idx],
    )
    return train_idx, val_idx


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((torch.argmax(logits, dim=1) == y).sum().item())
        total += batch_size

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    for x, y in loader:
        logits = model(x.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def _copy_state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _make_dataloaders(
    data_path: str | Path,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    downsample_factor: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ECGDataset(data_path, labels, train_idx, downsample_factor=downsample_factor)
    val_dataset = ECGDataset(data_path, labels, val_idx, downsample_factor=downsample_factor)
    test_dataset = ECGDataset(data_path, labels, test_idx, downsample_factor=downsample_factor)
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    return train_loader, val_loader, test_loader


def train_one_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    results_dir: Path,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    sampling_rate_hz: int,
    early_stopping_patience: int = 10,
    run_name: str | None = None,
    save_artifacts: bool = True,
    save_checkpoint: bool = True,
    generate_explainability: bool = True,
) -> Dict[str, float]:
    run_name = run_name or model_name
    display_name = MODEL_REGISTRY.get(model_name, model_name)
    model = get_model(model_name, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_f1_macro": [],
    }
    best_state = None
    best_val_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    start_time = time.time()

    print(f"\nTraining {display_name} [{run_name}] ({count_parameters(model):,} trainable parameters)")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        y_val, p_val = predict_probabilities(model, val_loader, device)
        val_metrics = compute_classification_metrics(y_val, p_val, class_names)
        val_f1 = val_metrics["f1_macro"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_f1_macro"].append(val_f1)

        improved = val_f1 > best_val_f1 + 1e-6
        if improved:
            best_val_f1 = val_f1
            best_state = _copy_state_to_cpu(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"val_f1={val_f1:.3f} best_f1={best_val_f1:.3f}"
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            print(
                f"  early stopping at epoch {epoch}; "
                f"best validation F1={best_val_f1:.4f} at epoch {best_epoch}"
            )
            break

    train_time = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = ""
    if save_checkpoint:
        best_models_dir = results_dir / "best_models"
        best_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = best_models_dir / f"{run_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "run_name": run_name,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_val_f1,
                "early_stopped": early_stopped,
            },
            checkpoint,
        )
        checkpoint_path = str(checkpoint)

    curves_path = ""
    if save_artifacts:
        curves_path = str(plot_training_curves(history, results_dir / "training_curves", run_name))

    y_test, p_test = predict_probabilities(model, test_loader, device)
    metrics = compute_classification_metrics(y_test, p_test, class_names)

    if save_artifacts:
        save_confusion_matrix(y_test, p_test, class_names, results_dir / "confusion_matrices", run_name)

    metrics.update(
        {
            "model": model_name,
            "display_name": display_name,
            "run_name": run_name,
            "best_epoch": best_epoch,
            "stopped_epoch": len(history["train_loss"]),
            "epochs_requested": epochs,
            "early_stopped": early_stopped,
            "best_val_f1_macro": best_val_f1,
            "train_time_sec": train_time,
            "parameters": count_parameters(model),
            "checkpoint_path": checkpoint_path,
            "training_curve_path": curves_path,
        }
    )

    if save_artifacts and generate_explainability and model_name == "cnn1d":
        try:
            explain_path = save_cnn_explainability_examples(
                model=model,
                dataset=test_loader.dataset,
                class_names=class_names,
                output_dir=results_dir / "explainability",
                device=device,
                sampling_rate_after_downsample=sampling_rate_hz / getattr(test_loader.dataset, "downsample_factor", 1),
            )
            metrics["explainability_path"] = str(explain_path)
        except Exception as exc:
            print(f"Could not generate CNN explainability figure: {exc}")
            metrics["explainability_path"] = ""
    else:
        metrics["explainability_path"] = ""

    return metrics


def train_models(
    data_path: str | Path = "data/synthetic_ecg_dataset.npy",
    labels_path: str | Path = "data/labels.csv",
    results_dir: str | Path = "results",
    report_path: str | Path = "report/experiment_summary.md",
    models: List[str] | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    downsample_factor: int = 5,
    seed: int = 42,
    device: str = "auto",
    samples_per_class: int = 2000,
    duration_seconds: float = 10.0,
    sampling_rate_hz: int = 250,
    early_stopping_patience: int = 10,
    write_report: bool = True,
) -> pd.DataFrame:
    seed_everything(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_names = models or list(MODEL_REGISTRY.keys())
    resolved_device = resolve_device(device)
    labels = load_labels(labels_path)
    data_shape = tuple(np.load(data_path, mmap_mode="r").shape)
    train_idx, val_idx, test_idx = make_splits(labels, seed=seed)

    print(f"Dataset shape: {data_shape}")
    print(f"Holdout split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Training on device: {resolved_device}")
    print(f"Early stopping patience: {early_stopping_patience}")

    train_loader, val_loader, test_loader = _make_dataloaders(
        data_path=data_path,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
        downsample_factor=downsample_factor,
        seed=seed,
    )

    all_metrics = []
    for model_position, model_name in enumerate(model_names):
        seed_everything(seed + model_position)
        metrics = train_one_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=CLASS_NAMES,
            results_dir=results_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            device=resolved_device,
            sampling_rate_hz=sampling_rate_hz,
            early_stopping_patience=early_stopping_patience,
            run_name=model_name,
            save_artifacts=True,
            save_checkpoint=True,
            generate_explainability=True,
        )
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    performance_path = results_dir / "model_performance.csv"
    metrics_df.to_csv(performance_path, index=False)
    print(f"Saved holdout model performance: {performance_path}")

    if write_report:
        write_experiment_summary(
            report_path=report_path,
            metrics_df=metrics_df,
            class_names=CLASS_NAMES,
            dataset_shape=data_shape,
            config={
                "duration_seconds": duration_seconds,
                "sampling_rate_hz": sampling_rate_hz,
                "samples_per_class": samples_per_class,
                "seed": seed,
                "models": ", ".join(model_names),
                "epochs": epochs,
                "batch_size": batch_size,
                "downsample_factor": downsample_factor,
                "device": str(resolved_device),
                "early_stopping_patience": early_stopping_patience,
                "k_folds": None,
                "cv_models": "",
            },
        )
        print(f"Saved experiment summary: {report_path}")
    return metrics_df


def _summarize_cross_validation(cv_results: pd.DataFrame) -> pd.DataFrame:
    if cv_results.empty:
        return pd.DataFrame()
    summary = cv_results.groupby("model")[METRIC_COLUMNS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "model" if col[0] == "model" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    display_names = cv_results.groupby("model")["display_name"].first().reset_index()
    n_folds = cv_results.groupby("model")["fold"].nunique().reset_index(name="n_folds")
    summary = summary.merge(display_names, on="model", how="left").merge(n_folds, on="model", how="left")
    ordered_cols = ["model", "display_name", "n_folds"]
    remaining_cols = [col for col in summary.columns if col not in ordered_cols]
    return summary[ordered_cols + remaining_cols]


def cross_validate_models(
    data_path: str | Path = "data/synthetic_ecg_dataset.npy",
    labels_path: str | Path = "data/labels.csv",
    results_dir: str | Path = "results",
    models: List[str] | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    downsample_factor: int = 5,
    seed: int = 42,
    device: str = "auto",
    k_folds: int = 5,
    early_stopping_patience: int = 10,
    sampling_rate_hz: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_everything(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_names = models or DEFAULT_CV_MODELS.copy()
    resolved_device = resolve_device(device)
    labels = load_labels(labels_path)

    class_counts = np.bincount(labels)
    actual_folds = min(int(k_folds), int(class_counts.min()))
    if actual_folds < 2:
        print("Skipping cross-validation because at least two samples per class are required.")
        empty = pd.DataFrame()
        empty.to_csv(results_dir / "cross_validation_results.csv", index=False)
        empty.to_csv(results_dir / "cross_validation_summary.csv", index=False)
        return empty, empty
    if actual_folds != k_folds:
        print(f"Requested {k_folds} folds, but using {actual_folds} because of class counts.")

    print(f"\nStarting {actual_folds}-fold stratified cross-validation for: {', '.join(model_names)}")
    print(f"Cross-validation device: {resolved_device}")

    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    fold_rows = []
    indices = np.arange(labels.size)

    for fold, (trainval_idx, test_idx) in enumerate(splitter.split(indices, labels), start=1):
        train_idx, val_idx = make_validation_split(trainval_idx, labels, seed=seed + fold)
        print(
            f"\nCV fold {fold}/{actual_folds}: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
        train_loader, val_loader, test_loader = _make_dataloaders(
            data_path=data_path,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=batch_size,
            downsample_factor=downsample_factor,
            seed=seed + fold,
        )

        for model_position, model_name in enumerate(model_names):
            print(f"CV fold {fold}: training {model_name}")
            run_seed = seed + fold * 100 + model_position
            seed_everything(run_seed)
            metrics = train_one_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=CLASS_NAMES,
                results_dir=results_dir,
                epochs=epochs,
                learning_rate=learning_rate,
                device=resolved_device,
                sampling_rate_hz=sampling_rate_hz,
                early_stopping_patience=early_stopping_patience,
                run_name=f"cv_{model_name}_fold_{fold}",
                save_artifacts=False,
                save_checkpoint=False,
                generate_explainability=False,
            )
            metrics.update(
                {
                    "fold": fold,
                    "n_folds": actual_folds,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx),
                    "seed": run_seed,
                }
            )
            fold_rows.append(metrics)

    cv_results = pd.DataFrame(fold_rows)
    cv_summary = _summarize_cross_validation(cv_results)
    cv_results_path = results_dir / "cross_validation_results.csv"
    cv_summary_path = results_dir / "cross_validation_summary.csv"
    cv_results.to_csv(cv_results_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    print(f"Saved fold-level cross-validation results: {cv_results_path}")
    print(f"Saved cross-validation summary: {cv_summary_path}")
    return cv_results, cv_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG classification models.")
    parser.add_argument("--data-path", default="data/synthetic_ecg_dataset.npy")
    parser.add_argument("--labels-path", default="data/labels.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report-path", default="report/experiment_summary.md")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--cv-models", nargs="+", default=DEFAULT_CV_MODELS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--downsample-factor", type=int, default=5)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-cross-validation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(
        data_path=args.data_path,
        labels_path=args.labels_path,
        results_dir=args.results_dir,
        report_path=args.report_path,
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        downsample_factor=args.downsample_factor,
        seed=args.seed,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.run_cross_validation:
        cross_validate_models(
            data_path=args.data_path,
            labels_path=args.labels_path,
            results_dir=args.results_dir,
            models=args.cv_models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            downsample_factor=args.downsample_factor,
            seed=args.seed,
            device=args.device,
            k_folds=args.k_folds,
            early_stopping_patience=args.early_stopping_patience,
        )
