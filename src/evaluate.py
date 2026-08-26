"""Evaluation, plotting, and report utilities for ECG experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _format_markdown_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "N/A"
        return f"{float(value):.4f}"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return str(value)


def _metrics_to_markdown_table(metrics_df: pd.DataFrame, columns: List[str]) -> str:
    if metrics_df.empty:
        return "No rows available."
    usable_columns = [column for column in columns if column in metrics_df.columns]
    if not usable_columns:
        return "No requested columns were available."
    header = "| " + " | ".join(usable_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(usable_columns)) + " |"
    rows = []
    for _, row in metrics_df[usable_columns].iterrows():
        rows.append("| " + " | ".join(_format_markdown_value(row[column]) for column in usable_columns) + " |")
    return "\n".join([header, separator, *rows])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        if y_true_bin.shape[1] == y_prob.shape[1]:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            )
        else:
            metrics["roc_auc_ovr_macro"] = np.nan
    except ValueError:
        metrics["roc_auc_ovr_macro"] = np.nan

    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_dir: str | Path,
    model_name: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    csv_path = output_dir / f"{model_name}_confusion_matrix.csv"
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=[name.replace("_", "\n") for name in class_names],
        yticklabels=[name.replace("_", "\n") for name in class_names],
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{model_name} confusion matrix",
    )
    threshold = matrix.max() / 2.0 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    png_path = output_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path


def plot_training_curves(history: Dict[str, Iterable[float]], output_dir: str | Path, model_name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loss = list(history["train_loss"])
    epochs = np.arange(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    if "val_f1_macro" in history:
        axes[1].plot(epochs, history["val_f1_macro"], marker="o", label="Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Accuracy and F1")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"{model_name} training curves")
    fig.tight_layout()
    output_path = output_dir / f"{model_name}_training_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _best_model_name(metrics_df: pd.DataFrame) -> str:
    if metrics_df.empty or "f1_macro" not in metrics_df.columns:
        return "N/A"
    return str(metrics_df.sort_values("f1_macro", ascending=False).iloc[0]["model"])


def write_experiment_summary(
    report_path: str | Path,
    metrics_df: pd.DataFrame,
    class_names: List[str],
    dataset_shape: Optional[tuple],
    config: Dict[str, object],
    cv_results_df: Optional[pd.DataFrame] = None,
    cv_summary_df: Optional[pd.DataFrame] = None,
    anova_df: Optional[pd.DataFrame] = None,
    tukey_df: Optional[pd.DataFrame] = None,
    signal_quality_summary_df: Optional[pd.DataFrame] = None,
    statistical_interpretation: str = "",
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    holdout_columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
        "best_epoch",
        "stopped_epoch",
        "early_stopped",
        "train_time_sec",
        "parameters",
    ]
    cv_columns = [
        "model",
        "display_name",
        "n_folds",
        "accuracy_mean",
        "accuracy_std",
        "precision_macro_mean",
        "precision_macro_std",
        "recall_macro_mean",
        "recall_macro_std",
        "f1_macro_mean",
        "f1_macro_std",
        "roc_auc_ovr_macro_mean",
        "roc_auc_ovr_macro_std",
    ]
    quality_columns = [
        "label",
        "snr_db_mean",
        "snr_db_std",
        "prd_percent_mean",
        "prd_percent_std",
        "rmse_mean",
        "rmse_std",
    ]

    cv_summary_df = cv_summary_df if cv_summary_df is not None else pd.DataFrame()
    cv_results_df = cv_results_df if cv_results_df is not None else pd.DataFrame()
    anova_df = anova_df if anova_df is not None else pd.DataFrame()
    tukey_df = tukey_df if tukey_df is not None else pd.DataFrame()
    signal_quality_summary_df = (
        signal_quality_summary_df if signal_quality_summary_df is not None else pd.DataFrame()
    )

    lines = [
        "# Digital Twin Synthetic ECG Experiment Summary",
        "",
        "## Dataset",
        "",
        f"- Shape: `{dataset_shape}`",
        f"- Classes: {', '.join(class_names)}",
        f"- Segment duration: `{config.get('duration_seconds')}` seconds",
        f"- Sampling rate: `{config.get('sampling_rate_hz')}` Hz",
        f"- Samples per class: `{config.get('samples_per_class')}`",
        f"- Random seed: `{config.get('seed')}`",
        "",
        "## Dataset Generation",
        "",
        (
            "Synthetic ECG segments are generated from Gaussian PQRST morphology with class-specific "
            "heart-rate ranges and rhythm irregularity. Each clean reference is preserved before "
            "baseline wander, Gaussian noise, powerline interference, motion artifact, clipping, "
            "missing segments, and sensor-fault corruption are applied."
        ),
        "",
        "## Training Configuration",
        "",
        f"- Holdout models: `{config.get('models')}`",
        f"- Cross-validation models: `{config.get('cv_models')}`",
        f"- Epochs: `{config.get('epochs')}`",
        f"- Batch size: `{config.get('batch_size')}`",
        f"- Learning rate: `{config.get('learning_rate')}`",
        f"- Training downsample factor: `{config.get('downsample_factor')}`",
        f"- Early stopping patience: `{config.get('early_stopping_patience')}`",
        f"- Cross-validation folds: `{config.get('k_folds')}`",
        f"- Device: `{config.get('device')}`",
        "",
        "## Holdout Model Performance",
        "",
        _metrics_to_markdown_table(metrics_df, holdout_columns),
        "",
        "## Best Holdout Model",
        "",
        f"- Highest holdout macro F1: `{_best_model_name(metrics_df)}`",
        "",
        "## Cross-Validation Summary",
        "",
        _metrics_to_markdown_table(cv_summary_df, cv_columns),
        "",
        f"- Fold-level rows: `{len(cv_results_df)}`",
        "",
        "## Statistical Significance",
        "",
        statistical_interpretation or "Statistical interpretation was not generated.",
        "",
        "### ANOVA",
        "",
        _metrics_to_markdown_table(
            anova_df,
            ["test", "df_between", "df_within", "f_statistic", "p_value", "alpha", "significant", "method"],
        ),
        "",
        "### Tukey HSD",
        "",
        _metrics_to_markdown_table(
            tukey_df,
            ["group1", "group2", "mean_diff", "p_adj", "lower", "upper", "reject", "method"],
        ),
        "",
        "## Signal Quality Validation",
        "",
        (
            "SNR, PRD, and RMSE are computed per segment by comparing the final generated ECG with "
            "its clean morphology reference."
        ),
        "",
        _metrics_to_markdown_table(signal_quality_summary_df, quality_columns),
        "",
        "## Artifacts",
        "",
        "- Dataset: `data/synthetic_ecg_dataset.npy`",
        "- Labels: `data/labels.csv`",
        "- Metadata: `data/metadata.csv`",
        "- Example signals: `figures/example_signals.png`",
        "- Class distribution: `figures/class_distribution.png`",
        "- SNR by class: `figures/snr_by_class.png`",
        "- PRD by class: `figures/prd_by_class.png`",
        "- RMSE by class: `figures/rmse_by_class.png`",
        "- Holdout model performance: `results/model_performance.csv`",
        "- Cross-validation fold results: `results/cross_validation_results.csv`",
        "- Cross-validation summary: `results/cross_validation_summary.csv`",
        "- ANOVA results: `results/anova_results.csv`",
        "- Tukey HSD results: `results/tukey_results.csv`",
        "- Signal quality metrics: `results/signal_quality_metrics.csv`",
        "- Signal quality summary: `results/signal_quality_summary.csv`",
        "- Confusion matrices: `results/confusion_matrices/`",
        "- Training curves: `results/training_curves/`",
        "- Best model checkpoints: `results/best_models/`",
        "- Explainability: `results/explainability/`",
        "",
        "## Notes",
        "",
        (
            "This is a synthetic research pipeline intended for digital twin prototyping and "
            "methodological comparison. It is not a substitute for clinical validation on real ECG cohorts."
        ),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
