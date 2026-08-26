"""Signal quality validation metrics for synthetic ECG generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUALITY_COLUMNS = ["snr_db", "prd_percent", "rmse"]


def compute_signal_quality_metrics(reference: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    """Compute SNR, PRD, and RMSE against a clean ECG reference."""
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    error = generated - reference
    reference_energy = float(np.sum(reference**2))
    error_energy = float(np.sum(error**2))
    eps = 1e-12

    snr_db = 10.0 * np.log10((reference_energy + eps) / (error_energy + eps))
    prd_percent = 100.0 * np.sqrt(error_energy / (reference_energy + eps))
    rmse = np.sqrt(float(np.mean(error**2)))
    return {
        "snr_db": float(snr_db),
        "prd_percent": float(prd_percent),
        "rmse": float(rmse),
    }


def summarize_signal_quality(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return class-level mean and standard deviation for all quality metrics."""
    summary = metrics_df.groupby("label")[QUALITY_COLUMNS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "label" if col[0] == "label" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return summary


def _plot_quality_metric(
    metrics_df: pd.DataFrame,
    class_names: List[str],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        metrics_df.loc[metrics_df["label"] == class_name, metric].dropna().to_numpy()
        for class_name in class_names
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(values, showfliers=False, patch_artist=True)
    for patch, color in zip(
        box["boxes"],
        ["#2f6f9f", "#559e83", "#d6a01d", "#c65353", "#7b5aa6", "#5c677d"],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(class_names) + 1))
    ax.set_xticklabels([name.replace("_", "\n") for name in class_names], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_signal_quality_outputs(
    metrics_df: pd.DataFrame,
    results_dir: str | Path,
    figures_dir: str | Path,
    class_names: List[str],
) -> Dict[str, Path]:
    """Save per-sample metrics, class summaries, and quality figures."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "signal_quality_metrics.csv"
    summary_path = results_dir / "signal_quality_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df = summarize_signal_quality(metrics_df)
    summary_df.to_csv(summary_path, index=False)

    snr_path = figures_dir / "snr_by_class.png"
    prd_path = figures_dir / "prd_by_class.png"
    rmse_path = figures_dir / "rmse_by_class.png"
    _plot_quality_metric(metrics_df, class_names, "snr_db", "SNR (dB)", snr_path)
    _plot_quality_metric(metrics_df, class_names, "prd_percent", "PRD (%)", prd_path)
    _plot_quality_metric(metrics_df, class_names, "rmse", "RMSE", rmse_path)

    return {
        "signal_quality_metrics": metrics_path,
        "signal_quality_summary": summary_path,
        "snr_by_class": snr_path,
        "prd_by_class": prd_path,
        "rmse_by_class": rmse_path,
    }


def load_signal_quality_summary(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
