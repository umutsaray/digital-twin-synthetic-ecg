"""Central experiment configuration for publication and quick-test runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


ALL_MODELS = ["cnn1d", "lstm", "cnn_lstm", "transformer"]
DEFAULT_CV_MODELS = ["cnn1d", "cnn_lstm"]


@dataclass
class ExperimentConfig:
    samples_per_class: int = 2000
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    k_folds: int = 5
    cv_models: List[str] = field(default_factory=lambda: DEFAULT_CV_MODELS.copy())
    random_seed: int = 42
    device: str = "auto"
    early_stopping_patience: int = 10
    duration_seconds: float = 10.0
    sampling_rate_hz: int = 250
    downsample_factor: int = 5
    models: List[str] = field(default_factory=lambda: ALL_MODELS.copy())
    data_dir: str = "data"
    figures_dir: str = "figures"
    results_dir: str = "results"
    report_dir: str = "report"
    run_cross_validation: bool = True


def full_config() -> ExperimentConfig:
    """Publication-mode defaults."""
    return ExperimentConfig()


def quick_config() -> ExperimentConfig:
    """Small deterministic run for syntax and workflow validation."""
    return ExperimentConfig(
        samples_per_class=20,
        epochs=1,
        batch_size=32,
        k_folds=2,
        cv_models=["cnn1d"],
        models=["cnn1d"],
        early_stopping_patience=2,
        data_dir=".quick/data",
        figures_dir=".quick/figures",
        results_dir=".quick/results",
        report_dir=".quick/report",
    )

