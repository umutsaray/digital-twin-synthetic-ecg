"""Run the synthetic ECG publication pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig, full_config, quick_config
from src.evaluate import write_experiment_summary
from src.generate_dataset import CLASS_NAMES, generate_dataset
from src.models import MODEL_REGISTRY
from src.statistical_analysis import run_statistical_analysis
from src.train_models import cross_validate_models, train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Digital twin synthetic ECG publication pipeline."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a small end-to-end smoke experiment.")
    mode.add_argument("--full", action="store_true", help="Run the full publication experiment. This is the default.")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sampling-rate", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-cross-validation", action="store_true")
    parser.add_argument("--models", nargs="+", default=None, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--cv-models", nargs="+", default=None, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--device", default=None, help="Use 'auto', 'cpu', or 'cuda'.")
    return parser.parse_args()


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    overrides = {
        "samples_per_class": args.samples_per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "k_folds": args.k_folds,
        "random_seed": args.seed,
        "device": args.device,
        "early_stopping_patience": args.early_stopping_patience,
        "duration_seconds": args.duration,
        "sampling_rate_hz": args.sampling_rate,
        "downsample_factor": args.downsample_factor,
        "data_dir": args.data_dir,
        "figures_dir": args.figures_dir,
        "results_dir": args.results_dir,
        "report_dir": args.report_dir,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(config, field_name, value)
    if args.models is not None:
        config.models = args.models
    if args.cv_models is not None:
        config.cv_models = args.cv_models
    if args.skip_cross_validation:
        config.run_cross_validation = False
    return config


def _empty_csv(path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame()
    empty.to_csv(path, index=False)
    return empty


def main() -> None:
    args = parse_args()
    config = quick_config() if args.quick else full_config()
    config = _apply_overrides(config, args)

    data_dir = Path(config.data_dir)
    figures_dir = Path(config.figures_dir)
    results_dir = Path(config.results_dir)
    report_path = Path(config.report_dir) / "experiment_summary.md"

    print("=" * 72)
    print("Digital Twin Synthetic ECG Publication Pipeline")
    print("=" * 72)
    print(f"Mode: {'quick test' if args.quick else 'full publication'}")
    print(f"Samples per class: {config.samples_per_class}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Models: {', '.join(config.models)}")
    print(f"CV models: {', '.join(config.cv_models) if config.run_cross_validation else 'disabled'}")

    paths = generate_dataset(
        data_dir=data_dir,
        figures_dir=figures_dir,
        results_dir=results_dir,
        samples_per_class=config.samples_per_class,
        duration=config.duration_seconds,
        sampling_rate=config.sampling_rate_hz,
        seed=config.random_seed,
        force=args.force_generate,
    )

    if args.skip_training:
        print("Skipping model training because --skip-training was provided.")
        metrics_df = _empty_csv(results_dir / "model_performance.csv")
        cv_results_df = _empty_csv(results_dir / "cross_validation_results.csv")
        cv_summary_df = _empty_csv(results_dir / "cross_validation_summary.csv")
    else:
        metrics_df = train_models(
            data_path=paths["data"],
            labels_path=paths["labels"],
            results_dir=results_dir,
            report_path=report_path,
            models=config.models,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            downsample_factor=config.downsample_factor,
            seed=config.random_seed,
            device=config.device,
            samples_per_class=config.samples_per_class,
            duration_seconds=config.duration_seconds,
            sampling_rate_hz=config.sampling_rate_hz,
            early_stopping_patience=config.early_stopping_patience,
            write_report=False,
        )

        if config.run_cross_validation:
            cv_results_df, cv_summary_df = cross_validate_models(
                data_path=paths["data"],
                labels_path=paths["labels"],
                results_dir=results_dir,
                models=config.cv_models,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                downsample_factor=config.downsample_factor,
                seed=config.random_seed,
                device=config.device,
                k_folds=config.k_folds,
                early_stopping_patience=config.early_stopping_patience,
                sampling_rate_hz=config.sampling_rate_hz,
            )
        else:
            print("Cross-validation disabled.")
            cv_results_df = _empty_csv(results_dir / "cross_validation_results.csv")
            cv_summary_df = _empty_csv(results_dir / "cross_validation_summary.csv")

    statistics = run_statistical_analysis(cv_results_df, results_dir=results_dir)
    quality_summary_path = Path(paths["signal_quality_summary"])
    signal_quality_summary_df = (
        pd.read_csv(quality_summary_path) if quality_summary_path.exists() else pd.DataFrame()
    )
    dataset_shape = tuple(np.load(paths["data"], mmap_mode="r").shape)

    write_experiment_summary(
        report_path=report_path,
        metrics_df=metrics_df,
        class_names=CLASS_NAMES,
        dataset_shape=dataset_shape,
        config={
            "duration_seconds": config.duration_seconds,
            "sampling_rate_hz": config.sampling_rate_hz,
            "samples_per_class": config.samples_per_class,
            "seed": config.random_seed,
            "models": ", ".join(config.models),
            "cv_models": ", ".join(config.cv_models) if config.run_cross_validation else "disabled",
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "downsample_factor": config.downsample_factor,
            "device": config.device,
            "early_stopping_patience": config.early_stopping_patience,
            "k_folds": config.k_folds if config.run_cross_validation else 0,
        },
        cv_results_df=cv_results_df,
        cv_summary_df=cv_summary_df,
        anova_df=statistics["anova"],
        tukey_df=statistics["tukey"],
        signal_quality_summary_df=signal_quality_summary_df,
        statistical_interpretation=str(statistics["interpretation"]),
    )
    print(f"Saved final experiment summary: {report_path}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
