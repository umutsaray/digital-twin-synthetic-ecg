"""Create transparent before/after metric tables for the ectopic RR fix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "revision_before_ectopic_fix"

TABLES = [
    ("Synthetic holdout", "results/model_performance.csv", ["model"], ["accuracy", "f1_macro", "roc_auc_ovr_macro"]),
    ("Cross-validation", "results/cross_validation_summary.csv", ["model"], ["accuracy_mean", "f1_macro_mean", "roc_auc_ovr_macro_mean"]),
    ("Calibration", "results/q34_extensions/calibration_metrics.csv", ["model"], ["ece_15", "multiclass_brier", "nll"]),
    ("MC-dropout uncertainty", "results/q34_extensions/mc_dropout_uncertainty_by_class.csv", ["class_name"], ["predictive_entropy_mean", "mutual_information_mean"]),
    ("Improved LSTM", "results/q34_extensions/lstm_improved_performance.csv", ["model"], ["accuracy", "f1_macro", "roc_auc_ovr_macro"]),
    ("Noise robustness", "results/q34_extensions/noise_robustness.csv", ["target_snr_db"], ["overall_accuracy", "overall_macro_f1", "normal_recall"]),
    ("Grad-CAM deletion fidelity", "results/q34_extensions/gradcam_fidelity_by_class.csv", ["class_name"], ["confidence_drop_30pct_mean", "deletion_auc_mean"]),
    ("MIT-BIH zero-shot", "results/q34_extensions/external_real_data_metrics.csv", ["dataset"], ["zero_shot_accuracy", "zero_shot_macro_f1_4class", "noise_or_fault_prediction_rate"]),
    ("MIT-BIH adaptation", "results/q34_extensions/domain_adaptation_metrics.csv", ["method"], ["accuracy", "macro_f1_4class", "artifact_prediction_rate"]),
]


def main() -> None:
    rows: list[dict[str, object]] = []
    for table, relative, keys, metrics in TABLES:
        old_path = BEFORE / relative
        new_path = ROOT / relative
        if not old_path.exists() or not new_path.exists():
            continue
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        usable_keys = [key for key in keys if key in old.columns and key in new.columns]
        if not usable_keys:
            continue
        merged = old.merge(new, on=usable_keys, suffixes=("_old", "_new"), how="outer")
        for _, record in merged.iterrows():
            identity = ", ".join(f"{key}={record[key]}" for key in usable_keys)
            for metric in metrics:
                old_col, new_col = f"{metric}_old", f"{metric}_new"
                if old_col not in merged.columns or new_col not in merged.columns:
                    continue
                old_value, new_value = record[old_col], record[new_col]
                if pd.isna(old_value) and pd.isna(new_value):
                    continue
                delta = new_value - old_value if pd.notna(old_value) and pd.notna(new_value) else np.nan
                rows.append(
                    {
                        "table": table,
                        "row": identity,
                        "metric": metric,
                        "old_value": old_value,
                        "new_value": new_value,
                        "delta": delta,
                    }
                )

    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "results" / "ectopic_fix_before_after_metrics.csv", index=False)
    lines = [
        "# Ectopic RR Fix: Before/After Metric Summary",
        "",
        "Values below compare the archived pre-fix outputs with the complete post-fix rerun.",
        "",
        "| Table | Row | Metric | Old | New | Delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: object) -> str:
            return "N/A" if pd.isna(value) else f"{float(value):.6f}"
        lines.append(
            f"| {row['table']} | {row['row']} | {row['metric']} | "
            f"{fmt(row['old_value'])} | {fmt(row['new_value'])} | {fmt(row['delta'])} |"
        )
    (ROOT / "ECTOPIC_FIX_BEFORE_AFTER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} metric comparisons.")


if __name__ == "__main__":
    main()
