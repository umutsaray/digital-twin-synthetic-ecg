"""Statistical significance analysis for cross-validation results."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _manual_anova(groups: Dict[str, np.ndarray]) -> Dict[str, float]:
    all_values = np.concatenate(list(groups.values()))
    grand_mean = float(np.mean(all_values))
    ss_between = sum(len(values) * (float(np.mean(values)) - grand_mean) ** 2 for values in groups.values())
    ss_within = sum(float(np.sum((values - float(np.mean(values))) ** 2)) for values in groups.values())
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    ms_between = ss_between / max(1, df_between)
    ms_within = ss_within / max(1, df_within)
    f_statistic = ms_between / ms_within if ms_within > 0 else np.inf
    return {
        "df_between": float(df_between),
        "df_within": float(df_within),
        "f_statistic": float(f_statistic),
        "p_value": np.nan,
    }


def _anova_results(cv_results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    groups = {
        model: values["f1_macro"].dropna().to_numpy(dtype=np.float64)
        for model, values in cv_results.groupby("model")
    }
    groups = {model: values for model, values in groups.items() if values.size > 0}
    if len(groups) < 2:
        return pd.DataFrame(
            [
                {
                    "test": "one_way_anova",
                    "df_between": np.nan,
                    "df_within": np.nan,
                    "f_statistic": np.nan,
                    "p_value": np.nan,
                    "alpha": alpha,
                    "significant": False,
                    "method": "not_enough_models",
                }
            ]
        )

    method = "scipy.stats.f_oneway"
    try:
        from scipy.stats import f_oneway

        f_statistic, p_value = f_oneway(*groups.values())
        df_between = len(groups) - 1
        df_within = int(sum(len(values) for values in groups.values()) - len(groups))
        result = {
            "df_between": float(df_between),
            "df_within": float(df_within),
            "f_statistic": float(f_statistic),
            "p_value": float(p_value),
        }
    except Exception:
        method = "manual_f_statistic_scipy_unavailable"
        result = _manual_anova(groups)

    p_value = result["p_value"]
    significant = bool(np.isfinite(p_value) and p_value < alpha)
    return pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                **result,
                "alpha": alpha,
                "significant": significant,
                "method": method,
            }
        ]
    )


def _tukey_results(cv_results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    cleaned = cv_results[["model", "f1_macro"]].dropna()
    if cleaned["model"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "group1": "",
                    "group2": "",
                    "mean_diff": np.nan,
                    "p_adj": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "reject": False,
                    "method": "not_enough_models",
                }
            ]
        )

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey = pairwise_tukeyhsd(endog=cleaned["f1_macro"], groups=cleaned["model"], alpha=alpha)
        rows = []
        for (group1, group2), mean_diff, p_adj, confint, reject in zip(
            combinations(tukey.groupsunique, 2),
            tukey.meandiffs,
            tukey.pvalues,
            tukey.confint,
            tukey.reject,
        ):
            rows.append(
                {
                    "group1": str(group1),
                    "group2": str(group2),
                    "mean_diff": float(mean_diff),
                    "p_adj": float(p_adj),
                    "lower": float(confint[0]),
                    "upper": float(confint[1]),
                    "reject": bool(reject),
                    "method": "statsmodels.pairwise_tukeyhsd",
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        rows = []
        means = cleaned.groupby("model")["f1_macro"].mean()
        for group1, group2 in combinations(sorted(means.index), 2):
            rows.append(
                {
                    "group1": group1,
                    "group2": group2,
                    "mean_diff": float(means[group2] - means[group1]),
                    "p_adj": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "reject": False,
                    "method": "mean_difference_statsmodels_unavailable",
                }
            )
        return pd.DataFrame(rows)


def interpret_statistics(anova_df: pd.DataFrame, tukey_df: pd.DataFrame, alpha: float = 0.05) -> str:
    if anova_df.empty:
        return "Statistical analysis was not available because cross-validation results were empty."

    row = anova_df.iloc[0]
    p_value = row.get("p_value", np.nan)
    method = str(row.get("method", "unknown"))
    if method == "not_enough_models":
        return "Statistical comparison was skipped because fewer than two models had cross-validation results."
    if not np.isfinite(p_value):
        return (
            "ANOVA F-statistic was computed, but the p-value was unavailable because SciPy was not installed. "
            "Tukey HSD p-values require statsmodels."
        )
    tukey_unavailable = (
        tukey_df.empty
        or "method" not in tukey_df.columns
        or tukey_df["method"].astype(str).str.contains("unavailable|not_enough", regex=True).any()
    )
    if p_value < alpha:
        if tukey_unavailable:
            return (
                f"One-way ANOVA found a significant model effect on macro F1 "
                f"(p={p_value:.4g}). Tukey HSD post-hoc p-values were unavailable because statsmodels "
                "was not installed; mean differences are reported without adjusted significance decisions."
            )
        if "reject" in tukey_df.columns:
            significant_pairs = tukey_df.loc[tukey_df["reject"] == True]  # noqa: E712
        else:
            significant_pairs = pd.DataFrame()
        if significant_pairs.empty:
            return (
                f"One-way ANOVA found a significant model effect on macro F1 "
                f"(p={p_value:.4g}), but Tukey HSD did not identify significant pairwise differences at alpha={alpha}."
            )
        pairs = [
            f"{pair.group1} vs {pair.group2}"
            for pair in significant_pairs.itertuples(index=False)
        ]
        return (
            f"One-way ANOVA found a significant model effect on macro F1 "
            f"(p={p_value:.4g}). Tukey HSD identified significant differences for: {', '.join(pairs)}."
        )
    if tukey_unavailable:
        return (
            f"One-way ANOVA did not find a significant difference in macro F1 across models "
            f"(p={p_value:.4g}). Tukey HSD p-values were unavailable because statsmodels was not installed."
        )
    return f"One-way ANOVA did not find a significant difference in macro F1 across models (p={p_value:.4g})."


def run_statistical_analysis(
    cv_results: pd.DataFrame,
    results_dir: str | Path,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Run ANOVA and Tukey HSD on fold-level macro F1 scores."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if cv_results.empty:
        anova_df = pd.DataFrame(
            [
                {
                    "test": "one_way_anova",
                    "df_between": np.nan,
                    "df_within": np.nan,
                    "f_statistic": np.nan,
                    "p_value": np.nan,
                    "alpha": alpha,
                    "significant": False,
                    "method": "empty_cross_validation_results",
                }
            ]
        )
        tukey_df = pd.DataFrame()
    else:
        anova_df = _anova_results(cv_results, alpha=alpha)
        tukey_df = _tukey_results(cv_results, alpha=alpha)

    anova_path = results_dir / "anova_results.csv"
    tukey_path = results_dir / "tukey_results.csv"
    anova_df.to_csv(anova_path, index=False)
    tukey_df.to_csv(tukey_path, index=False)
    interpretation = interpret_statistics(anova_df, tukey_df, alpha=alpha)
    return {
        "anova": anova_df,
        "tukey": tukey_df,
        "anova_path": anova_path,
        "tukey_path": tukey_path,
        "interpretation": interpretation,
    }
