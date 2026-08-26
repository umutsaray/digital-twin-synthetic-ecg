# Digital Twin Synthetic ECG Experiment Summary

## Dataset

- Shape: `(12000, 2500)`
- Classes: normal_sinus_rhythm, bradycardia, tachycardia, arrhythmia, noisy_ecg, sensor_fault_ecg
- Segment duration: `10.0` seconds
- Sampling rate: `250` Hz
- Samples per class: `2000`
- Random seed: `42`

## Dataset Generation

Synthetic ECG segments are generated from Gaussian PQRST morphology with class-specific heart-rate ranges and rhythm irregularity. Each clean reference is preserved before baseline wander, Gaussian noise, powerline interference, motion artifact, clipping, missing segments, and sensor-fault corruption are applied.

## Training Configuration

- Holdout models: `cnn1d, lstm, cnn_lstm, transformer`
- Cross-validation models: `cnn1d, cnn_lstm`
- Epochs: `50`
- Batch size: `64`
- Learning rate: `0.001`
- Training downsample factor: `5`
- Early stopping patience: `10`
- Cross-validation folds: `5`
- Device: `auto`

## Holdout Model Performance

| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_ovr_macro | best_epoch | stopped_epoch | early_stopped | train_time_sec | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn1d | 0.9900 | 0.9902 | 0.9900 | 0.9900 | 0.9999 | 13 | 23 | True | 46.0045 | 33606 |
| lstm | 0.5606 | 0.5754 | 0.5606 | 0.5489 | 0.8549 | 23 | 33 | True | 270.8972 | 9350 |
| cnn_lstm | 0.9850 | 0.9852 | 0.9850 | 0.9849 | 0.9995 | 27 | 37 | True | 110.6042 | 29878 |
| transformer | 0.9739 | 0.9747 | 0.9739 | 0.9739 | 0.9991 | 35 | 45 | True | 674.7928 | 21894 |

## Best Holdout Model

- Highest holdout macro F1: `cnn1d`

## Cross-Validation Summary

| model | display_name | n_folds | accuracy_mean | accuracy_std | precision_macro_mean | precision_macro_std | recall_macro_mean | recall_macro_std | f1_macro_mean | f1_macro_std | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn1d | 1D-CNN | 5 | 0.9948 | 0.0030 | 0.9948 | 0.0030 | 0.9948 | 0.0030 | 0.9947 | 0.0030 | 0.9998 | 0.0001 |
| cnn_lstm | CNN-LSTM | 5 | 0.9913 | 0.0041 | 0.9914 | 0.0041 | 0.9913 | 0.0041 | 0.9913 | 0.0041 | 0.9997 | 0.0002 |

- Fold-level rows: `10`

## Statistical Significance

One-way ANOVA did not find a significant difference in macro F1 across models (p=0.1762).

### ANOVA

| test | df_between | df_within | f_statistic | p_value | alpha | significant | method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one_way_anova | 1.0000 | 8.0000 | 2.2013 | 0.1762 | 0.0500 | False | scipy.stats.f_oneway |

### Tukey HSD

| group1 | group2 | mean_diff | p_adj | lower | upper | reject | method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn1d | cnn_lstm | -0.0034 | 0.1762 | -0.0087 | 0.0019 | False | statsmodels.pairwise_tukeyhsd |

## Signal Quality Validation

SNR, PRD, and RMSE are computed per segment by comparing the final generated ECG with its clean morphology reference.

| label | snr_db_mean | snr_db_std | prd_percent_mean | prd_percent_std | rmse_mean | rmse_std |
| --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 13.4160 | 3.1254 | 22.6905 | 7.6823 | 0.0496 | 0.0152 |
| bradycardia | 14.2675 | 3.1838 | 20.6359 | 7.2431 | 0.0372 | 0.0123 |
| noisy_ecg | 2.0602 | 2.3904 | 81.8974 | 22.5600 | 0.1779 | 0.0323 |
| normal_sinus_rhythm | 16.3697 | 3.1303 | 16.1647 | 5.5786 | 0.0354 | 0.0115 |
| sensor_fault_ecg | 3.6069 | 3.8889 | 73.9604 | 42.8026 | 0.1598 | 0.0827 |
| tachycardia | 16.1657 | 2.9784 | 16.4439 | 5.3265 | 0.0453 | 0.0137 |

## Artifacts

- Dataset: `data/synthetic_ecg_dataset.npy`
- Labels: `data/labels.csv`
- Metadata: `data/metadata.csv`
- Example signals: `figures/example_signals.png`
- Class distribution: `figures/class_distribution.png`
- SNR by class: `figures/snr_by_class.png`
- PRD by class: `figures/prd_by_class.png`
- RMSE by class: `figures/rmse_by_class.png`
- Holdout model performance: `results/model_performance.csv`
- Cross-validation fold results: `results/cross_validation_results.csv`
- Cross-validation summary: `results/cross_validation_summary.csv`
- ANOVA results: `results/anova_results.csv`
- Tukey HSD results: `results/tukey_results.csv`
- Signal quality metrics: `results/signal_quality_metrics.csv`
- Signal quality summary: `results/signal_quality_summary.csv`
- Confusion matrices: `results/confusion_matrices/`
- Training curves: `results/training_curves/`
- Best model checkpoints: `results/best_models/`
- Explainability: `results/explainability/`

## Notes

This is a synthetic research pipeline intended for digital twin prototyping and methodological comparison. It is not a substitute for clinical validation on real ECG cohorts.