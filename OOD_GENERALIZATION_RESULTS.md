# OOD Generalization Results

The four publication checkpoints were evaluated zero-shot on 600 fully disjoint OOD records. No model parameter was updated and the OOD data were not used for selection or tuning.

| Model | Original holdout macro F1 | OOD macro F1 | Absolute change | Relative decrease |
|---|---:|---:|---:|---:|
| 1D-CNN | 0.9978 | 0.5398 | -0.4580 | 45.9% |
| LSTM | 0.6983 | 0.3462 | -0.3521 | 50.4% |
| CNN-LSTM | 0.9950 | 0.5299 | -0.4651 | 46.7% |
| Transformer | 0.9800 | 0.5366 | -0.4434 | 45.2% |

## Interpretation

Performance does not hold at the near-ceiling level observed on the in-distribution holdout set. All four models degrade materially when heart rates move outside their training ranges and acquisition-artifact profiles are cross-applied. The 1D-CNN is marginally strongest at only 0.5398 macro F1, down from 0.9978. Its confusion matrix shows the intended shortcut stress directly: all 50 normal-rhythm records carrying the noisy-ECG profile are predicted as noisy ECG, 47 bradycardia records are predicted as sensor fault, and 49 tachycardia records are predicted as noisy ECG. Removing the characteristic artifact profiles also causes noisy and sensor-fault records to be redistributed among rhythm classes. The results therefore indicate that the original holdout scores substantially reflect the generator's class-specific parameter and artifact structure and should not be interpreted as evidence of robust generalization to unseen parameter combinations or clinical data.

Detailed metrics are in `results/ood_generalization_metrics.csv`; six-class confusion matrices are in `results/confusion_matrices/`.
