# OOD Generalization Protocol

## Purpose

This test measures zero-shot performance under deliberately unseen generator-parameter combinations. The 600 OOD records are never used for training, validation, checkpoint selection, or fine-tuning.

## Fixed format and seed

- 100 records per class; 600 total
- 10 seconds per record at 250 Hz
- Random seed: `4242`
- Evaluation preprocessing is identical to the publication test set: five-point mean downsampling followed by per-record z-normalization.

## Heart-rate perturbation

For an original training interval `[L, U]`, let `W = U - L`. OOD heart rates are sampled equally from the two external tails `[L - 0.15W, L]` and `[U, U + 0.15W]`. Endpoints have zero-probability relevance under continuous sampling.

| Class | Training range (bpm) | Lower OOD tail | Upper OOD tail |
|---|---:|---:|---:|
| normal_sinus_rhythm | 62–88 | 58.10–62.00 | 88.00–91.90 |
| bradycardia | 36–55 | 33.15–36.00 | 55.00–57.85 |
| tachycardia | 105–150 | 98.25–105.00 | 150.00–156.75 |
| arrhythmia | 58–104 | 51.10–58.00 | 104.00–110.90 |
| noisy_ecg | 60–96 | 54.60–60.00 | 96.00–101.40 |
| sensor_fault_ecg | 58–98 | 52.00–58.00 | 98.00–104.00 |

## Cross-applied artifact profiles

Exactly 50 of the 100 OOD records in each class use another class's complete acquisition-artifact profile: Gaussian noise, baseline wander, powerline noise, motion probability, clipping probability, missing-segment probability, amplitude range, and sensor-fault activation. The target class retains its own morphology, HRV range, and irregular-rhythm setting.

| Target class | Artifact donor for 50 records |
|---|---|
| normal_sinus_rhythm | noisy_ecg |
| bradycardia | sensor_fault_ecg |
| tachycardia | noisy_ecg |
| arrhythmia | sensor_fault_ecg |
| noisy_ecg | tachycardia |
| sensor_fault_ecg | normal_sinus_rhythm |

The remaining 50 records per class retain their target-class artifact profile but still use an external HR tail. This factorial mixture separates the effect of HR extrapolation from the combined HR-plus-artifact shift.

## Evaluation

The existing `cnn1d`, `lstm`, `cnn_lstm`, and `transformer` checkpoints in `results/best_models/` are loaded without any parameter update. Accuracy, macro precision, macro recall, macro F1, ROC-AUC where defined, and six-class confusion matrices are computed on the same 600 OOD records.
