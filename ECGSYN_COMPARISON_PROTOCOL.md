# ECGSYN Generator Comparison Protocol

## Scope

This analysis compares this repository's generator with ECGSYN on two already established realism axes: RR intervals against MIT-BIH and P/QRS/T timing against LUDB. The comparison includes `normal_sinus_rhythm`, `bradycardia`, and `tachycardia` only. Arrhythmia is explicitly excluded because standard ECGSYN does not implement the discrete short-coupled ectopic beat followed by a compensatory pause used by this repository. No substitute ECGSYN arrhythmia was fabricated.

This is a signal-realism comparison. No classifier was retrained or evaluated on ECGSYN signals.

## ECGSYN generation

- Implementation: NeuroKit2 0.3.0, `ecg_simulate(method="ecgsyn")`
- Records: 500 per class (1,500 total)
- Duration and sampling: 10 s at 250 Hz
- Heart rate: uniformly sampled per record from 62–88 bpm (normal), 36–55 bpm (bradycardia), or 105–150 bpm (tachycardia)
- `heart_rate_std=1`, `noise=0.01`
- Master NumPy seed: 7373; a deterministic per-record integer seed is passed to ECGSYN

## RR analysis

R peaks are detected from each ECGSYN output signal with NeuroKit2 `ecg_peaks(method="neurokit")`; RR intervals are successive detected-peak differences. This detector-based route was chosen instead of treating requested heart rate as observed timing. The existing `results/synthetic_rr_distributions.csv` and `results/real_mitdb_rr_distributions.csv` value arrays are reused unchanged. Each class reports two-sample KS tests and Wasserstein distances for our generator vs. real, ECGSYN vs. real, and our generator vs. ECGSYN.

## P/QRS/T timing analysis

ECGSYN outputs are cleaned with `ecg_clean(method="neurokit")`, then delineated with `ecg_delineate(method="dwt")`. Only beats with finite, positive and complete P onset/offset, R onset/offset, and T onset/offset are retained. Definitions match the LUDB analysis: P duration, PR interval, QRS duration, QT interval, ST segment, and T duration.

The unchanged LUDB extractor and our-generator extractor from `compare_pqrst_timing.py` recreate the value arrays required for distribution tests. Before use, their counts and means must match every corresponding row of the immutable prior `pqrst_timing_comparison.csv`; otherwise the new script stops. Thus the real reference and our-generator reference are exactly the previously reported samples, not a new mapping or definition.

## Methodological asymmetry

The timing comparison is not perfectly symmetric. ECGSYN boundaries are estimates made by an external DWT delineator on the generated waveform. Our generator has no finite annotated Gaussian boundaries, so its prior analysis uses known construction parameters and a declared ±2σ proxy. Consequently, differences in distance from LUDB reflect both generator morphology and boundary-measurement method. The values must not be interpreted as a pure head-to-head morphology score.

## Detection yield

| Class | Records | Detected R peaks | Complete delineated beats |
|---|---:|---:|---:|
| Normal sinus rhythm | 500 | 5,953 | 5,746 |
| Bradycardia | 500 | 3,546 | 3,334 |
| Tachycardia | 500 | 10,311 | 6,167 |

Incomplete delineations are omitted beat-wise; no class or interval is dropped.

