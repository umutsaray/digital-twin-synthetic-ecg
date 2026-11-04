# LUDB Lead-II P/QRS/T Timing Comparison Protocol

## Data and scope

LUDB 1.0.1 contains 200 ten-second, 500 Hz, 12-lead records with cardiologist-annotated P, QRS, and T boundaries. Only Lead II is used: the `ii` signal channel from each record and the matching `<record>.ii` annotation file. Raw LUDB files are extracted under `data/ludb/` for local analysis but are excluded from Git because they are third-party clinical data. The derived aggregate CSV and figure are committed.

## Empirically confirmed annotation convention

Before implementing the parser, record 1 was read with `wfdb.rdann(record_path, "ii")`. Its first annotations were:

- samples: `644, 662, 682, 776, 843, 878, 1250, 1278, 1302, ...`
- symbols: `(`, `N`, `)`, `(`, `t`, `)`, `(`, `p`, `)`, ...
- `aux_note`: empty strings

Across all 200 Lead-II files, the only symbols were `(`, `p`, `N`, `t`, and `)`. A complete wave is therefore parsed only from an exact three-symbol sequence:

- `(` → `p` → `)` for P onset, peak, and offset
- `(` → `N` → `)` for QRS onset, peak, and offset
- `(` → `t` → `)` for T onset, peak, and offset

For each QRS, the parser selects the closest complete P wave after the preceding QRS and before the current QRS, plus the first complete T wave after the current QRS and before the next QRS. Beats with a missing or ambiguous required wave are skipped; no marker is inferred.

## Class mapping

Mapping uses the LUDB `ludb.csv` rhythm and extrasystole fields. Precedence is exclusion → extrasystole → sinus rhythm subtype.

1. All atrial-fibrillation records (15, including the compound “aberrant conduction” entry) and all typical atrial-flutter records (3) are excluded, even when an extrasystole diagnosis is also present. AF/flutter is not the mechanism represented by the synthetic arrhythmia class.
2. Any remaining record with a non-empty LUDB extrasystole field—atrial or ventricular, including PAC/PVC pattern and localization variants—is mapped to `arrhythmia`.
3. Remaining sinus rhythm, sinus bradycardia, and sinus tachycardia records map to `normal_sinus_rhythm`, `bradycardia`, and `tachycardia`.
4. Sinus arrhythmia or irregular sinus rhythm without an extrasystole diagnosis is not forced into the synthetic arrhythmia class and is excluded as unmapped.
5. `noisy_ecg` and `sensor_fault_ecg` have no LUDB clinical analogue and are not compared.

Final mapped record counts after precedence and exclusions are:

| Synthetic class | LUDB records |
|---|---:|
| normal_sinus_rhythm | 136 |
| bradycardia | 24 |
| tachycardia | 4 |
| arrhythmia | 11 |

Eighteen AF/flutter records and seven other unmapped rhythm records are excluded. The README's extrasystole diagnosis counts overlap within patients; the CSV contains 14 unique records with a non-empty extrasystole field, three of which are AF and are excluded, leaving 11 mapped arrhythmia records.

**Tachycardia caveat:** four records are insufficient for a well-powered population comparison. All tachycardia results are preliminary and underpowered even though they yield 49 complete Lead-II beats.

## Interval definitions

All intervals are measured in seconds:

- P-wave duration: P offset − P onset
- PR interval: QRS onset − P onset
- QRS duration: QRS offset − QRS onset
- QT interval: T offset − QRS onset
- ST segment: T onset − QRS offset
- T-wave duration: T offset − T onset

## Synthetic timing ground truth and its limitation

The synthetic generator does **not** define finite wave onsets or offsets. It constructs each beat from five Gaussian components specified by a center and σ width. Gaussian support is formally infinite. Therefore, a transparent operational envelope of center ±2σ is used as the synthetic onset/offset proxy:

- P boundaries: P center ±2σP
- QRS boundaries: earliest of Q/R/S center −2σ through latest center +2σ
- T boundaries: T center ±2σT

This is a deterministic conversion of the actual construction parameters, not a signal detector, but it is not equivalent to a clinically annotated boundary. The comparison must be interpreted with this limitation.

Normal sinus rhythm, bradycardia, and tachycardia share the same timing centers and widths; their class adjustments affect amplitudes, not timing. Consequently, their synthetic P/T durations are fixed and their other timing distributions differ only through the common random center jitter. Arrhythmia varies QRS width on normal beats and uses a separate ectopic template, so it has additional timing variation.

Synthetic timing is sampled from 500 records per class with seed `6262`. For every in-window beat, `_generate_beat_train` supplies beat type and `_beat_morphology` supplies the exact Gaussian centers and widths.

## Statistics and figure

For each of four classes and six intervals, the output reports n, mean, sample SD, median, and IQR for each source, plus the two-sample Kolmogorov–Smirnov statistic/p-value and Wasserstein distance. Combinations with fewer than 20 usable LUDB beats are flagged and should not be interpreted quantitatively. The figure shows QRS and QT distributions, the two most standard intervals, in a 4×2 grid using the same palette as the RR comparison.
