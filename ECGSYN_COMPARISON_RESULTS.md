# ECGSYN Generator Comparison Results

Lower Wasserstein distance means closer agreement with the relevant real-data distribution. All distances are in seconds. `Direct` is our generator vs. ECGSYN and is descriptive rather than a distance from real data.

## RR intervals vs. MIT-BIH

| Class | Our vs. real | ECGSYN vs. real | Direct | Closer to real |
|---|---:|---:|---:|---|
| Normal sinus rhythm | 0.0205 | 0.0241 | 0.0051 | Our generator |
| Bradycardia | 0.2573 | 0.2628 | 0.0091 | Our generator |
| Tachycardia | 0.1384 | 0.1411 | 0.0058 | Our generator |

Our generator is marginally closer on all three RR classes. The mean class-level distance is 0.1387 s for our generator and 0.1427 s for ECGSYN. The direct distances are small, showing that their RR distributions are similar relative to their larger mismatch with the mapped real subsets, especially for bradycardia.

## P/QRS/T timing vs. LUDB

| Class | Interval | Our vs. real | ECGSYN vs. real | Direct | Closer to real |
|---|---|---:|---:|---:|---|
| Normal | P duration | 0.0494 | 0.0672 | 0.0185 | Our generator |
| Normal | PR interval | 0.0494 | 0.0107 | 0.0497 | ECGSYN |
| Normal | QRS duration | 0.0611 | 0.0609 | 0.0071 | ECGSYN (near tie) |
| Normal | QT interval | 0.1292 | 0.0434 | 0.0867 | ECGSYN |
| Normal | ST segment | 0.0774 | 0.0589 | 0.0186 | ECGSYN |
| Normal | T duration | 0.1456 | 0.0413 | 0.1051 | ECGSYN |
| Bradycardia | P duration | 0.0499 | 0.0871 | 0.0376 | Our generator |
| Bradycardia | PR interval | 0.0526 | 0.0838 | 0.0332 | Our generator |
| Bradycardia | QRS duration | 0.0565 | 0.0886 | 0.0321 | Our generator |
| Bradycardia | QT interval | 0.0956 | 0.1584 | 0.0633 | Our generator |
| Bradycardia | ST segment | 0.0951 | 0.0226 | 0.0754 | ECGSYN |
| Bradycardia | T duration | 0.1337 | 0.0896 | 0.0442 | ECGSYN |
| Tachycardia | P duration | 0.0615 | 0.0401 | 0.0214 | ECGSYN |
| Tachycardia | PR interval | 0.0751 | 0.0638 | 0.1388 | ECGSYN |
| Tachycardia | QRS duration | 0.0628 | 0.0744 | 0.0116 | Our generator |
| Tachycardia | QT interval | 0.1679 | 0.0093 | 0.1589 | ECGSYN |
| Tachycardia | ST segment | 0.0614 | 0.0813 | 0.0200 | Our generator |
| Tachycardia | T duration | 0.1666 | 0.0169 | 0.1506 | ECGSYN |

Timing results are mixed: ECGSYN is closer in 11 of 18 class–interval comparisons and our generator in 7. Averaged over six intervals, the distances are:

| Class | Our generator | ECGSYN |
|---|---:|---:|
| Normal sinus rhythm | 0.0854 | 0.0471 |
| Bradycardia | 0.0806 | 0.0883 |
| Tachycardia | 0.0992 | 0.0476 |
| Overall (18 comparisons) | 0.0884 | 0.0610 |

ECGSYN is therefore closer overall on the timing axis, particularly for normal and tachycardia QT and T duration. Our generator is closer overall for bradycardia timing and is better on several P/QRS/ST comparisons.

## Interpretation limit

The timing ranking is not a fully method-equivalent contest. ECGSYN is measured by an external DWT waveform delineator, whereas our generator is measured from its construction parameters using a ±2σ Gaussian boundary proxy. Detection behavior and the proxy definition can therefore contribute to the reported differences. This caveat does not apply in the same way to RR intervals, for which both comparisons use explicit interval samples.

