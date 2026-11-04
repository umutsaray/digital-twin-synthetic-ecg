# Synthetic versus LUDB P/QRS/T Timing Results

All 24 class-by-interval combinations had at least 20 complete LUDB beats, so none was dropped. Tachycardia nevertheless remains preliminary and underpowered because its 49 complete beats come from only four records.

| Class | Interval | Synthetic n | Synthetic mean ± SD (s) | LUDB n | LUDB mean ± SD (s) | KS D | KS p | Wasserstein (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Normal sinus | P duration | 6051 | 0.160 ± 0.000 | 956 | 0.111 ± 0.020 | 0.986 | <1e-300 | 0.049 |
| Normal sinus | PR interval | 6051 | 0.212 ± 0.007 | 956 | 0.163 ± 0.028 | 0.870 | 4.89e-322 | 0.049 |
| Normal sinus | QRS duration | 6051 | 0.155 ± 0.006 | 956 | 0.094 ± 0.016 | 0.984 | <1e-300 | 0.061 |
| Normal sinus | QT interval | 6051 | 0.522 ± 0.013 | 956 | 0.393 ± 0.037 | 0.981 | <1e-300 | 0.129 |
| Normal sinus | ST segment | 6051 | 0.039 ± 0.013 | 956 | 0.116 ± 0.039 | 0.914 | 2.57e-322 | 0.077 |
| Normal sinus | T duration | 6051 | 0.328 ± 0.000 | 956 | 0.182 ± 0.030 | 1.000 | <1e-300 | 0.146 |
| Bradycardia | P duration | 3765 | 0.160 ± 0.000 | 115 | 0.111 ± 0.022 | 0.974 | 2.27e-161 | 0.050 |
| Bradycardia | PR interval | 3765 | 0.212 ± 0.007 | 115 | 0.161 ± 0.029 | 0.893 | 2.98e-106 | 0.053 |
| Bradycardia | QRS duration | 3765 | 0.155 ± 0.006 | 115 | 0.099 ± 0.021 | 0.937 | 7.32e-129 | 0.057 |
| Bradycardia | QT interval | 3765 | 0.522 ± 0.013 | 115 | 0.427 ± 0.041 | 0.934 | 5.28e-127 | 0.096 |
| Bradycardia | ST segment | 3765 | 0.039 ± 0.013 | 115 | 0.134 ± 0.043 | 0.960 | 3.15e-146 | 0.095 |
| Bradycardia | T duration | 3765 | 0.328 ± 0.000 | 115 | 0.194 ± 0.035 | 1.000 | 6.20e-224 | 0.134 |
| Tachycardia* | P duration | 10198 | 0.160 ± 0.000 | 49 | 0.099 ± 0.012 | 1.000 | <1e-300 | 0.061 |
| Tachycardia* | PR interval | 10198 | 0.212 ± 0.007 | 49 | 0.137 ± 0.015 | 1.000 | <1e-300 | 0.075 |
| Tachycardia* | QRS duration | 10198 | 0.155 ± 0.006 | 49 | 0.092 ± 0.009 | 1.000 | <1e-300 | 0.063 |
| Tachycardia* | QT interval | 10198 | 0.522 ± 0.013 | 49 | 0.354 ± 0.016 | 1.000 | <1e-300 | 0.168 |
| Tachycardia* | ST segment | 10198 | 0.039 ± 0.013 | 49 | 0.100 ± 0.015 | 0.992 | 3.95e-104 | 0.061 |
| Tachycardia* | T duration | 10198 | 0.328 ± 0.000 | 49 | 0.161 ± 0.019 | 1.000 | <1e-300 | 0.167 |
| Arrhythmia | P duration | 6317 | 0.165 ± 0.013 | 63 | 0.114 ± 0.018 | 1.000 | 1.07e-152 | 0.051 |
| Arrhythmia | PR interval | 6317 | 0.198 ± 0.032 | 63 | 0.181 ± 0.030 | 0.592 | 2.52e-21 | 0.024 |
| Arrhythmia | QRS duration | 6317 | 0.170 ± 0.028 | 63 | 0.091 ± 0.015 | 0.997 | 2.30e-136 | 0.078 |
| Arrhythmia | QT interval | 6317 | 0.535 ± 0.033 | 63 | 0.391 ± 0.033 | 0.994 | 9.69e-125 | 0.144 |
| Arrhythmia | ST segment | 6317 | 0.024 ± 0.035 | 63 | 0.115 ± 0.027 | 0.976 | 1.90e-97 | 0.091 |
| Arrhythmia | T duration | 6317 | 0.342 ± 0.037 | 63 | 0.185 ± 0.025 | 1.000 | 1.07e-152 | 0.156 |

\* Tachycardia values derive from only four LUDB records and must be treated as preliminary.

## Plain-language interpretation

- **Normal sinus rhythm:** All six synthetic distributions differ strongly from LUDB. The ±2σ synthetic template produces longer P, PR, QRS, QT, and T measurements, while its ST segment is much shorter. The synthetic P and T durations are effectively fixed rather than physiologically distributed.
- **Bradycardia:** The same systematic template mismatch remains. Bradycardia does not receive class-specific timing parameters in the generator, so its synthetic timing is nearly identical to normal rhythm despite different LUDB timing.
- **Tachycardia:** Every interval shows a large discrepancy, but the comparison is underpowered at the record level. The generator changes tachycardia amplitudes but not P/QRS/T timing, so rate-dependent shortening visible in LUDB is absent from the synthetic template.
- **Arrhythmia:** PR has the closest agreement (Wasserstein 0.024 s), but the other intervals remain substantially different. Variable/ectopic synthetic QRS timing increases spread, yet QRS and QT are still markedly longer than LUDB and ST is shorter.

Overall, the synthetic morphology timing does **not** reproduce LUDB P/QRS/T timing distributions. This is partly a genuine generator limitation—timing is mostly shared across rhythm classes—and partly a boundary-definition limitation because the Gaussian generator has no finite annotated onset/offset. The results should be presented as evidence that the synthetic framework is a controlled methodological test bed, not a physiologically calibrated waveform simulator.
