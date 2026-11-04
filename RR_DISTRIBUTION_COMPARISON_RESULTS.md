# Synthetic versus MIT-BIH RR-Interval Distributions

RR intervals were pooled from 500 newly generated synthetic records per rhythm class (seed `5151`) and from the same 573 mapped MIT-BIH segments used in the external-validation experiment. Real segment labels and record selection were taken directly from `data/mitdb_subset/mitdb_subset_provenance.csv`; they were not re-derived for this analysis.

| Class | Synthetic mean ± SD (s) | Real mean ± SD (s) | Synthetic median / IQR (s) | Real median / IQR (s) | KS statistic | KS p-value | Wasserstein (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal sinus rhythm | 0.804 ± 0.086 | 0.796 ± 0.115 | 0.792 / 0.138 | 0.786 / 0.144 | 0.097 | 4.76e-16 | 0.021 |
| Bradycardia | 1.314 ± 0.171 | 1.057 ± 0.060 | 1.279 / 0.272 | 1.053 / 0.075 | 0.746 | 1.29e-94 | 0.257 |
| Tachycardia | 0.471 ± 0.054 | 0.601 ± 0.135 | 0.464 / 0.084 | 0.636 / 0.180 | 0.639 | 8.18e-280 | 0.138 |
| Arrhythmia | 0.759 ± 0.275 | 0.695 ± 0.240 | 0.725 / 0.342 | 0.633 / 0.303 | 0.156 | 3.35e-45 | 0.069 |

## Interpretation by class

- **Normal sinus rhythm:** The distributions are practically close in center and spread, with the smallest Wasserstein distance (0.021 s). The real distribution is slightly broader and has more tail mass. The KS test is significant because of the large pooled sample sizes, so the distributions are not statistically identical despite their visual similarity.
- **Bradycardia:** The synthetic distribution does **not** resemble the mapped MIT-BIH distribution well. It is shifted toward substantially longer RR intervals and is much broader; the 0.257 s Wasserstein distance is the largest of the four classes.
- **Tachycardia:** The synthetic distribution does **not** resemble the mapped MIT-BIH distribution well. Synthetic intervals are shorter and tightly concentrated, whereas the real distribution is broader, shifted toward longer intervals, and visibly more heterogeneous.
- **Arrhythmia:** The distributions show partial qualitative overlap and broad tails, but the synthetic distribution is shifted toward longer RR intervals and is somewhat broader. The difference is smaller than for bradycardia or tachycardia but remains systematic and statistically significant.

These results support only limited RR-distribution realism: normal sinus rhythm aligns reasonably well, arrhythmia aligns partially, and bradycardia/tachycardia show clear systematic mismatches. P-wave, QRS, and T-wave timing or amplitude comparisons are explicitly outside this analysis because the MIT-BIH Arrhythmia Database does not supply validated P/T fiducial annotations and adding an unvalidated delineation method would introduce a separate source of error.
