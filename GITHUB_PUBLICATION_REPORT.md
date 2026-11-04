# GitHub Publication Report

- Public repository: https://github.com/umutsaray/digital-twin-synthetic-ecg
- Published commit: `e84afd265cf784b7dfe7cb16dd4189c382164546`
- Repository contents: source code, reproducibility entry points, requirements, numerical results, figures, reports, citation metadata, and dataset documentation.
- Dataset strategy: the complete NumPy array is approximately 120 MB and exceeds GitHub's 100 MB single-file limit. A balanced 1,200-record sample (200 records per class), aligned labels/metadata, and the deterministic full-data generator are published instead. No Zenodo archive is used.
- Licensing: code is MIT licensed; synthetic data is CC BY 4.0 licensed with an explicit requirement to cite the accompanying article.
- Citation metadata: `CITATION.cff` and the README contain placeholders for author and publication metadata. These must be replaced after the definitive author list and journal/DOI details are available.
- Manuscript: only the paragraph under **Data and Code Availability** was updated with the public repository URL, licensing, and citation request. Automated comparison confirmed all other paragraphs, 11 tables, and 12 figures were preserved.
- Document QA: the updated manuscript rendered successfully as 15 pages; all pages were visually inspected, and no accessibility findings were reported.

## Ectopic RR sequencing correction (2026-11-04)

- Corrected the arrhythmia generator so that the shortened coupling interval ends at the beat labelled `ectopic`, while the compensatory pause is applied to the distinct subsequent interval.
- Verified 2,809 complete ectopic events: mean pre-ectopic RR `0.4361 s`, mean within-record median RR `0.7397 s`, and mean post-ectopic RR `1.0432 s`.
- Regenerated only the 2,000 arrhythmia records. SHA-256 checks confirmed that all five other class arrays remained byte-for-byte unchanged.
- Re-ran the four-model holdout experiment, five-fold cross-validation, calibration, MC-dropout uncertainty, noise robustness, Grad-CAM fidelity, and MIT-BIH zero-shot/classifier-only adaptation workflows.
- Headline changes: 1D-CNN synthetic-test macro F1 `0.9900 -> 0.9978`; MIT-BIH zero-shot macro F1 `0.1603 -> 0.1275`; MIT-BIH classifier-only fine-tuned macro F1 `0.3382 -> 0.3006`.
- Full before/after details are provided in `ECTOPIC_FIX_BEFORE_AFTER.md` and `results/ectopic_fix_before_after_metrics.csv`.
