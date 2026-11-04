# Digital-Twin-Style Synthetic ECG Classification

Reproducible synthetic ECG generation, neural classification, and
trustworthy-AI evaluation framework accompanying the manuscript:

> *Towards Trustworthy Artificial Intelligence in Biomedical Decision Support:
> A Digital-Twin-Style Framework for Synthetic ECG Classification*

The project is a parameterized, rule-based methodological test bed. It is not a
real-time patient-specific physiological twin and is not a clinically validated
diagnostic system.

## What is included

- Deterministic six-class synthetic ECG generation (seed 42)
- 1D-CNN, LSTM, CNN-LSTM, compact Transformer, and attention-pooled BiLSTM
- Holdout evaluation and five-fold stratified cross-validation
- ANOVA, Tukey HSD, and exact McNemar testing
- SNR, PRD, and RMSE signal-quality assessment
- Grad-CAM visualization and deletion-fidelity testing
- ECE, Brier score, NLL, and MC-dropout uncertainty
- Noise-injection robustness and classifier-floor diagnostics
- MIT-BIH zero-shot validation and record-disjoint classifier fine-tuning
- Zero-shot OOD testing with unseen HR tails and cross-applied artifact profiles
- Synthetic-versus-MIT-BIH RR-interval distribution comparison
- Synthetic Gaussian-template versus LUDB manual P/QRS/T timing comparison
- Numerical tables, figures, predictions, checkpoints, and provenance outputs

## Dataset

The full dataset comprises 12,000 synthetic ECG segments:

- 6 balanced classes, 2,000 segments per class
- 10 seconds per segment
- 250 Hz sampling rate (2,500 samples)
- deterministic random seed 42
- classes: normal sinus rhythm, bradycardia, tachycardia, arrhythmia, noisy ECG,
  and sensor-fault ECG

The full NumPy array is approximately 120 MB and exceeds GitHub's 100 MB
single-file limit. This repository therefore contains a balanced 1,200-segment
sample and the complete deterministic generator. Run the generator to recreate
the full dataset locally. No Zenodo or other external archive is used.

See [`data/README.md`](data/README.md) for exact files and licensing details.

## Reviewer-requested generalization analyses

The repository includes two additional analyses that do not retrain the four
publication classifiers:

- [`OOD_GENERALIZATION_PROTOCOL.md`](OOD_GENERALIZATION_PROTOCOL.md) defines a
  600-record OOD test set using external heart-rate tails and cross-applied
  acquisition-artifact profiles. Results and an honest interpretation are in
  [`OOD_GENERALIZATION_RESULTS.md`](OOD_GENERALIZATION_RESULTS.md).
- [`RR_DISTRIBUTION_COMPARISON_RESULTS.md`](RR_DISTRIBUTION_COMPARISON_RESULTS.md)
  compares pooled RR intervals from the synthetic rhythm classes with the same
  mapped MIT-BIH subset used for external validation. P/QRS/T delineation is
  explicitly outside the scope of this comparison.

Reproduce them with:

```bash
python scripts/generate_ood_test_set.py
python scripts/evaluate_ood_generalization.py
python scripts/compare_rr_distributions.py
```

The follow-up P/QRS/T timing analysis uses locally downloaded LUDB 1.0.1
Lead-II annotations. Raw LUDB clinical data are gitignored; setup instructions
are in [`data/ludb/README.md`](data/ludb/README.md). Run:

```bash
python scripts/compare_pqrst_timing.py
```

The exact mapping, annotation convention, interval definitions, and synthetic
Gaussian-boundary limitation are documented in
[`PQRST_TIMING_PROTOCOL.md`](PQRST_TIMING_PROTOCOL.md); results are summarized
without physiological-equivalence claims in
[`PQRST_TIMING_RESULTS.md`](PQRST_TIMING_RESULTS.md).

### Ectopic RR sequencing correction

Version 1.0.1 corrects the arrhythmia beat-timing sequence: the short coupling
interval now leads into the beat labelled `ectopic`, and the compensatory pause
is applied to the separate following interval. Across 2,809 complete ectopic
events, the mean RR intervals before and after ectopic beats were 0.4361 s and
1.0432 s, respectively, versus a mean within-record median RR of 0.7397 s.

Only the 2,000 arrhythmia records were regenerated; hash checks confirmed that
the other five classes were unchanged. All models and downstream analyses were
then rerun. See [`ECTOPIC_FIX_BEFORE_AFTER.md`](ECTOPIC_FIX_BEFORE_AFTER.md) for
the table-level changes and `results/ectopic_rr_before_after.png` for the
verification figure.

## Installation

Python 3.12.13 was used for the manuscript analyses.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

The manuscript-matched core versions are NumPy 2.5.2, pandas 3.0.5, SciPy
1.18.1, scikit-learn 1.9.0, PyTorch 2.13.0, and WFDB 4.3.1.

## Reproduce the analyses

Generate the full dataset and run the primary publication pipeline:

```bash
python main.py --full --force-generate
```

Run a small workflow check:

```bash
python main.py --quick
```

Run the extended trustworthy-AI and external-validation analyses after the
primary checkpoints exist:

```bash
python run_q34_extensions.py
```

This extended entry point runs the improved BiLSTM, calibration, uncertainty,
noise robustness, robustness diagnostics, McNemar comparison, Grad-CAM
deletion fidelity, MIT-BIH zero-shot evaluation, and record-disjoint
classifier-only adaptation. MIT-BIH records are retrieved through WFDB when
needed and are not redistributed here.

## Code map

| Area | Implementation |
|---|---|
| Generation and configuration | `src/generate_dataset.py`, `src/config.py` |
| Model definitions | `src/models.py` |
| Training and holdout workflow | `src/train_models.py`, `src/evaluate.py`, `main.py` |
| Cross-validation | `src/train_models.py`, `main.py` |
| Statistical tests | `src/statistical_analysis.py`, `run_q34_extensions.py` |
| Signal quality | `src/signal_quality.py` |
| Grad-CAM and fidelity | `src/explainability.py`, `run_q34_extensions.py` |
| Calibration and uncertainty | `run_q34_extensions.py` |
| Noise robustness | `run_q34_extensions.py` |
| MIT-BIH validation/adaptation | `run_q34_extensions.py` |

## Outputs

The `results/`, `figures/`, and `report/` directories contain the numerical and
graphical outputs used in the manuscript, including model summaries,
cross-validation statistics, calibration, uncertainty, robustness,
explanation fidelity, and external-validation results.

## Licenses and required attribution

- **Code:** MIT License (`LICENSE`)
- **Synthetic dataset:** Creative Commons Attribution 4.0 International
  (`DATA_LICENSE`)

If you use the code or synthetic dataset, please cite the accompanying article.
Use GitHub's **Cite this repository** control or the metadata in
`CITATION.cff`. In particular, CC BY 4.0 requires attribution for use of the
dataset.

## Citation

The manuscript is currently under review / in preparation. Author, journal,
volume, issue, pages, and DOI placeholders must be updated after publication.

```bibtex
@article{TODO2026TrustworthyECG,
  author  = {TODO: AUTHOR NAME(S)},
  title   = {Towards Trustworthy Artificial Intelligence in Biomedical Decision Support: A Digital-Twin-Style Framework for Synthetic ECG Classification},
  journal = {TBD -- update after publication},
  year    = {2026},
  volume  = {TBD},
  number  = {TBD},
  pages   = {TBD},
  doi     = {TBD}
}
```

Before publication, replace the author placeholder in both this README and
`CITATION.cff`. After publication, fill in the journal, volume, issue, pages,
and DOI fields. No publication metadata has been invented.
