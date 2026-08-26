# Data

The full publication dataset contains 12,000 synthetic 10-second ECG segments,
six balanced classes, 2,500 samples per segment, and a 250 Hz sampling rate. It
is generated deterministically with random seed 42.

The complete `synthetic_ecg_dataset.npy` file is approximately 120 MB, which
exceeds GitHub's 100 MB single-file limit. The repository therefore includes:

- `synthetic_ecg_sample_1200.npy`: balanced 1,200-segment sample (200 per class)
- `sample_labels.csv`: labels aligned to the sample array
- `sample_metadata.csv`: generation metadata aligned to the sample array
- `mitdb_subset_provenance.csv`: provenance for the external-data experiment

Generate the complete dataset locally with:

```bash
python main.py --full --force-generate
```

The generated full files are:

- `data/synthetic_ecg_dataset.npy`
- `data/labels.csv`
- `data/metadata.csv`

The synthetic data and sample subset are licensed under CC BY 4.0. If you use
them, attribution to the accompanying article is required; see `DATA_LICENSE`
and `CITATION.cff`.

MIT-BIH recordings are not redistributed. The external-validation workflow
uses WFDB/PhysioNet and records its provenance separately.
