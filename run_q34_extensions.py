"""Publication-strength extension analyses for the synthetic ECG manuscript.

All deterministic sampling uses seed 42. Outputs are written under
``results/q34_extensions`` and ``figures/q34_extensions``.
"""

from __future__ import annotations

import json
import os
import platform
import random
import itertools
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import wfdb
from scipy.signal import resample_poly
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

from src.explainability import grad_cam_1d
from src.generate_dataset import CLASS_NAMES
from src.models import MODEL_REGISTRY, get_model
from src.train_models import (
    ECGDataset,
    make_loader,
    make_splits,
    predict_probabilities,
    resolve_device,
    seed_everything,
    train_one_model,
)


SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_ecg_dataset.npy"
LABELS_PATH = ROOT / "data" / "labels.csv"
RESULTS = ROOT / "results" / "q34_extensions"
FIGURES = ROOT / "figures" / "q34_extensions"
REAL_DATA = ROOT / "data" / "mitdb_subset"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
REAL_DATA.mkdir(parents=True, exist_ok=True)


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "wfdb": wfdb.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": str(SEED),
    }


def load_checkpoint(name: str, path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = get_model(name, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def synthetic_test_loader(batch_size: int = 128):
    labels = pd.read_csv(LABELS_PATH)["label_id"].to_numpy(np.int64)
    _, _, test_idx = make_splits(labels, seed=SEED)
    dataset = ECGDataset(DATA_PATH, labels, test_idx, downsample_factor=5)
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=SEED)
    return labels, test_idx, dataset, loader


def train_improved_lstm(device: torch.device) -> pd.DataFrame:
    cached = RESULTS / "lstm_improved_performance.csv"
    checkpoint = RESULTS / "best_models" / "lstm_improved.pt"
    if cached.exists() and checkpoint.exists():
        return pd.read_csv(cached)
    labels = pd.read_csv(LABELS_PATH)["label_id"].to_numpy(np.int64)
    train_idx, val_idx, test_idx = make_splits(labels, seed=SEED)
    train_ds = ECGDataset(DATA_PATH, labels, train_idx, downsample_factor=5)
    val_ds = ECGDataset(DATA_PATH, labels, val_idx, downsample_factor=5)
    test_ds = ECGDataset(DATA_PATH, labels, test_idx, downsample_factor=5)
    train_loader = make_loader(train_ds, batch_size=64, shuffle=True, seed=SEED)
    val_loader = make_loader(val_ds, batch_size=64, shuffle=False, seed=SEED)
    test_loader = make_loader(test_ds, batch_size=64, shuffle=False, seed=SEED)
    seed_everything(SEED)
    metric = train_one_model(
        model_name="lstm_improved",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=CLASS_NAMES,
        results_dir=RESULTS,
        epochs=25,
        learning_rate=5e-4,
        device=device,
        sampling_rate_hz=250,
        early_stopping_patience=6,
        run_name="lstm_improved",
        save_artifacts=True,
        save_checkpoint=True,
        generate_explainability=False,
    )
    out = pd.DataFrame([metric])
    out.to_csv(RESULTS / "lstm_improved_performance.csv", index=False)
    comparison = pd.read_csv(ROOT / "results" / "model_performance.csv")
    original = comparison.loc[comparison.model == "lstm"].copy()
    original["variant"] = "Original final-state BiLSTM"
    improved = out.copy()
    improved["variant"] = "Improved attention-pooled BiLSTM"
    cols = ["variant", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro", "best_epoch", "train_time_sec", "parameters"]
    pd.concat([original, improved], ignore_index=True)[cols].to_csv(RESULTS / "lstm_baseline_comparison.csv", index=False)
    return out


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += mask.mean() * gap
        mce = max(mce, gap)
    onehot = np.eye(probs.shape[1])[y_true]
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))
    nll = -np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)))
    return {"ece_15": float(ece), "mce_15": float(mce), "multiclass_brier": float(brier), "nll": float(nll)}


def run_calibration(device: torch.device):
    _, _, _, loader = synthetic_test_loader()
    rows = []
    predictions = {}
    for name in ["cnn1d", "lstm", "cnn_lstm", "transformer"]:
        model = load_checkpoint(name, ROOT / "results" / "best_models" / f"{name}.pt", device)
        y, p = predict_probabilities(model, loader, device)
        predictions[name] = (y, p)
        rows.append({"model": name, "display_name": MODEL_REGISTRY[name], **calibration_metrics(y, p)})
        np.savez_compressed(RESULTS / f"{name}_holdout_predictions.npz", y_true=y, probabilities=p)
    improved_path = RESULTS / "best_models" / "lstm_improved.pt"
    if improved_path.exists():
        model = load_checkpoint("lstm_improved", improved_path, device)
        y, p = predict_probabilities(model, loader, device)
        predictions["lstm_improved"] = (y, p)
        rows.append({"model": "lstm_improved", "display_name": MODEL_REGISTRY["lstm_improved"], **calibration_metrics(y, p)})
        np.savez_compressed(RESULTS / "lstm_improved_holdout_predictions.npz", y_true=y, probabilities=p)
    pd.DataFrame(rows).to_csv(RESULTS / "calibration_metrics.csv", index=False)

    y, p = predictions["cnn1d"]
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            xs.append(conf[mask].mean()); ys.append(correct[mask].mean()); ns.append(mask.sum())
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f6f8b", linewidth=2, label="1D-CNN")
    for x, yy, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yy), xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean predicted confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1.08))
    ax.set_title("1D-CNN reliability diagram", pad=8)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "cnn1d_reliability_diagram.png", dpi=300); plt.close(fig)
    return predictions


def enable_mc_dropout(model: nn.Module):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def run_uncertainty(device: torch.device, passes: int = 30):
    _, _, _, loader = synthetic_test_loader(batch_size=128)
    model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    enable_mc_dropout(model)
    all_probs = []
    y_ref = None
    for pass_id in range(passes):
        torch.manual_seed(SEED + pass_id)
        enable_mc_dropout(model)
        probs, targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                targets.append(yb.numpy())
        y, p = np.concatenate(targets), np.concatenate(probs)
        y_ref = y
        all_probs.append(p)
    stack = np.stack(all_probs)
    mean_p = stack.mean(axis=0)
    predictive_entropy = -np.sum(mean_p * np.log(np.clip(mean_p, 1e-12, 1)), axis=1)
    expected_entropy = -np.mean(np.sum(stack * np.log(np.clip(stack, 1e-12, 1)), axis=2), axis=0)
    mutual_information = predictive_entropy - expected_entropy
    variation_ratio = 1.0 - np.max(np.apply_along_axis(lambda x: np.bincount(x, minlength=6), 0, stack.argmax(axis=2)), axis=0) / passes
    pred = mean_p.argmax(axis=1)
    df = pd.DataFrame({"y_true": y_ref, "y_pred": pred, "predictive_entropy": predictive_entropy, "mutual_information": mutual_information, "variation_ratio": variation_ratio, "correct": pred == y_ref})
    df.to_csv(RESULTS / "mc_dropout_sample_uncertainty.csv", index=False)
    summary = df.groupby("y_true").agg(n=("y_true", "size"), predictive_entropy_mean=("predictive_entropy", "mean"), predictive_entropy_std=("predictive_entropy", "std"), mutual_information_mean=("mutual_information", "mean"), error_rate=("correct", lambda s: 1-s.mean())).reset_index()
    summary["class_name"] = summary.y_true.map(dict(enumerate(CLASS_NAMES)))
    summary.to_csv(RESULTS / "mc_dropout_uncertainty_by_class.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ordered = summary.sort_values("predictive_entropy_mean")
    ax.barh(ordered.class_name.str.replace("_", " "), ordered.predictive_entropy_mean, xerr=ordered.predictive_entropy_std, color="#6B8E9F", alpha=.9)
    ax.set(xlabel="Predictive entropy (mean +/- SD)", title=f"1D-CNN uncertainty by class ({passes} MC-dropout passes)")
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "mc_dropout_uncertainty_by_class.png", dpi=300); plt.close(fig)


def run_robustness(device: torch.device):
    labels, test_idx, _, _ = synthetic_test_loader()
    raw = np.load(DATA_PATH, mmap_mode="r")
    model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    snrs = [np.inf, 20, 15, 10, 5, 0]
    rng = np.random.default_rng(SEED)
    rows = []
    for snr in snrs:
        probs, targets = [], []
        for idx in test_idx:
            x = np.asarray(raw[idx], dtype=np.float32).copy()
            if labels[idx] == 0 and np.isfinite(snr):
                signal_power = float(np.mean(x**2) + 1e-12)
                noise_power = signal_power / (10 ** (snr / 10))
                gaussian = rng.normal(0, np.sqrt(noise_power * .75), x.size)
                t = np.arange(x.size) / 250.0
                phase = rng.uniform(0, 2*np.pi)
                wander = np.sqrt(noise_power * .25) * np.sqrt(2) * np.sin(2*np.pi*.25*t + phase)
                x = x + gaussian + wander
            x = x[:2500].reshape(-1, 5).mean(axis=1)
            x = (x-x.mean())/(x.std()+1e-6)
            with torch.no_grad():
                p = torch.softmax(model(torch.tensor(x[None, None], dtype=torch.float32, device=device)), dim=1).cpu().numpy()[0]
            probs.append(p); targets.append(labels[idx])
        probs = np.asarray(probs); targets = np.asarray(targets); pred = probs.argmax(axis=1)
        normal_mask = targets == 0
        rows.append({"target_snr_db": "clean" if not np.isfinite(snr) else int(snr), "overall_macro_f1": f1_score(targets, pred, average="macro", zero_division=0), "overall_accuracy": accuracy_score(targets, pred), "normal_recall": recall_score(targets[normal_mask], pred[normal_mask], labels=[0], average="macro", zero_division=0)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "noise_robustness.csv", index=False)
    plot = out.iloc[::-1].copy()
    xlabels = plot.target_snr_db.astype(str)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(xlabels, plot.overall_macro_f1, "o-", label="Overall macro F1", color="#1f6f8b")
    ax.plot(xlabels, plot.normal_recall, "s-", label="Normal-class recall", color="#c55a3d")
    ax.set(xlabel="Injected-noise target SNR (dB)", ylabel="Score", title="1D-CNN robustness to noise on normal-sinus test segments", ylim=(0, 1.02))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "noise_robustness_curve.png", dpi=300); plt.close(fig)


def run_robustness_diagnostics(device: torch.device):
    """Confirm whether the low-SNR plateau reflects identical inputs or classifier collapse."""
    labels, test_idx, _, _ = synthetic_test_loader()
    normal_idx = test_idx[labels[test_idx] == 0]
    raw = np.load(DATA_PATH, mmap_mode="r")
    model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    rng = np.random.default_rng(SEED)
    levels = [20, 15, 10, 5, 0]
    processed, predictions = {}, {}
    for snr in levels:
        xs = []
        for idx in normal_idx:
            x = np.asarray(raw[idx], dtype=np.float32).copy()
            signal_power = float(np.mean(x**2) + 1e-12)
            noise_power = signal_power / (10 ** (snr / 10))
            gaussian = rng.normal(0, np.sqrt(noise_power * .75), x.size)
            t = np.arange(x.size) / 250.0
            phase = rng.uniform(0, 2*np.pi)
            wander = np.sqrt(noise_power * .25) * np.sqrt(2) * np.sin(2*np.pi*.25*t + phase)
            x = x + gaussian + wander
            x = x[:2500].reshape(-1, 5).mean(axis=1)
            x = ((x-x.mean())/(x.std()+1e-6)).astype(np.float32)
            xs.append(x)
        arr = np.stack(xs)
        processed[snr] = arr
        with torch.no_grad():
            probs = []
            for start in range(0, len(arr), 128):
                xb = torch.tensor(arr[start:start+128, None], device=device)
                probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        predictions[snr] = np.concatenate(probs).argmax(axis=1)
    rows = []
    for a, b in [(10, 5), (5, 0), (10, 0)]:
        xa, xb = processed[a], processed[b]
        correlations = [np.corrcoef(u, v)[0, 1] for u, v in zip(xa, xb)]
        mse = np.mean((xa-xb)**2, axis=1)
        rows.append({"snr_pair": f"{a}_vs_{b}", "mean_sample_correlation": np.mean(correlations), "std_sample_correlation": np.std(correlations), "mean_normalized_mse": np.mean(mse), "identical_prediction_fraction": np.mean(predictions[a] == predictions[b])})
    pd.DataFrame(rows).to_csv(RESULTS / "robustness_floor_diagnostics.csv", index=False)
    dist_rows = []
    for snr in [10, 5, 0]:
        counts = np.bincount(predictions[snr], minlength=6)
        for cls, count in enumerate(counts):
            dist_rows.append({"target_snr_db": snr, "predicted_class": CLASS_NAMES[cls], "count": int(count), "fraction": count/len(normal_idx)})
    pd.DataFrame(dist_rows).to_csv(RESULTS / "robustness_floor_prediction_distribution.csv", index=False)


def run_domain_adaptation(device: torch.device):
    """Fine-tune only the 1D-CNN classifier using a record-disjoint MIT-BIH subset."""
    cached = np.load(REAL_DATA / "mitdb_mapped_10s_segments.npz")
    x = cached["signals"].astype(np.float32)
    y = cached["labels"].astype(np.int64)
    provenance = pd.read_csv(RESULTS / "mitdb_subset_provenance.csv")
    groups = provenance["record"].astype(str).to_numpy()
    unique = sorted(np.unique(groups).tolist())
    best = None
    for combo in itertools.combinations(unique, 4):
        train_mask = np.isin(groups, combo)
        train_counts = np.bincount(y[train_mask], minlength=4)
        test_counts = np.bincount(y[~train_mask], minlength=4)
        if np.all(train_counts > 0) and np.all(test_counts > 0):
            score = min(train_counts.min(), test_counts.min()) + .01 * train_counts.sum()
            if best is None or score > best[0]:
                best = (score, combo, train_mask, train_counts, test_counts)
    if best is None:
        raise RuntimeError("Could not find a record-disjoint split containing all four mapped classes.")
    _, train_records, train_mask, train_counts, test_counts = best
    test_mask = ~train_mask

    base_model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    with torch.no_grad():
        p0 = torch.softmax(base_model(torch.tensor(x[test_mask, None], device=device)), dim=1).cpu().numpy()
    zero_pred = p0.argmax(axis=1)

    model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    weights = len(y[train_mask]) / (4 * np.maximum(train_counts, 1))
    class_weights = torch.tensor(np.r_[weights, [1.0, 1.0]], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    train_ds = TensorDataset(torch.tensor(x[train_mask, None]), torch.tensor(y[train_mask]))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=generator)
    torch.manual_seed(SEED)
    for _ in range(60):
        model.features.eval(); model.classifier.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = model.pool(model.features(xb)).squeeze(-1)
            logits = model.classifier(features)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        p1 = torch.softmax(model(torch.tensor(x[test_mask, None], device=device)), dim=1).cpu().numpy()
    adapted_pred = p1.argmax(axis=1)
    yt = y[test_mask]
    rows = []
    for method, pred in [("Zero-shot", zero_pred), ("Classifier-only fine-tuning", adapted_pred)]:
        rows.append({"method": method, "train_records": ",".join(train_records), "test_records": ",".join(sorted(set(groups[test_mask]))), "n_train_segments": int(train_mask.sum()), "n_test_segments": int(test_mask.sum()), "accuracy": accuracy_score(yt, pred), "macro_f1_4class": f1_score(yt, pred, labels=[0,1,2,3], average="macro", zero_division=0), "artifact_prediction_rate": np.mean(np.isin(pred,[4,5]))})
    pd.DataFrame(rows).to_csv(RESULTS / "domain_adaptation_metrics.csv", index=False)
    split = {"train_records": list(train_records), "test_records": sorted(set(groups[test_mask])), "train_class_counts": train_counts.tolist(), "test_class_counts": test_counts.tolist(), "epochs": 60, "trainable_component": "classifier_only", "seed": SEED}
    (RESULTS / "domain_adaptation_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    torch.save({"model_name": "cnn1d", "state_dict": model.state_dict(), "split": split}, RESULTS / "best_models" / "cnn1d_mitdb_classifier_finetuned.pt")


def run_mcnemar(predictions):
    y, p1 = predictions["cnn1d"]
    _, p2 = predictions["cnn_lstm"]
    c1 = p1.argmax(axis=1) == y
    c2 = p2.argmax(axis=1) == y
    b = int(np.sum(c1 & ~c2)); c = int(np.sum(~c1 & c2))
    exact_p = float(binomtest(min(b, c), n=b+c, p=.5, alternative="two-sided").pvalue) if b+c else 1.0
    pd.DataFrame([{"cnn1d_correct_cnnlstm_wrong": b, "cnn1d_wrong_cnnlstm_correct": c, "discordant_total": b+c, "exact_mcnemar_p": exact_p}]).to_csv(RESULTS / "mcnemar_cnn1d_vs_cnnlstm.csv", index=False)


def run_gradcam_fidelity(device: torch.device, per_class: int = 30):
    _, _, dataset, _ = synthetic_test_loader(batch_size=1)
    model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    rows = []
    selected = {i: 0 for i in range(len(CLASS_NAMES))}
    fractions = np.array([0.0, .1, .2, .3, .4, .5])
    for local_idx in range(len(dataset)):
        x, y = dataset[local_idx]; cls = int(y)
        if selected[cls] >= per_class:
            continue
        selected[cls] += 1
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            base_probs = torch.softmax(model(xb), dim=1)[0]
            pred = int(base_probs.argmax())
            base = float(base_probs[pred])
        cam = grad_cam_1d(model, xb, target_class=pred)
        order = np.argsort(cam)[::-1].copy()
        curve = []
        for frac in fractions:
            masked = xb.clone()
            n = int(round(frac * masked.shape[-1]))
            if n:
                masked[..., order[:n]] = 0.0
            with torch.no_grad():
                curve.append(float(torch.softmax(model(masked), dim=1)[0, pred]))
        auc = float(np.trapezoid(np.asarray(curve) / max(base, 1e-12), fractions) / fractions[-1])
        rows.append({"class_id": cls, "class_name": CLASS_NAMES[cls], "predicted_class": pred, "baseline_confidence": base, "confidence_drop_30pct": base-curve[3], "deletion_auc_normalized": auc})
        if all(v >= per_class for v in selected.values()):
            break
    detail = pd.DataFrame(rows); detail.to_csv(RESULTS / "gradcam_fidelity_samples.csv", index=False)
    summary = detail.groupby("class_name").agg(n=("class_name", "size"), baseline_confidence_mean=("baseline_confidence", "mean"), confidence_drop_30pct_mean=("confidence_drop_30pct", "mean"), confidence_drop_30pct_std=("confidence_drop_30pct", "std"), deletion_auc_mean=("deletion_auc_normalized", "mean"), deletion_auc_std=("deletion_auc_normalized", "std")).reset_index()
    summary.to_csv(RESULTS / "gradcam_fidelity_by_class.csv", index=False)


def preprocess_real_segment(x: np.ndarray, fs: int) -> np.ndarray:
    x = resample_poly(np.asarray(x, dtype=np.float32), 250, fs)
    if x.size < 2500:
        x = np.pad(x, (0, 2500-x.size), mode="edge")
    x = x[:2500].reshape(-1, 5).mean(axis=1)
    return ((x-x.mean())/(x.std()+1e-6)).astype(np.float32)


def run_real_data(device: torch.device):
    records = ["100", "101", "103", "105", "106", "109", "118", "119", "200", "203", "207", "208"]
    normal_symbols = {"N", "L", "R", "e", "j"}
    cache_path = REAL_DATA / "mitdb_mapped_10s_segments.npz"
    cache_provenance_path = REAL_DATA / "mitdb_subset_provenance.csv"
    if cache_path.exists():
        cached = np.load(cache_path)
        x, y = cached["signals"], cached["labels"]
        if not cache_provenance_path.exists():
            raise FileNotFoundError(
                "Cached MIT-BIH segments require data/mitdb_subset/mitdb_subset_provenance.csv. "
                "Delete the segment cache to rebuild both files from PhysioNet."
            )
        prov = pd.read_csv(cache_provenance_path)
        if len(prov) != len(y):
            raise ValueError("MIT-BIH cache and provenance contain different numbers of segments.")
        prov.to_csv(RESULTS / "mitdb_subset_provenance.csv", index=False)
    else:
        rng = np.random.default_rng(SEED)
        segments = []
        provenance = []
        for record_name in records:
            record = wfdb.rdrecord(record_name, pn_dir="mitdb", channels=[0])
            annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
            signal = record.p_signal[:, 0]
            fs = int(record.fs)
            seg_len = 10 * fs
            for start in range(0, len(signal)-seg_len+1, seg_len):
                stop = start + seg_len
                positions = annotation.sample[(annotation.sample >= start) & (annotation.sample < stop)]
                symbols = [s for pos, s in zip(annotation.sample, annotation.symbol) if start <= pos < stop and s not in {"+", "~", "|", "[", "]", "!", "x", "(", ")", "p", "t", "u", "`", "'", "^"}]
                if len(positions) < 4 or not symbols:
                    continue
                abnormal_fraction = np.mean([s not in normal_symbols for s in symbols])
                hr = len(positions) * 6.0
                if abnormal_fraction >= .10:
                    label = 3
                elif hr < 60:
                    label = 1
                elif hr > 100:
                    label = 2
                else:
                    label = 0
                segments.append((preprocess_real_segment(signal[start:stop], fs), label))
                provenance.append({"record": record_name, "start_second": start/fs, "heart_rate_bpm": hr, "abnormal_beat_fraction": abnormal_fraction, "mapped_class": CLASS_NAMES[label]})
        prov = pd.DataFrame(provenance)
        chosen = []
        for cls in range(4):
            idx = np.flatnonzero(prov.mapped_class.to_numpy() == CLASS_NAMES[cls])
            if idx.size > 250:
                idx = rng.choice(idx, 250, replace=False)
            chosen.extend(idx.tolist())
        chosen = np.asarray(sorted(chosen))
        x = np.stack([segments[i][0] for i in chosen])
        y = np.asarray([segments[i][1] for i in chosen], dtype=np.int64)
        selected_provenance = prov.iloc[chosen].reset_index(drop=True)
        selected_provenance.to_csv(RESULTS / "mitdb_subset_provenance.csv", index=False)
        selected_provenance.to_csv(cache_provenance_path, index=False)
        np.savez_compressed(cache_path, signals=x, labels=y)
    model = load_checkpoint("cnn1d", ROOT / "results" / "best_models" / "cnn1d.pt", device)
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            xb = torch.tensor(x[start:start+128, None], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs); pred = probs.argmax(axis=1)
    rows = [{"dataset": "MIT-BIH Arrhythmia Database", "n_segments": len(y), "mapped_classes": 4, "zero_shot_accuracy": accuracy_score(y, pred), "zero_shot_macro_f1_4class": f1_score(y, pred, labels=[0,1,2,3], average="macro", zero_division=0), "noise_or_fault_prediction_rate": np.mean(np.isin(pred,[4,5]))}]
    for cls in range(4):
        rows[0][f"recall_{CLASS_NAMES[cls]}"] = recall_score(y == cls, pred == cls, zero_division=0)
    pd.DataFrame(rows).to_csv(RESULTS / "external_real_data_metrics.csv", index=False)
    matrix = confusion_matrix(y, pred, labels=range(6))[:4, :]
    pd.DataFrame(matrix, index=CLASS_NAMES[:4], columns=CLASS_NAMES).to_csv(RESULTS / "external_real_data_confusion_matrix.csv")
    short_names = ["NSR", "Brady", "Tachy", "Arrh.", "Noisy", "Fault"]
    fig, ax = plt.subplots(figsize=(4.8, 3.8)); im=ax.imshow(matrix, cmap="Blues"); fig.colorbar(im, ax=ax, fraction=.046)
    ax.set_xticks(range(6), short_names, rotation=35, ha="right", fontsize=8); ax.set_yticks(range(4), short_names[:4], fontsize=8)
    ax.set(xlabel="Synthetic-trained model prediction", ylabel="Mapped MIT-BIH label", title="Zero-shot external sanity check")
    for i in range(4):
        for j in range(6): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8,color="white" if matrix[i,j] > matrix.max()/2 else "black")
    fig.tight_layout(); fig.savefig(FIGURES / "external_mitdb_confusion_matrix.png", dpi=300); plt.close(fig)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = resolve_device("auto")
    (RESULTS / "software_versions.json").write_text(json.dumps(versions(), indent=2), encoding="utf-8")
    train_improved_lstm(device)
    predictions = run_calibration(device)
    run_uncertainty(device)
    run_robustness(device)
    run_robustness_diagnostics(device)
    run_mcnemar(predictions)
    run_gradcam_fidelity(device)
    run_real_data(device)
    run_domain_adaptation(device)
    print(f"Completed extension analyses in {RESULTS}")


if __name__ == "__main__":
    main()
