"""Explainability utilities for ECG models."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def grad_cam_1d(model: torch.nn.Module, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
    """Compute 1D Grad-CAM for models exposing `features`, `pool`, and `classifier`."""
    model.eval()
    model.zero_grad(set_to_none=True)
    features = model.features(x)
    features.retain_grad()
    logits = model.classifier(model.pool(features).squeeze(-1))
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())
    score = logits[:, target_class].sum()
    score.backward()
    gradients = features.grad
    weights = gradients.mean(dim=2, keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
    return _normalize(cam.detach().cpu().numpy()[0, 0])


def saliency_1d(model: torch.nn.Module, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
    """Fallback gradient saliency for any differentiable classifier."""
    model.eval()
    model.zero_grad(set_to_none=True)
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())
    logits[:, target_class].sum().backward()
    saliency = x.grad.detach().abs().cpu().numpy()[0, 0]
    return _normalize(saliency)


def save_cnn_explainability_examples(
    model: torch.nn.Module,
    dataset,
    class_names: List[str],
    output_dir: str | Path,
    device: torch.device,
    sampling_rate_after_downsample: float,
    max_classes: int = 6,
) -> Path:
    """Save Grad-CAM examples for one test sample per class."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    seen = set()
    for local_idx in range(len(dataset)):
        _, label = dataset[local_idx]
        label_int = int(label)
        if label_int not in seen:
            selected.append((local_idx, label_int))
            seen.add(label_int)
        if len(selected) >= min(max_classes, len(class_names)):
            break

    if not selected:
        raise ValueError("No samples available for explainability.")

    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 2.3 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, (local_idx, label_int) in zip(axes, selected):
        x, _ = dataset[local_idx]
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = int(torch.argmax(model(x_batch), dim=1).item())

        if all(hasattr(model, attr) for attr in ["features", "pool", "classifier"]):
            importance = grad_cam_1d(model, x_batch, target_class=prediction)
            title_kind = "Grad-CAM"
        else:
            importance = saliency_1d(model, x_batch, target_class=prediction)
            title_kind = "Saliency"

        signal = x.numpy()[0]
        t = np.arange(signal.size) / sampling_rate_after_downsample
        y_min = float(signal.min())
        y_max = float(signal.max())
        ax.imshow(
            importance[np.newaxis, :],
            aspect="auto",
            cmap="inferno",
            alpha=0.35,
            extent=[t[0], t[-1], y_min, y_max],
            origin="lower",
        )
        ax.plot(t, signal, color="#0f172a", linewidth=0.9)
        ax.set_ylabel(class_names[label_int].replace("_", "\n"), fontsize=8)
        ax.set_title(
            f"{title_kind}: true {class_names[label_int]} / predicted {class_names[prediction]}",
            fontsize=9,
        )
        ax.grid(alpha=0.20)

    axes[-1].set_xlabel("Time (seconds, after training downsampling)")
    fig.tight_layout()
    output_path = output_dir / "cnn1d_gradcam_examples.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

