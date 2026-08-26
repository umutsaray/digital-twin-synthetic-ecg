"""PyTorch models for synthetic ECG arrhythmia classification."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class ECG1DCNN(nn.Module):
    """Compact 1D CNN with an exposed feature extractor for Grad-CAM."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled)


class LSTMClassifier(nn.Module):
    """Small bidirectional LSTM classifier."""

    def __init__(self, num_classes: int = 6, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(last)


class ImprovedLSTMClassifier(nn.Module):
    """BiLSTM with learned input projection and attention pooling.

    The original baseline compressed the entire 500-step sequence into the
    final hidden state. This variant preserves information across time and
    learns a normalized weighted average over all recurrent outputs.
    """

    def __init__(self, num_classes: int = 6, hidden_size: int = 32):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(1, 8),
            nn.LayerNorm(8),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.20),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(x.transpose(1, 2))
        outputs, _ = self.lstm(sequence)
        weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
        pooled = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class CNNLSTMClassifier(nn.Module):
    """Convolutional front-end followed by a lightweight LSTM."""

    def __init__(self, num_classes: int = 6, hidden_size: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=9, padding=4),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).transpose(1, 2)
        _, (hidden, _) = self.lstm(features)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SmallTransformerClassifier(nn.Module):
    """Small Transformer encoder with strided convolutional tokenization."""

    def __init__(self, num_classes: int = 6, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.position = PositionalEncoding(d_model=d_model, max_len=768)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=96,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.20),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x).transpose(1, 2)
        tokens = self.position(tokens)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def get_model(model_name: str, num_classes: int = 6) -> nn.Module:
    name = model_name.lower()
    if name in {"cnn", "cnn1d", "1d_cnn", "1d-cnn"}:
        return ECG1DCNN(num_classes=num_classes)
    if name == "lstm":
        return LSTMClassifier(num_classes=num_classes)
    if name in {"lstm_improved", "improved_lstm", "attention_lstm"}:
        return ImprovedLSTMClassifier(num_classes=num_classes)
    if name in {"cnn_lstm", "cnn-lstm", "cnnlstm"}:
        return CNNLSTMClassifier(num_classes=num_classes)
    if name in {"transformer", "small_transformer", "small-transformer"}:
        return SmallTransformerClassifier(num_classes=num_classes)
    raise ValueError(f"Unknown model '{model_name}'.")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


MODEL_REGISTRY: Dict[str, str] = {
    "cnn1d": "1D-CNN",
    "lstm": "LSTM",
    "lstm_improved": "Improved LSTM",
    "cnn_lstm": "CNN-LSTM",
    "transformer": "Small Transformer",
}
