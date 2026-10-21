"""Network architectures (Methods: universal bias model; TF-specific residual model).

Layer order and parameter names are kept exactly as in the original training
code: the bundled bias-model checkpoint loads into ``UpdatedCNN`` by name, and
parameters are created in the same order, so seeded runs draw the same random
initial weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

_BASE_INDEX = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(b"ACGT"):
    _BASE_INDEX[_b] = _i


class SimpleCNN(nn.Module):
    """TF-specific residual branch: one convolution, ReLU, global max pooling, one logit."""

    def __init__(self, num_filters=100, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.global_max_pool(x).squeeze(-1)
        return self.fc(x)


class UpdatedCNN(nn.Module):
    """Universal bias model: stacked convolution + batch norm + ReLU, max pooling,
    optional dropout, a 64-unit hidden layer and one logit."""

    def __init__(self, num_filters=128, kernel_size=7, n_conv_layers=2, use_dropout=False, dropout_rate=0.2):
        super().__init__()
        self.use_dropout = use_dropout
        layers = []
        in_channels = 4
        for _ in range(n_conv_layers):
            layers += [nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(num_filters),
                       nn.ReLU()]
            in_channels = num_filters
        self.conv_stack = nn.Sequential(*layers)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None
        self.fc_hidden = nn.Linear(num_filters, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.global_max_pool(x).squeeze(-1)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc_hidden(x))
        return self.out(x)


class ResidualModel(nn.Module):
    """f_SELEX(x) = alpha * f_BIAS(x) + f_RES(x), with alpha = tanh(bias_scale).

    The bias model's parameters are frozen (no gradients). As in the original
    code, ``model.train()`` also puts the bias model's batch-norm and dropout
    layers in training mode, so its batch-norm running statistics adapt to the
    paired-cycle data during training; its learned weights do not change.
    """

    def __init__(self, bias_model, num_filters=128, kernel_size=7):
        super().__init__()
        self.bias_model = bias_model
        self.bias_scale = nn.Parameter(torch.zeros(1))
        self.residual_branch = SimpleCNN(num_filters=num_filters, kernel_size=kernel_size)

    def forward(self, x, return_residual=False):
        with torch.no_grad():
            bias_logit = self.bias_model(x)
        residual_logit = self.residual_branch(x)
        tf_logit = torch.tanh(self.bias_scale) * bias_logit + residual_logit
        if return_residual:
            return tf_logit, residual_logit
        return tf_logit

    @property
    def alpha(self) -> float:
        return torch.tanh(self.bias_scale).item()


def one_hot_encode(sequences) -> np.ndarray:
    """(N, 4, L) float32 one-hot array, channels A, C, G, T; other characters stay all-zero."""
    seqs = [str(s) for s in sequences]
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(f"sequences have different lengths {sorted(lengths)[:5]}; "
                         "trim them to a common length first (e.g. --trim)")
    length = lengths.pop() if lengths else 0
    codes = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), length)
    idx = _BASE_INDEX[codes]
    out = np.zeros((len(seqs), 4, length), dtype=np.float32)
    rows, cols = np.nonzero(idx >= 0)
    out[rows, idx[rows, cols], cols] = 1.0
    return out


class SequenceDataset(Dataset):
    """One-hot encoded sequences with float labels (late cycle = 1, early cycle = 0)."""

    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.X = torch.tensor(one_hot_encode(seqs), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]
