"""Seeding, exactly as in the original training scripts."""

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch (CPU and CUDA) and make cuDNN deterministic.

    Called at the start of each pair's training run.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
