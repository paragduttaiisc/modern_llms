import torch
import random
from typing import Tuple


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def train_test_split(
        encoded_data: torch.Tensor, test_size: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(len(encoded_data) * (1 - test_size))
    return encoded_data[:split_idx], encoded_data[split_idx:]


def pretty_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K"
    else:
        return str(count)
