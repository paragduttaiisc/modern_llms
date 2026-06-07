import torch
import random
from typing import Tuple


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def tokenize_char(text: str) -> Tuple[dict, dict]:
    chars = sorted(list(set(text)))
    token_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_token = {idx: char for char, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


def encode_text(text: str, token_to_idx: dict) -> torch.Tensor:
    return torch.tensor([token_to_idx[char] for char in text], dtype=torch.long)


def decode_text(encoded_text: torch.Tensor, idx_to_token: dict) -> str:
    return ''.join([idx_to_token[idx.item()] for idx in encoded_text])


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


def get_batch(
        data: torch.Tensor, batch_size: int, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]: # type: ignore
    len_dataset = len(data) - block_size - 1
    idxs = random.sample(range(len_dataset), batch_size)
    sequences = torch.stack([data[idx:idx + block_size + 1] for idx in idxs])
    return sequences[:, :-1], sequences[:, 1:]