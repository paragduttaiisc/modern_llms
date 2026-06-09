import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from utils import train_test_split


class Tokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.token_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_token = {idx: char for char, idx in self.token_to_idx.items()}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.token_to_idx[char] for char in text], dtype=torch.long)

    def decode(self, encoded_text: torch.Tensor) -> str:
        return ''.join([self.idx_to_token[idx.item()] for idx in encoded_text]) # type: ignore


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data[idx:idx + self.block_size + 1]
        return sequence[:-1].long(), sequence[1:].long()


def get_dataloader_and_tokenizer(
        args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, Tokenizer]:
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = Tokenizer(text)
    data = tokenizer.encode(text)
    train_data, val_data = train_test_split(data)
    train_dataset = TextDataset(train_data, args.block_size)
    val_dataset = TextDataset(val_data, args.block_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    return train_dataloader, val_dataloader, tokenizer
