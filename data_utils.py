import argparse
import torch
from typing import Tuple, List, Dict, Optional
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset

from utils import train_test_split


class Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]
    def __init__(self, text: str, **kwargs) -> None:
        chars = sorted(list(set(text)))
        self.token_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_token = {idx: char for char, idx in self.token_to_idx.items()}
        super().__init__(pad_token=None, eos_token=None, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def get_vocab(self) -> Dict[str, int]:
        return self.token_to_idx

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_idx.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self.idx_to_token.get(index, "")
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens) # TODO: remove this method when using subword tokenization

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        return ()

    def encode(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor([self.token_to_idx[char] for char in text], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> dict:
        seq = self.data[idx:idx + self.block_size + 1]
        return {"input_ids": seq[:-1].long(), "labels": seq[1:].long()}


def load_text_corpus(data_path: str) -> str:
    with open(data_path, 'r', encoding='utf-8') as f:
        return f.read()


def split_dataset(
        data: torch.Tensor, args: argparse.Namespace
) -> Tuple[TextDataset, TextDataset]:
    train_data, val_data = train_test_split(data, test_size=args.test_size)
    train_dataset = TextDataset(train_data, args.block_size)
    val_dataset = TextDataset(val_data, args.block_size)
    return train_dataset, val_dataset
