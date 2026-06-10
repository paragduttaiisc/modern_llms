import argparse
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import Tuple

from utils import train_test_split


def get_tokenizer(tokenizer_path: str = "tokenizer") -> PreTrainedTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print("Creating a new tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase")
        SPL_TOKS = [
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
            "<|filename|>", "<|gh_stars|>", "<|issue_start|>",
            "<|issue_comment|>", "<|issue_closed|>", "<|jupyter_start|>",
            "<|jupyter_text|>", "<|jupyter_code|>", "<|jupyter_output|>",
            "<|empty_output|>", "<|commit_before|>", "<|commit_msg|>",
            "<|commit_after|>", "<|reponame|>", "<|im_start|>", "<|im_end|>"
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
            "<|video_pad|>", "<|tool_start|>", "<|tool_end|>",
            "<|tool_response_start|>", "<|tool_response_end|>",
            "<|action_start|>", "<|action_end|>"]
        SPL_TOKS.extend([
            f"<|reserved_token_{i}|>" for i in range(64 - len(SPL_TOKS))])
        tokenizer.add_special_tokens({"additional_special_tokens": SPL_TOKS})
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer



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
