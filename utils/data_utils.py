import json
import torch
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data._utils.collate import default_collate

from torch.utils.data import Dataset, IterableDataset


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


class TokenDataset(IterableDataset):
    def __init__(
            self, shard_list_file: str, block_size: int, subset: str = "train"
    ) -> None:
        self.block_size = block_size
        data = json.load(open(shard_list_file, 'r'))
        self.shard_paths = data[subset]
    
    def __iter__(self):
        for shard_path in self.shard_paths:
            tokens = np.load(shard_path)
            n_examples = (len(tokens) - 1) // self.block_size
            for i in range(n_examples):
                start_idx = i * self.block_size
                end_idx = start_idx + self.block_size + 1
                seq = tokens[start_idx:end_idx]
                
                input_ids = torch.from_numpy(seq[:-1].copy()).long()
                attention_mask = torch.ones_like(input_ids)
                labels = torch.from_numpy(seq[1:].copy()).long()
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }


class HellaswagDataset(Dataset):
    def __init__(self, data_path: str):
        sentences = np.load(data_path, allow_pickle=True)

        assert len(sentences) % 4 == 0

        self.sentences = sentences.reshape(-1, 4, sentences.shape[-1])

        self.lengths = np.array([
            [
                np.where(candidate != 0)[0][-1] + 1
                for candidate in example
            ]
            for example in self.sentences
        ])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentences = self.sentences[idx]      # [4, seq_len]
        lengths = self.lengths[idx]          # [4]

        input_ids = []
        labels = []
        attention_masks = []

        for sentence, length in zip(sentences, lengths):
            pad_len = len(sentence) - length

            input_ids.append(sentence[:-1])
            labels.append(sentence[1:])

            attention_masks.append(
                np.concatenate([
                    np.ones(length - 1),
                    np.zeros(pad_len)
                ])
            )

        return {
            "input_ids": torch.tensor(
                np.stack(input_ids),
                dtype=torch.long
            ),
            "labels": torch.tensor(
                np.stack(labels),
                dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                np.stack(attention_masks),
                dtype=torch.long
            )
        }


def hellaswag_collate_fn(batch):
    batch = default_collate(batch)

    return {
        k: v.flatten(0, 1)
        for k, v in batch.items()
    }
