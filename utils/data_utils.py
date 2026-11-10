"""Datasets and tokenizer for pretraining and HellaSwag evaluation.

Pretraining data is pre-tokenized: each shard is a 1-D `.npy` array of token
ids, and a JSON manifest lists the shards of each split. `TokenDataset`
streams fixed-length training windows from those shards. `HellaswagDataset`
serves the tokenized HellaSwag benchmark for the zero-shot accuracy eval in
`LLMTrainer.evaluate_hellaswag`.
"""
import json
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(tokenizer_path: str = "tokenizer") -> PreTrainedTokenizer:
    """Load the project tokenizer, building and saving it on first use.

    The tokenizer is StarCoderBase's BPE tokenizer plus 64 extra special
    tokens (FIM, chat, tool-use and vision markers, padded out with reserved
    tokens), for a vocab of 49216. That size must match `--vocab-size`, and
    the shards must have been tokenized with the same tokenizer. If
    `tokenizer_path` cannot be loaded, the tokenizer is rebuilt from
    `bigcode/starcoderbase` (downloaded from the HF Hub) and saved there.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:  # noqa: BLE001
        print("Creating a new tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase")
        SPL_TOKS = [
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
            "<|filename|>", "<|gh_stars|>", "<|issue_start|>",
            "<|issue_comment|>", "<|issue_closed|>", "<|jupyter_start|>",
            "<|jupyter_text|>", "<|jupyter_code|>", "<|jupyter_output|>",
            "<|empty_output|>", "<|commit_before|>", "<|commit_msg|>",
            "<|commit_after|>", "<|reponame|>", "<|im_start|>", "<|im_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
            "<|video_pad|>", "<|tool_start|>", "<|tool_end|>",
            "<|tool_response_start|>", "<|tool_response_end|>",
            "<|action_start|>", "<|action_end|>"]
        SPL_TOKS.extend([
            f"<|reserved_token_{i}|>" for i in range(64 - len(SPL_TOKS))])
        tokenizer.add_special_tokens({"additional_special_tokens": SPL_TOKS})
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer


class TokenDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream fixed-length next-token-prediction examples from token shards.

    Each example is a window of `block_size + 1` consecutive tokens: the
    first `block_size` are the input and the last `block_size` (shifted by
    one) are the labels. Windows do not overlap (consecutive windows share
    one boundary token, the last label of one is the first input of the
    next), are read in shard order and are not shuffled. The leftover tail
    of each shard that does not fill a window is dropped. Every example
    yields `input_ids`, `attention_mask` (all ones) and `labels`, each a
    `[block_size]` int64 tensor.

    Args:
        shard_list_file: JSON manifest `{"train": [...], "val": [...]}` of
            `.npy` shard paths (relative to the repo root).
        block_size: sequence length of each example.
        subset: which manifest key to read.
        resume_iter: optimizer step being resumed from. Whole shards already
            consumed by `resume_iter` steps are skipped.
        batch_size: tokens consumed per optimizer step (over all ranks and
            gradient-accumulation steps), used only for resume skipping.
    """

    def __init__(
            self,
            shard_list_file: str,
            block_size: int,
            subset: str = "train",
            resume_iter: int = 0,
            batch_size: int = 2**19
    ) -> None:
        self.block_size = block_size
        with open(shard_list_file, 'r') as f:
            data = json.load(f)
        self.shard_paths = data[subset]
        if resume_iter > 0:
            # Every shard is assumed to hold 2e7 tokens. Skipping is at shard
            # granularity, so a partly consumed shard is replayed from its
            # start. (train.py sets `ignore_data_skip=True`, so HF does not
            # also skip batches on resume.)
            total_tokens_consumed = batch_size * resume_iter
            num_shards_consumed = int(total_tokens_consumed // 2e7)
            self.shard_paths = self.shard_paths[num_shards_consumed:]

    def __iter__(self):
        """Yield examples shard by shard, loading one shard at a time."""
        for shard_path in self.shard_paths:
            tokens = np.load(shard_path)
            n_examples = (len(tokens) - 1) // self.block_size
            for i in range(n_examples):
                start_idx = i * self.block_size
                end_idx = start_idx + self.block_size + 1
                seq = tokens[start_idx:end_idx]
                # .copy(): from_numpy shares memory, so without a copy each
                # tensor would be a view that keeps the whole shard alive.
                input_ids = torch.from_numpy(seq[:-1].copy()).long()
                attention_mask = torch.ones_like(input_ids)
                labels = torch.from_numpy(seq[1:].copy()).long()
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }


class HellaswagDataset(Dataset[dict[str, torch.Tensor]]):
    """HellaSwag examples, each a group of 4 tokenized candidate sequences.

    The `.npy` file holds `4 * N` rows of equal length `seq_len`: for every
    example, its 4 candidates (context followed by one ending), with the
    correct ending first, right-padded with token id 0. Each item is scored
    by language-model loss; see `LLMTrainer.evaluate_hellaswag`.
    """

    def __init__(self, data_path: str):
        """Load the rows and record each candidate's unpadded length."""
        sentences = np.load(data_path, allow_pickle=True)
        assert len(sentences) % 4 == 0
        self.sentences = sentences.reshape(-1, 4, sentences.shape[-1])
        # Length = index of the last non-zero token + 1 (0 is the padding id)
        self.lengths = np.array([
            [ np.where(candidate != 0)[0][-1] + 1 for candidate in example ]
            for example in self.sentences
        ])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """Return one example as `[4, seq_len - 1]` tensors.

        `input_ids` and `labels` are the candidates shifted by one token.
        The attention mask covers the real (non-padding) label positions;
        the model sets the labels of masked positions to -100, so padding
        does not count towards the loss.
        """
        sentences = self.sentences[idx]      # [4, seq_len]
        lengths = self.lengths[idx]          # [4]
        input_ids, labels, attention_masks = [], [], []
        for sentence, length in zip(sentences, lengths):
            pad_len = len(sentence) - length
            input_ids.append(sentence[:-1])
            labels.append(sentence[1:])
            attention_masks.append(
                np.concatenate([np.ones(length - 1), np.zeros(pad_len)])
            )
        return {
            "input_ids": torch.tensor(np.stack(input_ids), dtype=torch.long),
            "labels": torch.tensor(np.stack(labels), dtype=torch.long),
            "attention_mask": torch.tensor(np.stack(attention_masks), dtype=torch.long)
        }


def hellaswag_collate_fn(batch):
    """Collate HellaSwag items and flatten the candidate dimension.

    `[B, 4, L]` tensors become `[B * 4, L]`, so the 4 candidates of each
    example are consecutive rows that can be regrouped with
    `reshape(-1, 4)`.
    """
    batch = cast(dict[str, torch.Tensor], default_collate(batch))
    return {
        k: v.flatten(0, 1) for k, v in batch.items()
    }
