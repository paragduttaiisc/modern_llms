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
from torch.utils.data import Dataset, IterableDataset, get_worker_info
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
    next). The leftover tail of each shard that does not fill a window is
    dropped. Every example yields `input_ids`, `attention_mask` (all ones)
    and `labels`, each a `[block_size]` int64 tensor.

    Order: with `interleave_shards = 1`, windows are read shard by shard in
    order. Each shard comes from a single source (web, code, math...) and
    holds 2e7 tokens, about 38 optimizer steps of 2^19 tokens, so every
    batch would come from one source for dozens of steps at a time even if
    the shard list itself is shuffled. With `interleave_shards = K > 1`, the
    shards are taken K at a time and all windows of a block are yielded in a
    random order, so each batch mixes the sources in roughly the corpus
    proportions. The order is seeded by `seed` and the block's position in
    the full shard list, so it is reproducible and survives resuming.

    Args:
        shard_list_file: JSON manifest `{"train": [...], "val": [...]}` of
            `.npy` shard paths (relative to the repo root).
        block_size: sequence length of each example.
        subset: which manifest key to read.
        resume_iter: optimizer step being resumed from. Exactly the
            windows consumed by `resume_iter` steps are skipped, so a resumed
            run continues with the next unseen window (shard lengths are read
            from the `.npy` headers).
        batch_size: tokens consumed per optimizer step (over all ranks and
            gradient-accumulation steps), used only for resume skipping.
        interleave_shards: number of shards whose windows are mixed (1 =
            read in order).
        seed: seed of the window order when interleaving.
    """

    def __init__(
            self,
            shard_list_file: str,
            block_size: int,
            subset: str = "train",
            resume_iter: int = 0,
            batch_size: int = 2**19,
            interleave_shards: int = 1,
            seed: int = 0,
    ) -> None:
        self.block_size = block_size
        self.interleave_shards = max(1, interleave_shards)
        self.seed = seed
        with open(shard_list_file, 'r') as f:
            data = json.load(f)
        self.shard_paths = data[subset]
        # Index (in blocks of interleave_shards) of the first block left after
        # resume skipping, which keeps each block's shuffle seed the same as
        # in the original run, and the number of windows of that block that
        # were already consumed.
        self.first_block = 0
        self.skip_windows = 0
        if resume_iter > 0:
            # Skip exactly the windows already trained on: whole blocks, then
            # the consumed part of the current block in its (seeded) order.
            # (train.py sets `ignore_data_skip=True`, so HF does not also
            # skip batches on resume.)
            remaining = resume_iter * batch_size // block_size
            K = self.interleave_shards
            for b in range(0, len(self.shard_paths), K):
                n = sum(self._n_windows(path)
                        for path in self.shard_paths[b:b + K])
                if remaining < n:
                    break
                remaining -= n
                self.first_block += 1
            self.skip_windows = remaining
            self.shard_paths = self.shard_paths[self.first_block * K:]

    def _n_windows(self, path: str) -> int:
        """Number of windows in a shard (reads only the .npy header)."""
        return (np.load(path, mmap_mode="r").shape[0] - 1) // self.block_size

    def _example(self, seq: np.ndarray) -> dict[str, torch.Tensor]:
        """One `block_size + 1`-token window -> input_ids / mask / labels."""
        # .copy(): from_numpy shares memory, so without a copy each tensor
        # would be a view that keeps the whole shard (or its mmap) alive.
        input_ids = torch.from_numpy(seq[:-1].copy()).long()
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.from_numpy(seq[1:].copy()).long(),
        }

    def __iter__(self):
        """Yield examples shard by shard, or block by block (shuffled)."""
        info = get_worker_info()
        if info is not None and info.num_workers > 1:
            # Every worker would iterate the full stream, so each window
            # would be yielded num_workers times.
            raise RuntimeError("TokenDataset is a single stream; use at most "
                               "one DataLoader worker")
        bs = self.block_size
        skip = self.skip_windows
        if self.interleave_shards == 1:
            for shard_path in self.shard_paths:
                tokens = np.load(shard_path)
                for i in range(skip, (len(tokens) - 1) // bs):
                    yield self._example(tokens[i * bs:i * bs + bs + 1])
                skip = 0
            return
        K = self.interleave_shards
        for b in range(0, len(self.shard_paths), K):
            # Memory-mapped: only the windows actually read are paged in, so
            # a block of K shards costs no more RAM than one loaded shard.
            shards = [np.load(path, mmap_mode="r")
                      for path in self.shard_paths[b:b + K]]
            counts = [(len(tokens) - 1) // bs for tokens in shards]
            shard_ids = np.repeat(np.arange(len(shards)), counts)
            window_ids = np.concatenate([np.arange(c) for c in counts])
            rng = np.random.default_rng([self.seed, self.first_block + b // K])
            order = rng.permutation(len(shard_ids))[skip:]
            skip = 0
            for j in order:
                start = int(window_ids[j]) * bs
                yield self._example(
                    np.asarray(shards[shard_ids[j]][start:start + bs + 1]))


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
