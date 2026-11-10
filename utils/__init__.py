"""Training-side utilities: data loading, the trainer, and small helpers.

- `data_utils`: the tokenizer, the sharded pretraining `TokenDataset`, and
  the HellaSwag evaluation dataset.
- `train_utils`: `LLMTrainer`, a HuggingFace `Trainer` subclass with the
  Muon/AdamW optimizer, the LR schedule, Matryoshka sampling, HellaSwag eval
  and extra logging.
- `misc_utils`: number formatting for printouts.
- `spec_decode` (not re-exported here): greedy and MTP self-speculative
  decoding, used by `infer.py`.
"""
from .data_utils import (
    HellaswagDataset,
    TokenDataset,
    get_tokenizer,
    hellaswag_collate_fn,
)
from .misc_utils import human_readable_numbers
from .train_utils import LLMTrainer

__all__ = [
    "HellaswagDataset",
    "LLMTrainer",
    "TokenDataset",
    "get_tokenizer",
    "hellaswag_collate_fn",
    "human_readable_numbers"
]
