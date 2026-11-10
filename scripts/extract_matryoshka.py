"""Extract a standalone smaller model from a Matryoshka checkpoint.

    python scripts/extract_matryoshka.py --save-dir=outputs/<run>/checkpoint-XXXX \
        --frac=0.25 --out=outputs/<run>/checkpoint-XXXX-f0.25

The result is a normal checkpoint (load with Model.from_pretrained, run with
infer.py) whose outputs equal the parent run at width `frac`. Size f keeps
the first f of the residual width, attention heads, MLA latent and FFN width
(prefix slices of every weight); head and RoPE dims are unchanged.
"""
import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from utils.misc_utils import human_readable_numbers as hrn


def main(args: argparse.Namespace) -> None:
    """Load `--save-dir`, slice it to width `--frac` and save it to `--out`."""
    model = Model.from_pretrained(args.save_dir)
    small = model.extract_matryoshka(args.frac)
    small.save_pretrained(args.out)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    c = small.config
    print(f"Saved {args.out}: {hrn(count(small))} params "
          f"(parent {hrn(count(model))}); d_model={c.embedding_size}, "
          f"heads={c.num_attention_heads}, kv_latent={c.kv_latent_size}, "
          f"ffn_hidden={c.ff_hidden_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--out", type=str, required=True)
    main(parser.parse_args())
