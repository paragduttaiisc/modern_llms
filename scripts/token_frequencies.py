"""Count token frequencies over (a sample of) the training shards.

The result is a [vocab_size] int64 array of counts, used by train.py
(`--draft-vocab-file`) to pick the most frequent tokens as the reduced
vocabulary of the extra MTP heads.

    python scripts/token_frequencies.py --shard-list-file=data/web_small_10B.json \
        --out=data/token_counts_web_small_10B.npy
"""
import argparse
import json

import numpy as np
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    with open(args.shard_list_file) as f:
        shards = json.load(f)[args.subset]
    if 0 < args.max_shards < len(shards):
        # Evenly spaced sample so every source in the manifest is covered
        idx = np.linspace(0, len(shards) - 1, args.max_shards).round()
        shards = [shards[int(i)] for i in np.unique(idx)]

    counts = np.zeros(args.vocab_size, dtype=np.int64)
    for path in tqdm(shards, desc="Counting"):
        tokens = np.load(path, mmap_mode="r")
        counts += np.bincount(tokens, minlength=args.vocab_size)[:args.vocab_size]

    np.save(args.out, counts)
    total = counts.sum()
    order = np.sort(counts)[::-1]
    print(f"{len(shards)} shards, {total / 1e9:.2f}B tokens -> {args.out}")
    for k in (4096, 8192, 16384, 32768):
        print(f"  top-{k:5d} tokens cover {order[:k].sum() / total:.2%}")
    print(f"  tokens never seen: {(counts == 0).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shard-list-file", type=str, default="data/web_small_10B.json")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--max-shards", type=int, default=50, help="0 = all shards")
    parser.add_argument("--vocab-size", type=int, default=49216)
    parser.add_argument("--out", type=str, default="data/token_counts_web_small_10B.npy")
    main(parser.parse_args())
