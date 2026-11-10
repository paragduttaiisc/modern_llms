#!/bin/bash
# 100B-token run of the GPT-3 Large-shaped MoE (8 experts, top-2; 2.72B total /
# 0.85B active params) on 8 GPUs with ZeRO-2 (FSDP2).
#
# Batch 5 per GPU x 6 micro-batches = 491,520 tokens per step; 203,400 steps
# = ~100B tokens. Warmup 750 steps (~370M tokens) and cosine decay to 10% at
# step 176,500 (~87% of training), as in GPT-3. QK-norm is on (train.py's
# default). Measured on 8x A100-40GB without QK-norm: batch 5 ~125k tokens/s
# (~9.3 days), batch 6 ~130k. QK-norm adds ~1 GB of activations, which put
# batch 6 at the GPUs' full 40.4 GB even without eval, so batch 5 is used.
#
# Start it inside tmux so it survives SSH disconnects:
#   tmux new -s llm100b ./run_100b.sh       (detach: Ctrl+b d; reattach: tmux attach -t llm100b)
#
# If training crashes, the loop waits a minute and resumes from the latest
# checkpoint (resuming continues with exactly the next unseen data window).
# After a server reboot, run the same tmux command again to resume.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dl
cd "$(dirname "$0")"
# Less allocator fragmentation (more headroom for eval and saving).
export PYTORCH_ALLOC_CONF=expandable_segments:True
# The first attempt (MoE-8x2-100B, no QK-norm) diverged at ~step 1300:
# attention logits grew to ~2000 and every head attended to one token.
NAME=MoE-8x2-100B-QKNorm
SAVE=models/$NAME
mkdir -p "$SAVE" logs
for attempt in 1 2 3 4 5; do
  RESUME=""
  ls -d "$SAVE"/checkpoint-* > /dev/null 2>&1 && RESUME="--resume-training"
  echo "=== attempt $attempt $(date) $RESUME" >> "logs/$NAME.log"
  accelerate launch --num_processes=8 train.py --fsdp \
      --embed-size=1536 --n-layers=24 --n-heads=16 --attn-head-size=96 --rope-size=32 \
      --kv-latent-size=384 --mlp-hidden-size=4096 --n-experts=8 --n-active-experts=2 \
      --batch-size=5 --grad-accum-steps=6 --shard-list-file=data/llm_mix_100B.json \
      --n-iters=203400 --warmup-iters=750 --last-decay-iter=176500 \
      --eval-interval=2000 --save-interval=1000 --save-dir="$SAVE" \
      --wandb-run-name="$NAME" $RESUME 2>&1 | tee -a "logs/$NAME.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] && break      # finished normally
  echo "=== crashed (exit ${PIPESTATUS[0]}), resuming in 60 s $(date)" >> "logs/$NAME.log"
  sleep 60
done
