#!/bin/bash
# 100B-token run of the GPT-3 Large-shaped MoE (8 experts, top-2; 2.72B total /
# 0.85B active params) on 8 GPUs with ZeRO-2 (FSDP2).
#
# Start it inside tmux so it survives SSH disconnects:
#   tmux new -s llm100b ./run_100b.sh       (detach: Ctrl+b d; reattach: tmux attach -t llm100b)
#
# If training crashes, the loop waits a minute and resumes from the latest
# checkpoint (resuming continues with exactly the next unseen data window).
# After a server reboot, run the same tmux command again to resume.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dl
cd "$(dirname "$0")"
NAME=MoE-8x2-100B
SAVE=models/$NAME
mkdir -p "$SAVE" logs
for attempt in 1 2 3 4 5; do
  RESUME=""
  ls -d "$SAVE"/checkpoint-* > /dev/null 2>&1 && RESUME="--resume-training"
  echo "=== attempt $attempt $(date) $RESUME" >> "logs/$NAME.log"
  accelerate launch --num_processes=8 train.py --fsdp \
      --embed-size=1536 --n-layers=24 --n-heads=16 --attn-head-size=96 --rope-size=32 \
      --kv-latent-size=384 --mlp-hidden-size=4096 --n-experts=8 --n-active-experts=2 \
      --batch-size=4 --shard-list-file=data/llm_mix_100B.json \
      --n-iters=190735 --warmup-iters=715 --last-decay-iter=165300 \
      --eval-interval=2000 --save-interval=2000 --save-dir="$SAVE" \
      --wandb-run-name="$NAME" $RESUME 2>&1 | tee -a "logs/$NAME.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] && break      # finished normally
  echo "=== crashed (exit ${PIPESTATUS[0]}), resuming in 60 s $(date)" >> "logs/$NAME.log"
  sleep 60
done
