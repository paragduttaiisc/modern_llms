"""Pretrain a modern decoder-only language model from scratch.

This is the training entry point. It turns the command-line arguments into a
`ModelConfig`, builds a `Model` (MLA attention with RoPE, dense or MoE FFNs,
optional multi-token prediction and Matryoshka nesting), and hands it to
`LLMTrainer`, a HuggingFace `Trainer` subclass that adds the Muon/AdamW
optimizer, a warmup + cosine schedule, HellaSwag eval and extra logging.
Training data is streamed by `TokenDataset` from the pre-tokenized `.npy`
shards listed in `--shard-list-file`.

Launch with Accelerate, one process per GPU:

    # Dense model (one expert = a normal FFN)
    accelerate launch --num_processes=8 --mixed_precision=bf16 train.py \\
        --n-experts=1 --batch-size=16 --wandb-run-name="<RunName>"

    # Mixture of Experts (default: 8 experts, 2 active per token)
    accelerate launch --num_processes=8 --mixed_precision=bf16 train.py \\
        --batch-size=8 --wandb-run-name="<RunName>"

    # Larger models: ZeRO-2 via FSDP2 (no --mixed_precision, see main())
    accelerate launch --num_processes=8 train.py --fsdp ...

For a quick smoke test, add `--dont-log --dont-save --dont-eval --n-iters=20
--n-layers=2` and use fewer processes.
"""
import argparse
import math
import os

import numpy as np
import torch
from accelerate import PartialState
from transformers import TrainingArguments, set_seed

from model import Model, ModelConfig
from utils import (
    LLMTrainer,
    TokenDataset,
    get_tokenizer,
)
from utils import human_readable_numbers as hrn


def main(args: argparse.Namespace):
    """Build the data, model and trainer from `args` and run training.

    Steps: global setup (seeds, matmul precision, torch.compile limits),
    gradient accumulation from the token budget per optimizer step, the
    train/val `TokenDataset`s (skipping shards already seen when resuming),
    the `Model` (plus its draft vocab for multi-token prediction), and finally
    `LLMTrainer.train()`.
    """
    # Setup
    # Allow TF32 (or bf16-based) matmuls for any fp32 math; much faster on
    # Ampere+ GPUs at a small precision cost.
    torch.set_float32_matmul_precision('medium')
    set_seed(args.seed)
    matryoshka_fracs = [float(f) for f in args.matryoshka_fracs.split(",")]
    if len(matryoshka_fracs) > 1:
        # torch.compile builds one graph per (size, train/eval); raise the
        # recompile limit so it doesn't silently fall back to eager.
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, name):
                setattr(torch._dynamo.config, name, 64)
        # Keep the width a Python constant in each graph. Otherwise dynamo
        # turns it into a symbolic float after a recompile, which crashes
        # inside the checkpointed MTP heads.
        torch._dynamo.config.specialize_float = True
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    # Gradient accumulation: one optimizer step should see about
    # `--effective-tokens-target` tokens (2**19 = 0.5M by default), whatever
    # the number of GPUs and the per-GPU batch size. Each micro-batch step
    # covers batch_size * world_size * block_size tokens, so accumulate
    # ceil(target / that) micro-batches unless --grad-accum-steps is given.
    effective_tokens_target = args.effective_tokens_target
    effective_batch_size = args.batch_size
    world_size = 1
    if torch.cuda.is_available():
        world_size =\
            int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        effective_batch_size *= world_size
    effective_token_size = effective_batch_size * args.block_size
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps != -1\
        else math.ceil(effective_tokens_target / effective_token_size)
    if args.fsdp:
        # Pure bf16: Accelerate's FSDP2 upcasts the weights to fp32 masters
        # whenever mixed precision is on, so turn it off. The model is cast
        # to bf16 below; optimizer moments are fp32 (see MuonFP32State).
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    # Only the process/rank info is needed here (to print on rank 0 only).
    # PartialState sets up the distributed process group without fixing
    # mixed-precision or FSDP settings; the HF Trainer later creates its own
    # Accelerator from the TrainingArguments below, and an Accelerator built
    # here first would conflict with it.
    accelerator = PartialState()
    if accelerator.is_main_process:
        print("Targeted Effective tokens:", hrn(args.effective_tokens_target))
        print("Num of GPUs:", world_size)
        print("Batch size per GPU:", args.batch_size)
        print("Sequence length:", args.block_size)
        print("Total effective token size:", hrn(effective_token_size))
        print("Gradient accumulation steps:", grad_accum_steps)
        if args.fsdp:
            print("FSDP2 ZeRO-2 (shard_grad_op), pure bf16 weights and grads")

    # Prepare data
    tokenizer = get_tokenizer()
    ckpt_path, resume_iter = None, 0
    if args.resume_training:
        # Resume from --resume-iter, or else the latest checkpoint-<step>
        # directory in --save-dir.
        resume_iter = args.resume_iter if args.resume_iter > 0 else max([
        int(d.split("-")[-1]) for d in os.listdir(args.save_dir)
        if d.startswith("checkpoint-")])
        ckpt_path = os.path.join(args.save_dir, f"checkpoint-{resume_iter}")
    # `batch_size` here is the number of tokens consumed per optimizer step
    # (all GPUs, all micro-batches); with `resume_iter` it is used to skip
    # the shards already seen, since the HF data skipping is turned off
    # (`ignore_data_skip` below).
    trainset = TokenDataset(
        args.shard_list_file, block_size=args.block_size, subset="train",
        resume_iter=resume_iter, batch_size=effective_token_size *
        grad_accum_steps)
    valset = TokenDataset(
        args.shard_list_file, block_size=args.block_size, subset="val")
    if accelerator.is_main_process:
        print("Tokenizer vocab size:", hrn(tokenizer.vocab_size))
        print("Num tokens in trainset:", hrn(int(len(trainset.shard_paths) * 2e7)))
        # Every shard is assumed to hold 2e7 tokens
        print("Num tokens in valset:", hrn(int(len(valset.shard_paths) * 2e7)))

    # create model and optimizer
    # CLI names differ from config names (e.g. --embed-size -> embedding_size,
    # --n-experts -> experts); a new hyperparameter needs an argparse entry,
    # a keyword here and a field in model/config.py.
    model = Model(ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        embedding_size=args.embed_size,
        sa_head_size=args.attn_head_size,
        ff_hidden_size=args.mlp_hidden_size,
        rope_size=args.rope_size,
        kv_latent_size=args.kv_latent_size,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_residual_streams=args.n_res_streams,
        experts=args.n_experts,
        active_experts=args.n_active_experts,
        router_loss_coef=args.router_loss_weight,
        non_linearity=args.non_linearity,
        dropout=args.dropout,
        n_future_tokens=args.n_future_tokens,
        mtp_loss_weight=args.mtp_loss_weight,
        mtp_checkpoint=not args.no_mtp_checkpoint,
        block_checkpoint=args.activation_checkpointing,
        draft_vocab_size=args.draft_vocab_size if args.n_future_tokens > 1 else 0,
        mtp_position_frac=args.mtp_position_frac,
        matryoshka_fracs=matryoshka_fracs,
    ))
    if model.config.draft_vocab_size > 0:
        # Draft vocab of the extra MTP heads = most frequent training tokens.
        # Heads 2..n then unembed only these rows of lm_head and ignore other
        # targets, which cuts the cost of their vocab projections; head 1
        # (the normal next-token head) always uses the full vocab. The ids
        # are stored in model buffers, so checkpoints are self-contained.
        counts = np.load(args.draft_vocab_file)
        if len(counts) != args.vocab_size:
            raise ValueError(f"{args.draft_vocab_file} has {len(counts)} "
                             f"counts, expected vocab size {args.vocab_size}")
        top = np.argsort(-counts, kind="stable")[:args.draft_vocab_size]
        model.set_draft_vocab(torch.from_numpy(top))
        if accelerator.is_main_process:
            print(f"Draft vocab: {hrn(args.draft_vocab_size)} tokens covering "
                  f"{counts[top].sum() / counts.sum():.2%} of counted tokens")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Initialize weights explicitly (Xavier-normal Linears, 1/sqrt(d)
    # embeddings, unit RMSNorm scales), then cast to bf16: weights are kept
    # in bf16 on GPU (the MoE grouped_mm path requires it).
    model.apply(model.weight_init)
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)  # pyright: ignore[reportCallIssue]

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters in model:", hrn(num_params))
        print("Model parameter dtype:", next(model.parameters()).dtype)

    # # train
    # Standard HF Trainer settings. The optimizer and LR schedule are not
    # HF's: LLMTrainer builds Muon + AdamW groups and its own warmup/cosine
    # schedule (learning_rate/weight_decay below are the AdamW values).
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        torch_compile=True,
        use_cache=False,
        max_steps=args.n_iters,
        ignore_data_skip=True, # TODO: Adapt dataloader accordingly
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_on_start=not args.dont_eval,
        eval_strategy="no" if args.dont_eval else "steps",
        eval_steps=None if args.dont_eval else args.eval_interval,
        gradient_accumulation_steps=grad_accum_steps,
        save_total_limit=0 if args.dont_save else 5,
        save_strategy="no" if args.dont_save else "steps",
        save_steps=-1 if args.dont_save else args.save_interval,
        learning_rate=args.adamw_lr,
        weight_decay=args.adamw_weight_decay,
        # bf16 autocast, except with FSDP, where weights are already bf16
        bf16=not args.fsdp,
        # ZeRO-2: gradients and optimizer state sharded, weights gathered
        # once per step and kept through backward (FSDP2 shard_grad_op).
        fsdp="shard_grad_op auto_wrap" if args.fsdp else "",
        fsdp_config={"version": 2, "transformer_layer_cls_to_wrap": ["Block"]}
            if args.fsdp else None,
        logging_steps=args.log_interval,
        run_name=args.wandb_run_name,
        project=args.wandb_project,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=1,
        ddp_find_unused_parameters=False,
        # wandb if a run name is given, else TensorBoard (or nothing)
        report_to="wandb" if args.wandb_run_name else\
            "none" if args.dont_log else "tensorboard",
    )
    trainer = LLMTrainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        block_size=args.block_size,
        warmup_iters=args.warmup_iters,
        last_decay_iter=args.last_decay_iter,
        muon_lr=args.muon_lr,
        muon_weight_decay=args.muon_weight_decay,
        matryoshka_full_prob=args.matryoshka_full_prob,
        matryoshka_full_per_step=args.matryoshka_full_per_step,
        matryoshka_all_sizes=args.matryoshka_all_sizes,
    )
    if accelerator.is_main_process:
        print("Starting training...")
    trainer.train(resume_from_checkpoint=ckpt_path)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer")
    # Data and reproducibility
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shard-list-file", type=str, default="data/web_small_10B.json")
    parser.add_argument("--save-dir", type=str, default="outputs")
    # Model shape. vocab_size must match the tokenizer (49216).
    parser.add_argument("--vocab-size", type=int, default=49216)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=12)
    # Hyper-connection residual streams (1 = plain residual add)
    parser.add_argument("--n-res-streams", type=int, default=1)
    # Attention heads (MLA), RoPE dims per head, model width, FFN width
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--attn-head-size", type=int, default=64)
    parser.add_argument("--rope-size", type=int, default=16)
    parser.add_argument("--embed-size", type=int, default=768)
    parser.add_argument("--mlp-hidden-size", type=int, default=3072)
    # Mixture of Experts (--n-experts=1 = dense FFN)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--n-active-experts", type=int, default=2)
    # Width of the MLA latent (what the KV cache stores); FFN activation
    parser.add_argument("--kv-latent-size", type=int, default=128)
    parser.add_argument("--non-linearity", type=str, default="SwiGLU", choices=["GELU", "SwiGLU", "SqReLU"])
    # Optimization and regularization. --grad-accum-steps=-1 derives it from
    # --effective-tokens-target.
    # Muon updates the hidden 2-D/3-D weights, AdamW everything else.
    parser.add_argument("--grad-accum-steps", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--muon-weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Not read anywhere in this script: the AdamW groups take their betas
    # from the TrainingArguments defaults (adam_beta1/adam_beta2)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    # Weight of the MoE load-balancing loss
    parser.add_argument("--router-loss-weight", type=float, default=0.01)
    # Multi-token prediction
    parser.add_argument("--n-future-tokens", type=int, default=1, help="Multi-token prediction heads (1 = standard next-token LM)")
    parser.add_argument("--mtp-loss-weight", type=float, default=1.0, help="Weight of the mean loss of MTP heads 2..n")
    parser.add_argument("--no-mtp-checkpoint", action="store_true", help="Don't checkpoint MTP heads (faster, but keeps every head's logits in memory)")
    parser.add_argument("--draft-vocab-size", type=int, default=0, help="Extra MTP heads predict only the N most frequent tokens (0 = full vocab)")
    parser.add_argument("--draft-vocab-file", type=str, default="data/token_counts_web_small_10B.npy", help="Token counts from scripts/token_frequencies.py")
    parser.add_argument("--mtp-position-frac", type=float, default=1.0, help="Fraction of positions used for the extra MTP heads' loss")
    # Matryoshka (nested widths sharing one set of weights)
    parser.add_argument("--matryoshka-fracs", type=str, default="1", help="Comma-separated nested widths trained jointly, e.g. 1,0.5,0.25 (1 = no nesting)")
    parser.add_argument("--matryoshka-full-prob", type=float, default=0.5, help="Probability a micro-batch trains the full model (else a smaller size, uniformly)")
    parser.add_argument("--matryoshka-full-per-step", action="store_true", help="Make the first micro-batch of every optimizer step full size (recommended: prevents divergence)")
    parser.add_argument("--matryoshka-all-sizes", action="store_true", help="Train every Matryoshka size on every micro-batch (loss = sum over sizes) instead of sampling one")
    # Scale: memory savings for larger models
    parser.add_argument("--fsdp", action="store_true", help="ZeRO-2 via FSDP2: shard gradients and optimizer state across GPUs (pure bf16 weights)")
    parser.add_argument("--activation-checkpointing", action="store_true", help="Recompute every transformer block in backward to save activation memory")
    # Not read anywhere in this script
    parser.add_argument("--use-fused-optimizer", type=bool, default=True)
    # Batch size per GPU, and tokens per optimizer step (all GPUs)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--effective-tokens-target", type=int, default=2**19)
    # Schedule: optimizer steps, linear warmup, cosine decay to 0.1x LR at
    # --last-decay-iter, then constant
    parser.add_argument("--n-iters", type=int, default=19050)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--last-decay-iter", type=int, default=17150)
    # Logging / checkpoints / resuming
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--resume-iter", type=int, default=0)
    parser.add_argument("--dont-log", action="store_true")
    parser.add_argument("--dont-save", action="store_true")
    parser.add_argument("--dont-eval", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="improved-transformer")
    parser.add_argument("--wandb-entity", type=str, default="statsml-csa-iisc")
    args = parser.parse_args()
    main(args)
