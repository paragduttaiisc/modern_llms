import os
import math
import torch
import argparse
from accelerate import Accelerator
from transformers import set_seed, TrainingArguments

from model import Model, ModelConfig
from utils import (
    TokenDataset,
    LLMTrainer,
    get_tokenizer,
    human_readable_numbers as hrn,
)


def main(args: argparse.Namespace):
    # Setup
    torch.set_float32_matmul_precision('medium' if args.use_bf16 else 'high')
    set_seed(args.seed)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    effective_tokens_target = args.effective_tokens_target
    effective_batch_size = args.batch_size
    if torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        effective_batch_size *= world_size
    effective_token_size = effective_batch_size * args.block_size
    accelerator = Accelerator(
        mixed_precision="bf16" if args.use_bf16 and torch.cuda.is_available()\
            else "no", gradient_accumulation_steps=args.grad_accum_steps\
                if args.grad_accum_steps != -1 else\
                    math.ceil(effective_tokens_target / effective_token_size),
    )
    if accelerator.is_main_process:
        print("Targeted Effective tokens:", hrn(args.effective_tokens_target))
        print("Num of GPUs:", world_size)
        print("Batch size per GPU:", args.batch_size)
        print("Sequence length:", args.block_size)
        print("Total effective token size:", hrn(effective_token_size))
        print("Gradient accumulation steps:",
              accelerator.gradient_accumulation_steps)
    
    # Prepare data
    tokenizer = get_tokenizer()
    trainset = TokenDataset(
        args.shard_list_file,
        block_size=args.block_size,
        subset="train"
    )
    valset = TokenDataset(
        args.shard_list_file,
        block_size=args.block_size,
        subset="val"
    )
    if accelerator.is_main_process:
        print("Tokenizer vocab size:", hrn(tokenizer.vocab_size))
        print("Num tokens in trainset:", hrn(int(len(trainset.shard_paths) * 2e7)))
        print("Num tokens in valset:", hrn(int(len(valset.shard_paths) * 2e7)))

    # create model and optimizer
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
        experts=args.n_experts,
        active_experts=args.n_active_experts,
        router_loss_coeff=args.router_loss_weight,
        non_linearity=args.non_linearity,
        dropout=args.dropout,
    ))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters in model:", hrn(num_params))

    # # train
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        torch_compile=True,
        use_cache=False,
        max_steps=args.n_iters,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_on_start=not args.dont_eval,
        eval_strategy="no" if args.dont_eval else "steps",
        eval_steps=None if args.dont_eval else args.eval_interval,
        gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
        save_total_limit=0 if args.dont_save else 5,
        save_strategy="no" if args.dont_save else "steps",
        save_steps=-1 if args.dont_save else args.save_interval,
        learning_rate=args.adamw_lr,
        weight_decay=args.adamw_weight_decay,
        bf16=args.use_bf16 and torch.cuda.is_available(),
        logging_steps=args.log_interval,
        run_name=args.wandb_run_name,
        project=args.wandb_project,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=args.n_experts > 1,
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
        muon_weight_decay=args.muon_weight_decay
    )
    model.apply(model.weight_init)
    if accelerator.is_main_process:
        print("Starting training...")
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shard-list-file", type=str, default="data/web_small_10B.json")
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=49216)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--attn-head-size", type=int, default=64)
    parser.add_argument("--rope-size", type=int, default=16)
    parser.add_argument("--embed-size", type=int, default=768)
    parser.add_argument("--mlp-hidden-size", type=int, default=3072)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--n-active-experts", type=int, default=2)
    parser.add_argument("--kv-latent-size", type=int, default=128)
    parser.add_argument("--non-linearity", type=str, default="SqReLU", choices=["GELU", "SwiGLU", "SqReLU"])
    parser.add_argument("--grad-accum-steps", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--muon-weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    parser.add_argument("--router-loss-weight", type=float, default=0.01)
    parser.add_argument("--use-fused-optimizer", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effective-tokens-target", type=int, default=2**19)
    parser.add_argument("--n-iters", type=int, default=75250)
    parser.add_argument("--warmup-iters", type=int, default=750)
    parser.add_argument("--last-decay-iter", type=int, default=72000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--dont-log", action="store_true")
    parser.add_argument("--dont-save", action="store_true")
    parser.add_argument("--dont-eval", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="improved-transformer")
    parser.add_argument("--wandb-entity", type=str, default="statsml-csa-iisc")
    args = parser.parse_args()
    main(args)
