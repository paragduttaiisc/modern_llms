import os, math, argparse
import torch, torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import set_seed, TrainingArguments, Trainer

from model import Model, ModelConfig
from data_utils import get_tokenizer, TokenDataset


def main(args: argparse.Namespace):
    # Prepare data
    tokenizer = get_tokenizer()
    dataset = TokenDataset(args.shard_list_file, block_size=args.block_size)

    # create model and optimizer
    model = Model(ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        hidden_size=args.n_embed,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        dropout=args.dropout
    ))
    if os.path.exists("models/checkpoint-35000"):
        print("Loading model from checkpoint...")
        model = Model.from_pretrained("models/checkpoint-35000")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-6, total_iters=args.warmup_iters)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.last_decay_iter - args.warmup_iters,
        eta_min=args.learning_rate / 10)
    constant_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=0.1,
        total_iters=args.n_iters - args.last_decay_iter)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
        milestones=[args.warmup_iters, args.last_decay_iter])

    # train
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        torch_compile=True,
        max_steps=args.n_iters,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_on_start=True,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=2,
        save_steps=args.save_interval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.use_bf16 and torch.cuda.is_available(),
        logging_steps=args.log_interval,
        report_to="wandb" if args.wandb_run_name else "none",
        run_name=args.wandb_run_name,
        project=args.wandb_project,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(optimizer, scheduler) # type: ignore
    )
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shard-list-file", type=str, default="data/web_small.txt")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=49216)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-embed", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-tokens-target", type=int, default=2**19)
    parser.add_argument("--n-iters", type=int, default=38145)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--last-decay-iter", type=int, default=35000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="improved-transformer")
    parser.add_argument("--wandb-entity", type=str, default="statsml-csa-iisc")
    args = parser.parse_args()
    torch.set_float32_matmul_precision('medium' if args.use_bf16 else 'high')
    set_seed(args.seed)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    effective_tokens_target = args.effective_tokens_target
    effective_batch_size = args.batch_size
    if torch.cuda.is_available():
        effective_batch_size *= torch.cuda.device_count()
    effective_tokens_size = effective_batch_size * args.block_size
    args.gradient_accumulation_steps =\
        max(1, math.ceil(effective_tokens_target / effective_tokens_size))
    main(args)
