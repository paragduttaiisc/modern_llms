import os, argparse, wandb, time
import torch, torch.optim as optim
from accelerate import Accelerator

from model import Model, ModelConfig
from data_utils import get_dataloader_and_tokenizer
from train_utils import train
from utils import set_seed
torch.set_float32_matmul_precision('high')


def main(args: argparse.Namespace):
    if args.train: # Set up accelerator for training
        accelerator = Accelerator()
        args.device = accelerator.device
    
    # Prepare data
    train_loader, val_loader, tokenizer = get_dataloader_and_tokenizer(args)

    # create model
    model = Model(ModelConfig(
        vocab_size=len(tokenizer.token_to_idx),
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    )).to(args.device)
    if hasattr(torch, 'compile'):
        model: torch.nn.Module = torch.compile(model) # type: ignore

    if args.train: # Train the model
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)
        wandb_run = None
        if accelerator.is_main_process:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args.save_dir = os.path.join(args.save_dir, timestamp)
            os.makedirs(args.save_dir, exist_ok=True)
            if args.wandb_run_name is not None:
                wandb.init(
                    name=f"{args.wandb_run_name}_{timestamp}",
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=vars(args)
                )
                wandb_run = wandb.run
        train(
            model, optimizer, train_loader, val_loader,
            accelerator, args, wandb_run
        )
    else: # Generate text
        # Load the model from the specified directory
        model = Model.from_pretrained(args.save_dir).to(args.device)
        model.eval()
        with torch.no_grad():
            start_idx = tokenizer.encode(args.prompt).unsqueeze(0).to(args.device)
            model.generate_stream(start_idx, tokenizer, max_new_tokens=args.max_new_tokens) # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and sample from a character-level transformer")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-embed", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=400)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=10_000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="improved-transformer")
    parser.add_argument("--wandb-entity", type=str, default="statsml-csa-iisc")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
