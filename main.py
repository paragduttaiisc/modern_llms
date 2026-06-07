import argparse
import torch, torch.optim as optim

from model import Model
from train_utils import train_step, evaluate
from utils import set_seed, load_data, tokenize_char, encode_text, train_test_split


def main(args):
    # Load data
    data = load_data(args.data_path)
    token_to_idx, idx_to_token = tokenize_char(data)
    encoded_data = encode_text(data, token_to_idx)
    train_data, val_data = train_test_split(encoded_data, test_size=args.test_size)

    # create model and optimizer
    model = Model(len(token_to_idx), args).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train the model
    model.train()
    for steps in range(args.n_iters + 1):
        if steps % args.eval_interval == 0:
            print(f"Evaluating at step: {steps} | ", end="")
            train_loss, val_loss = evaluate(model, train_data, val_data, args)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_step(model, optimizer, train_data, args)
    torch.save(model.state_dict(), f"{args.save_dir}/model.pth")

    # Generate text after training
    model.eval()
    with torch.no_grad():
        start_idx = encode_text(args.prompt, token_to_idx).unsqueeze(0).to(args.device)
        model.generate(start_idx, idx_to_token, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and sample from a character-level transformer")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-embed", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-iters", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=10_000)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
