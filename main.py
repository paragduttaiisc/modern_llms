import torch
import torch.optim as optim
from model import Model
from utils import Config, set_seed, load_data, pretty_count, tokenize_char, encode_text, decode_text, train_test_split
from train_utils import train_step, evaluate


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data = load_data(args.data_path)
    print(f"Data length: {pretty_count(len(data))} characters")
    
    # convert data to model input format
    token_to_idx, idx_to_token = tokenize_char(data)
    encoded_data = encode_text(data, token_to_idx)
    print(f"Unique characters: {pretty_count(len(token_to_idx))}")
    print(f"First 15 encoded characters: {encoded_data[:15]}")
    print(f"First 15 decoded characters: {decode_text(encoded_data[:15], idx_to_token)}")

    # train-val split
    train_data, val_data = train_test_split(encoded_data, test_size=args.test_size)
    print(f"Train data length: {pretty_count(len(train_data))}")
    print(f"Validation data length: {pretty_count(len(val_data))}")

    # create model
    model = Model(vocab_size=len(token_to_idx))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Generating text...")
    start_idx = torch.tensor([[0]], dtype=torch.long).to(device)
    generated_indices = model.generate(start_idx, max_new_tokens=100)
    generated_text = decode_text(generated_indices[0].cpu(), idx_to_token)
    print(f"Generated text:\n{generated_text}")

    model.train()
    for steps in range(args.n_iters):
        loss = train_step(
            model, optimizer, train_data, args.batch_size, args.block_size, device)

        if (steps + 1) % args.eval_interval == 0:
            avg_train_loss, avg_val_loss = evaluate(
                model, train_data, val_data, args.batch_size, args.block_size, args.eval_iters, device)
            print(f"Step {steps + 1}/{args.n_iters} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}")
    
    # Generate text after training
    model.eval()
    with torch.no_grad():
        start_idx = torch.tensor([[0]], dtype=torch.long).to(device)
        generated_indices = model.generate(start_idx, max_new_tokens=500)
        generated_text = decode_text(generated_indices[0].cpu(), idx_to_token)
        print(f"Generated text after training:\n{generated_text}")


if __name__ == "__main__":
    args = Config(
        seed=2026,
        data_path='data/tinyshakespeare.txt',
        test_size=0.1,
        block_size=8,
        batch_size=32,
        n_iters=5000,
        learning_rate=3e-3,
        eval_interval=500,
        eval_iters=200,
    )
    main(args)
