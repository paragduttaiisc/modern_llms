import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline, set_seed

from model import Model, ModelConfig
from data_utils import load_text_corpus, Tokenizer


def main(args: argparse.Namespace):
    # Register and load model
    AutoConfig.register("modern_transformer", ModelConfig)
    AutoModelForCausalLM.register(ModelConfig, Model)
    model = Model.from_pretrained(args.save_dir).to(args.device)
    model.eval()

    # load tokenizer
    text_cropus = load_text_corpus(args.data_path)
    tokenizer = Tokenizer(text_cropus)
    
    # Generate text using Hugging Face pipeline
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        do_sample=True, pad_token_id=None, eos_token_id=None,
        num_return_sequences=args.num_return_sequences)
    generator = pipeline(
        'text-generation', model=model, tokenizer=tokenizer, device=args.device)
    outputs = generator(args.prompt, generation_config=generation_config)
    print("\n=== Generated Outputs ===")
    for idx, out in enumerate(outputs):
        print(f"\n--- Variant {idx + 1} ---")
        print(out['generated_text'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a character-level transformer")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
