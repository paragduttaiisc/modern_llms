import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline, set_seed

from model import Model, ModelConfig
from data_utils import get_tokenizer


def main(args: argparse.Namespace):
    # Register and load model
    AutoConfig.register("modern_transformer", ModelConfig)
    AutoModelForCausalLM.register(ModelConfig, Model)
    model = Model.from_pretrained(args.save_dir).to(args.device)
    model.eval()

    # load tokenizer
    tokenizer = get_tokenizer()
    
    # Generate text using Hugging Face pipeline
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        do_sample=True, num_return_sequences=args.num_return_sequences)
    generator = pipeline(
        'text-generation', model=model, tokenizer=tokenizer, clean_up_tokenization_spaces=False, device=args.device)
    outputs = generator(args.prompt, generation_config=generation_config)
    print("\n=== Generated Outputs ===")
    for idx, out in enumerate(outputs):
        print(f"\n--- Variant {idx + 1} ---")
        print(out['generated_text'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained transformer model")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
