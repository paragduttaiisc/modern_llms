import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline, set_seed

from model import Model, ModelConfig
from utils import get_tokenizer


def main(args: argparse.Namespace):
    # Register and load model
    AutoConfig.register("modern_transformer", ModelConfig)
    AutoModelForCausalLM.register(ModelConfig, Model)
    model = Model.from_pretrained(args.save_dir).to(args.device)
    model.eval()
    tokenizer = get_tokenizer()
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, 
        num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        do_sample=True, num_return_sequences=args.num_return_sequences)
    generator = pipeline(
        'text-generation', model=model, tokenizer=tokenizer, clean_up_tokenization_spaces=False, device=args.device)
    outputs = generator(args.prompt, generation_config=generation_config)
    if args.num_return_sequences == 1:
        print(outputs[0]['generated_text'])
    else:
        for i, output in enumerate(outputs):
            print(f"Sequence {i+1}: {output['generated_text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained transformer model")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--prompt", type=str, default="The river Mississippi")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
