import time
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
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        num_beams=5,
        use_cache=args.use_cache,
        temperature=args.temperature, 
        do_sample=True,
    )
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        clean_up_tokenization_spaces=False,
        device=args.device,
    )
    print("Generating text " + ("with" if args.use_cache else "without") + " cache...")
    outputs = generator(args.prompt, generation_config=generation_config)
    if args.device.startswith("cuda"):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            outputs = generator(args.prompt, generation_config=generation_config)
        end_event.record()
        
        # Force CPU to wait for GPU to finish
        torch.cuda.synchronize() 
        elapsed_time_s = start_event.elapsed_time(end_event) / 1000.0
    else:
        # Fallback for CPU execution
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = generator(args.prompt, generation_config=generation_config)
        elapsed_time_s = time.perf_counter() - start_time
    
    prompt_tokens = len(tokenizer.encode(args.prompt))
    total_generated_tokens = 0

    if args.num_return_sequences == 1:
        print(outputs[0]['generated_text'])
        total_generated_tokens = len(tokenizer.encode(outputs[0]['generated_text'])) - prompt_tokens
    else:
        for i, output in enumerate(outputs):
            print(f"Sequence {i+1}:\n{output['generated_text']}\n")
            seq_tokens = len(tokenizer.encode(output['generated_text'])) - prompt_tokens
            total_generated_tokens += seq_tokens
    
    print("-" * 40)
    print(f"Total Time:         {elapsed_time_s:.3f} seconds")
    print(f"Tokens Generated:   {total_generated_tokens} tokens")
    print(f"Throughput:         {total_generated_tokens / elapsed_time_s:.2f} tokens/second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained transformer model")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models/checkpoint-muon")
    parser.add_argument("--prompt", type=str, default="The river Mississippi")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
