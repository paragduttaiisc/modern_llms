# Modern LLMs

Extending Karpathy's nanoGPT to include recent techniques and train modern billion-parameter LLMs from scratch on local hardware.

## Description

This project builds on nanoGPT and experiments with newer training techniques to scale to modern, large language models that can be trained on commodity/local hardware (where possible).

## Getting Started

- Clone the repo
- Create a Python virtual environment and install requirements

Example:

```bash
git clone git@github.com:paragduttaiisc/modern_llms.git
cd modern_llms
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Train and Infer

- To train a model, run `train.py` with appropriate arguments. For example:

```bash
accelerate launch train.py --save-dir models/your_model --max-iters 500 --wandb-name your_model_run
```

- To generate text using a trained model, run `infer.py` with appropriate arguments. For example:

```bash
python infer.py --save-dir models/your_model --prompt "To be, or not" --max-new-tokens 200 --temperature 0.85 --num-return-sequences 3
```

## License

Creative Commons Zero v1.0 Universal
