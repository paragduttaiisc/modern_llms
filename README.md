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

## Notes

- Data files are excluded from version control (`data/` is ignored).
- Adjust training scripts and hyperparameters in `train_utils.py` and `main.py`.

## License

Creative Commons Zero v1.0 Universal
