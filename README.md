# Hallucination Detection in Large Language Models Using Generation-Time Uncertainty Signals

This project investigates hallucination detection in large language models by analyzing uncertainty signals produced during text generation. At this stage, the repository only contains the code for loading and verifying the dataset used in the study.

## Dataset

[TruthfulQA](https://huggingface.co/datasets/truthful_qa) (generation configuration), loaded via the Hugging Face `datasets` library.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python load_dataset.py
```
