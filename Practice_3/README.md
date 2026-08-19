# Practice 3 - Get started with Hugging Face

## Chapter 4: Fine-tune Models and Training Algorithms

Practice 3 gồm 2 bài tập theo yêu cầu của bài giảng:

### Exercise 1 - Sentiment Analysis with Hugging Face
- Install `transformers`.
- Use a pre-trained sentiment analysis model from Hugging Face Hub.
- Tokenize a sample sentence.
- Perform sentiment analysis.

### Exercise 2 - Finetuning a Pretrained Model for Binary Text Classification
- Install `transformers`, `datasets`, `evaluate`.
- Load a binary text classification dataset.
- Load a pretrained model and tokenizer.
- Preprocess the dataset.
- Define training arguments.
- Create `Trainer` and fine-tune the model.
- Evaluate the fine-tuned model.

## Repository structure

```text
Practice_3/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_sentiment_analysis.ipynb
│   └── 02_finetuning_binary_text_classification.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── configs/
├── results/
├── checkpoints/
└── docs/
```

> Dataset files, model checkpoints, cache files, and generated outputs should not be committed unless they are intentionally required for submission.
