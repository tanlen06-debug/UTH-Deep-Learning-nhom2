# Practice 3 - Notebooks

Practice 3 is organized like Practice 2: **6 members = 6 working notebooks**. Each member owns one notebook, while the six notebooks together form the complete Practice 3 workflow.

| Notebook | Member | Main responsibility | Dependency |
|---|---|---|---|
| `01_setup_project_overview.ipynb` | TV1 - Group Leader | Project setup, common configuration, integration, final checks and summary | All notebooks |
| `02_sentiment_analysis.ipynb` | TV2 | Exercise 1: pretrained sentiment analysis, tokenization and prediction | Independent |
| `03_dataset_preprocessing.ipynb` | TV3 | Exercise 2: load binary text classification dataset and preprocess/tokenize it | Dataset choice from TV1 |
| `04_model_tokenizer.ipynb` | TV4 | Exercise 2: load compatible pretrained model and tokenizer | Dataset/task choice from TV1 |
| `05_training_finetuning.ipynb` | TV5 | Exercise 2: training arguments, `Trainer`, and fine-tuning | TV3 + TV4 |
| `06_evaluation.ipynb` | TV6 | Exercise 2: evaluate the fine-tuned model and analyze results | TV5 |

## Recommended workflow

```text
TV1: Project setup + agree dataset/model
              │
      ┌───────┴────────┐
      ▼                ▼
    TV3              TV4
 Dataset +          Model +
Preprocessing      Tokenizer
      └───────┬────────┘
              ▼
             TV5
       Trainer + Fine-tuning
              │
              ▼
             TV6
          Evaluation
              │
              ▼
             TV1
      Final integration
```

TV2 can work in parallel because Exercise 1 is independent of the Exercise 2 training pipeline.

## Working rules

1. Each member works only in their assigned notebook unless the group leader asks for a shared change.
2. TV1 must approve the common dataset and pretrained model before TV3/TV4 finalize their work.
3. TV3 and TV4 must provide reproducible code and clearly state the outputs needed by TV5.
4. TV5 must record the training configuration and final model location needed by TV6.
5. TV6 must report evaluation results and explain what the metrics mean.
6. Every member must be able to explain their own notebook during the lab presentation.
7. Before submission, TV1 runs all six notebooks and checks that the complete workflow is consistent.
