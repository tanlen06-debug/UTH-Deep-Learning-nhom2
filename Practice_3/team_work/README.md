# Practice 3 – Team Workspace

This folder is the shared workspace for the six members working on Practice 3.

## Assignment flow

```text
TV1 – Group Leader
        |
   +----+----+
   |         |
  TV2       TV3 + TV4
Exercise 1      |
                v
               TV5
        Training/Fine-tuning
                |
                v
               TV6
          Evaluation
                |
                v
               TV1
             Final QA
```

## Work folders

- `TV1_Group_Leader/` – integration, GitHub management, final review
- `TV2_Exercise_1/` – Hugging Face sentiment analysis
- `TV3_Dataset_Preprocessing/` – binary dataset and preprocessing
- `TV4_Model_Tokenizer/` – pretrained model and tokenizer
- `TV5_Training_Finetuning/` – Trainer and fine-tuning
- `TV6_Evaluation_Documentation/` – evaluation and analysis

## Shared notebook policy

Exercise 2 is maintained as one integrated notebook:

`../notebooks/02_finetuning_binary_text_classification.ipynb`

Members should contribute their assigned section and coordinate before modifying another member's section.

## Git workflow recommendation

Each member should work on a personal branch and submit a Pull Request to `main`. The group leader reviews and merges the changes.
