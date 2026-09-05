# COURSEWORK 1 – MASTER PLAN

## 1. Project title
**CNN-Based Patient Age Regression from NIH ChestX-ray14**

## 2. Problem statement
Given a chest X-ray image, train a convolutional neural network to predict the patient's age as a continuous numerical value.

- Input: chest X-ray image
- Target: Patient Age
- Task: Regression
- Model requirement: CNN + Global Average Pooling + Linear output
- Main extensions: compare MSE vs MAE; test data augmentation

## 3. Research questions
1. Can a CNN learn to predict patient age from chest X-ray images?
2. Does MSE or MAE provide better age-regression performance?
3. Does appropriate image augmentation improve generalization?

## 4. Final pipeline
NIH ChestX-ray14
→ metadata validation/cleaning
→ patient-level train/validation/test split
→ image preprocessing
→ CNN feature extractor
→ Global Average Pooling
→ Linear(1)
→ age prediction
→ MAE/MSE/RMSE evaluation
→ controlled experiments
→ comparison/discussion

## 5. Team workflow

### Phase A – Foundation
**Member 02** prepares the dataset pipeline first.

Output:
- validated metadata
- patient-level train/val/test split
- reusable Dataset/DataLoader
- baseline transforms

### Phase B – Model
**Member 03** builds the CNN independently using the agreed input shape and output contract.

Output:
- model code
- architecture diagram
- parameter count
- forward-pass test

### Phase C – Training baseline
**Member 01 + Member 03** integrate DataLoader + CNN and run one smoke-test training.

Output:
- one complete end-to-end run
- saved configuration
- first loss/MAE curves

### Phase D – Loss comparison
**Member 04** runs MSE and MAE under the same conditions.

Output:
- MSE experiment
- MAE experiment
- comparable metrics and curves

### Phase E – Augmentation
**Member 05** adds medically reasonable training-only augmentation.

Output:
- augmentation pipeline
- before/after examples
- augmented experiments

### Phase F – Evaluation
**Member 06** evaluates all final models on the untouched test set.

Output:
- MAE, MSE, RMSE
- actual-vs-predicted plot
- residual/error distribution
- final comparison table

### Phase G – Integration and submission
**Member 01** reviews, integrates and freezes the final version.

Output:
- final notebook
- final results
- report/presentation materials
- reproducibility checklist

## 6. Required experiments

| ID | Model | Loss | Augmentation | Purpose |
|---|---|---|---|---|
| E1 | CNN + GAP + Linear(1) | MSE | No | Baseline |
| E2 | Same | MAE/L1 | No | Compare loss functions |
| E3 | Same | MSE | Yes | Test augmentation with MSE |
| E4 | Same | MAE/L1 | Yes | Test augmentation with MAE |

Keep the following fixed when comparing losses: data split, preprocessing, architecture, optimizer, learning rate, batch size, epochs and random seed. When testing augmentation, change only the augmentation component.

## 7. Evaluation metrics
- MAE: primary and easiest-to-interpret metric, measured in years.
- MSE: required for loss comparison.
- RMSE: secondary metric, measured in years.

Do not use classification accuracy as the primary metric because this is a regression problem.

## 8. Data leakage rule
The split must be performed at **Patient ID level**, not independently by image. A patient must belong to only one of train/validation/test.

## 9. Train/validation/test rule
Recommended starting split: 70/15/15 or 80/10/10 by unique patient. The exact split must be documented. Use a fixed random seed.

## 10. Transform rule
Training:
- resize
- tensor conversion
- normalization
- approved mild augmentation for augmentation experiments

Validation/Test:
- resize
- tensor conversion
- normalization
- NO random augmentation

## 11. Model contract
Input shape should be consistent, e.g. `[B, 1, 224, 224]` for grayscale input.

Output must be `[B, 1]`, representing predicted age.

Required architecture:
CNN → Global Average Pooling → Linear(1)

## 12. Required visualizations
1. Age distribution in the dataset.
2. Training vs validation loss by epoch.
3. Validation MAE by epoch.
4. Actual Age vs Predicted Age with reference line y=x.
5. Prediction error/residual distribution.
6. Final experiment comparison.

## 13. Report structure
1. Introduction
2. Problem Definition
3. Dataset
4. Data Preparation
5. CNN Architecture
6. Training Setup
7. Experimental Design
8. Results
9. Discussion
10. Limitations
11. Conclusion
12. References

## 14. What each member must submit
Every member must provide:
- code/notebook for their responsibility;
- short README explaining what was done;
- result files/figures where applicable;
- reproducible instructions;
- a clear commit history.

## 15. Git workflow
Recommended branches:
- `coursework/member-01-leader`
- `coursework/member-02-data`
- `coursework/member-03-model`
- `coursework/member-04-loss`
- `coursework/member-05-augmentation`
- `coursework/member-06-evaluation`

Each member:
1. pulls latest `main`;
2. works only on their branch/workspace;
3. commits logically;
4. pushes branch;
5. opens Pull Request to `main`;
6. Member 01 reviews before merge.

## 16. Definition of Done
The coursework is ready only when:
- [ ] dataset source and preprocessing are documented;
- [ ] patient-level split is verified;
- [ ] CNN + GAP + Linear(1) is implemented;
- [ ] baseline MSE model trains successfully;
- [ ] MAE experiment is complete;
- [ ] augmentation experiment is complete;
- [ ] all four experiments are evaluated on the same untouched test set;
- [ ] MAE/MSE/RMSE are reported;
- [ ] required plots are generated;
- [ ] results are interpreted, not merely displayed;
- [ ] limitations are discussed;
- [ ] final notebook runs from top to bottom;
- [ ] no large raw dataset is committed to GitHub;
- [ ] README explains how to reproduce the work.
