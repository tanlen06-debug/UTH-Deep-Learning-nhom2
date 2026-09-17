# MEMBER TASK SPECIFICATIONS

## MEMBER 01 – GROUP LEADER
### Objective
Control integration, quality and final submission.
### Must do
- [ ] Confirm target = Patient Age and task = regression.
- [ ] Freeze the agreed model interface: CNN → GAP → Linear(1).
- [ ] Maintain experiment matrix E1–E4.
- [ ] Review Member 02 data split before model training.
- [ ] Review Member 03 model shape and architecture.
- [ ] Review Member 04 loss comparison fairness.
- [ ] Review Member 05 augmentation validity.
- [ ] Review Member 06 test-set evaluation.
- [ ] Integrate final notebook and run from top to bottom.
- [ ] Prepare final presentation and Q&A.
### Deliverables
`integration/final_coursework.ipynb`, project checklist, final experiment table.

---

## MEMBER 02 – DATASET & PREPROCESSING
### Objective
Create a trustworthy image/target pipeline without leakage.
### Steps
1. Obtain/read NIH ChestX-ray14 metadata.
2. Identify image filename, Patient ID and Patient Age columns.
3. Inspect missing/invalid ages and duplicate records.
4. Produce age distribution and basic statistics.
5. Validate image paths and unreadable images.
6. Split unique patients into train/validation/test with a fixed seed.
7. Verify patient sets are disjoint.
8. Build PyTorch Dataset and DataLoader.
9. Implement baseline preprocessing.
### Deliverables
`notebooks/02_data_preprocessing.ipynb`, reusable dataset code, processed metadata files and EDA figures.
### Acceptance test
`set(train_patient_ids) ∩ set(val_patient_ids) = ∅`, and equivalent for every pair.

---

## MEMBER 03 – CNN MODEL
### Objective
Implement the exact required model family.
### Architecture
Input `[B,1,224,224]`
→ Conv Block(s)
→ Global Average Pooling
→ Flatten
→ Linear(1)
→ output `[B,1]`.
### Steps
1. Implement model in PyTorch.
2. Test a forward pass with random input.
3. Print tensor shapes.
4. Count trainable parameters.
5. Document each layer's purpose.
6. Avoid adding an unrelated pretrained architecture unless explicitly approved.
### Deliverables
`src/model.py`, model notebook, architecture diagram and parameter summary.
### Acceptance test
A sample batch must pass forward propagation and produce one scalar prediction per image.

---

## MEMBER 04 – LOSS COMPARISON
### Objective
Answer whether MSE or MAE is more suitable under controlled conditions.
### Experiments
- E1: baseline CNN + MSELoss.
- E2: baseline CNN + L1Loss/MAE.
### Keep fixed
Dataset split, transforms, architecture, optimizer, learning rate, batch size, epochs and random seed.
### Record
Train loss, validation loss, validation MAE, test MAE, test MSE and test RMSE.
### Deliverables
`notebooks/04_loss_comparison.ipynb`, metrics CSV and comparison figures.
### Acceptance test
The comparison must change the loss function, not the entire training recipe.

---

## MEMBER 05 – DATA AUGMENTATION
### Objective
Test whether mild, realistic training augmentation improves generalization.
### Steps
1. Establish baseline transform.
2. Propose mild augmentation.
3. Visualize augmented samples.
4. Apply random augmentation to training only.
5. Keep validation/test deterministic.
6. Run E3 and E4.
7. Compare against E1/E2.
### Deliverables
`notebooks/05_augmentation.ipynb`, transform code, sample visualizations and results.
### Acceptance test
No random augmentation is used on validation/test. Augmentation must not create obviously unrealistic anatomy.

---

## MEMBER 06 – EVALUATION & REPORT
### Objective
Turn model outputs into defensible evidence and a clear story.
### Steps
1. Load the best/final checkpoints.
2. Evaluate only on the untouched test set.
3. Compute MAE, MSE and RMSE.
4. Plot actual vs predicted age with y=x reference.
5. Plot residual/error distribution.
6. Compare E1–E4 in one table.
7. Identify the best model using the predefined primary metric.
8. Discuss overfitting, errors and limitations.
9. Draft Discussion and Conclusion.
### Deliverables
`notebooks/06_evaluation.ipynb`, final metrics table, figures, Discussion, Conclusion and presentation notes.
### Acceptance test
No test-set result is used to tune the model after seeing it.

---

# SHARED OUTPUT CONTRACT
All members must use the same:
- target definition: Patient Age;
- image preprocessing convention;
- train/validation/test split;
- random seed where applicable;
- output convention `[B,1]`;
- metric definitions.

# HANDOFF ORDER
**02 Data → 03 Model → 04 Loss / 05 Augmentation → 06 Evaluation → 01 Integration**

Do not start final evaluation until the data split and model contract are frozen.
