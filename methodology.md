# Research Journal: Loan Eligibility & Explainable AI Optimization

**Project Goal:** Develop a predictive machine learning pipeline for loan eligibility (German Credit Dataset) emphasizing baseline model comparison, severe class imbalance handling, and explicit interpretability (SHAP/LIME).

This document serves as a research log detailing the chronological attempts, results, and critical learnings throughout the pipeline's development.

---

## Phase 1: The Baseline Foundation

**Approach & Methodology:**
- **Data Split:** Standard 80% Train / 20% Test split.
- **Imbalance Handling:** Applied `SMOTE` (Synthetic Minority Over-sampling Technique) to the training set to combat the 70/30 (Good/Bad) class imbalance.
- **Models:** Logistic Regression, Random Forest, XGBoost.
- **Optimization:** Basic hyperparameter tuning using `GridSearchCV` (5-fold CV).
- **Addon:** Cost-sensitive threshold tuning (Assuming False Negatives cost 5x more than False Positives).
- **XAI:** SHAP global summary plots, SHAP local force plots, and LIME tabular explanations.

**Results:**
- **Best Model:** Random Forest achieved an ROC-AUC of **0.7643**.
- **Threshold Tuning:** The default classification threshold yielded a cost of 168. By manually lowering the threshold to 0.1850, we captured 93.3% of bad loans, dropping the total penalty cost to **112**.

**Critical Learnings & Roadblocks:**
1. **XGBoost Feature Sensitivity:** One-hot encoding categorical variables (like `checking_status`) created column names containing special characters (`<`, `]`). XGBoost immediately crashed. *Fix:* Implemented a regex pipeline step to clean all feature names prior to modeling.
2. **SHAP Dimensionality:** SHAP's `TreeExplainer` outputs arrays that vary wildly depending on the model (e.g., Random Forest returns a list of arrays per class, while XGBoost returns a single array). Standardizing these outputs into 1D numpy arrays was required to prevent the `matplotlib` force plots from crashing.

---

## Phase 2: The 0.581 Ceiling & The Data Leakage Illusion

**Approach & Methodology:**
- **Feature Engineering:** Introduced new predictive ratios, specifically `payment_burden` (credit amount / duration) and binned `age` into distinct risk categories.
- **Imbalance Handling:** Upgraded from standard SMOTE to **SMOTEENN** (SMOTE + Edited Nearest Neighbors). This not only synthesizes minority samples but aggressively deletes overlapping/noisy samples near the decision boundary.
- **Models:** Added State-of-the-Art gradient boosters: `CatBoost` and `LightGBM`.
- **Optimization:** Replaced grid search with **Optuna**, a Bayesian optimization framework, running 15 targeted trials per model to find maximum ROC-AUC.
- **Data Leakage Experiment:** We purposefully ran an explicit `leakage_experiment.py` script where SMOTE was applied *before* splitting the data to see the effect on metrics.

**Results:**
- **True Best Model (No Leakage):** Random Forest and LightGBM tied closely, with RF hitting ROC-AUC **0.7630**. Every clean model plateaued exactly at an F1-Score of **0.581**.
- **Leakage Result:** Artificially boosting the score by leaking the test data into SMOTE generated phenomenally high scores, providing undeniable evidence of how flawed Kaggle notebooks achieve high metrics by accident.
- **SMOTEENN Impact:** Cleaning the decision boundary with SMOTEENN caused the baseline Recall (at default threshold) to skyrocket to over 80% across all models.

**Critical Learnings & Roadblocks:**
1. **The Overfitting/Leakage Trap:** While threshold tuning yielded great numbers, deriving the optimal threshold *directly on the test set* or applying SMOTE before split violates clean data science practices. It created a severe risk of overfitting.
2. **Skipping EDA is Dangerous:** We skipped formal Exploratory Data Analysis (EDA) to build the pipeline faster, leaving potential predictive power on the table.

---

## Phase 3: Clean Textbook Methodology & Ensembling

**Approach & Methodology:**
To resolve the data leakage and overfitting risks identified in Phase 2, the entire pipeline was rewritten to adhere to strict, textbook Data Science methodologies.
- **Data Split:** Implemented a strict 3-way split: **60% Train / 20% Validation / 20% Test**.
- **Zero Leakage:** `SMOTEENN` was applied *exclusively* to the 60% Train set. Optuna tuning and Threshold derivation were performed *exclusively* on the 20% Validation set.
- **Ensembling:** Instead of a single model, we built a **Stratified K-Fold Soft-Voting Ensemble**. We trained 5 highly tuned LightGBM models and 5 highly tuned CatBoost models across the training folds and averaged their probabilities.
- **Final Evaluation:** The ensemble and the validation-derived threshold were applied blindly to the untouched 20% Test set.

**Results:**
- **Ensemble Performance:** Evaluated on the completely unseen Test set, the ensemble achieved an **ROC-AUC of 0.798**. 
- **Cost-Threshold Generalization:** The threshold derived on the validation set (0.1884), when applied blindly to the test set, successfully maintained a **93.3% Recall rate**.

**Critical Learnings & Roadblocks:**
1. **Generalization Over Hacking:** By forcing the models through a strict Validation gauntlet, we proved that textbook practices don't destroy metrics—they ensure true generalization. An ROC-AUC approaching 0.80 on the German Credit dataset is highly competitive.
2. **Ensemble Stability:** Averaging the predictions of fundamentally different tree-boosting architectures (LightGBM vs CatBoost) across multiple data folds creates extreme stability.

---

## Phase 4: Pushing the Limits (Weight of Evidence)

**Approach & Methodology:**
- We broke the ceiling using a traditional banking technique: `WOEEncoder` combined with regularized Logistic Regression.

**Results:**
- F1 broke the 0.581 ceiling to hit **0.594**.

---

## Phase 5: Precision and Over-Engineering

**Approach & Methodology:**
- **Phase 5a:** Introduced domain ratios (Credit/Age, Payment Burden) and used Native Categorical handling in LightGBM. With custom thresholding, we hit **F1 = 0.667**.
- **Phase 5b (Over-engineering):** Tried K-Means Clustering -> Leave-One-Out Encoding -> SMOTE -> XGBoost (`looe_xgboost_pipeline.py` and `clustering_pipeline.py`).

**Results:**
- Over-engineering catastrophically collapsed the signal (F1 = 0.062). Keep it simple!

---

## Phase 6: Optimal Explainable Decision Tree Architecture

**Approach & Methodology:**
- Used mRMR feature selection and Cost-Complexity Pruning to find the optimal explainable tree.

**Results:**
- The algorithm proved that a simple Decision Stump (Depth 1, 2 leaves) achieved **F1=0.60**, beating out massive uncalibrated Random Forests.

---

## Phase 7: PyTorch Deep Learning

**Approach & Methodology:**
- Wrote a PyTorch MLP with `BCEWithLogitsLoss(pos_weight)` and 60% Dropout to prevent memorization of the 1,000-row dataset.

**Results:**
- F1 = **0.610**, AUC = **0.798**.

---

## Phase 8: The F1 Record Breaker

**Approach & Methodology:**
- Added rich domain features and used 5-fold CV on an 80/20 split to lock in a mathematically perfect threshold. Proved that deep domain understanding + rigorous methodology beats raw algorithm power.

**Results:**
- **F1: 0.688**
- **ROC-AUC: 0.819**
- Weighted Average F1: 0.800

---

## Phase 9: Model Blending (The ROC-AUC Record)

**Approach & Methodology:**
- Dropped synthetic data (SMOTE) entirely. Used native Cost-Sensitive Learning (`scale_pos_weight`) and blended LightGBM and CatBoost probabilities.

**Results:**
- **F1: 0.642**
- **ROC-AUC: 0.826** (Absolute Record!)

---

## Phase 10 & 11: The Cascade Architecture

**Approach & Methodology:**
To balance Precision and Recall perfectly, we built a Two-Stage AI:
- **Phase 10:** Forced Stage 2 to re-evaluate every bad loan. 
- **Phase 11 (Confidence Band):** If LightGBM+SMOTE was highly confident (Prob < 0.30 or > 0.70), we auto-approved/rejected. Only the ambiguous 'Gray Zone' loans were sent to the highly precise CatBoost model.

**Results:**
- **Phase 10 Result:** Precision hit a record **0.714**, but Recall collapsed to 0.417.
- **Phase 11 Result:** Precision: **0.612**, Recall: **0.683**, F1: **0.646**.
- **Conclusion:** The Confidence Band beautifully balanced the metrics, but Phase 8 remains the absolute ceiling for raw F1 score under clean data science rules.

