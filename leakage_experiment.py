import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("THE KAGGLE LEAKAGE EXPERIMENT (SMOTE BEFORE SPLIT)")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'class']
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    X = df.drop('class', axis=1)
    y = df['class']

    # =================================================================
    # THE #1 MOST COMMON KAGGLE MISTAKE:
    # Applying SMOTE to balance the dataset BEFORE splitting it.
    # This guarantees that synthetic copies of test-set rows end up in the training set.
    # =================================================================
    print("Applying SMOTE to ENTIRE dataset (Intentional Leakage)...")
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    smote = SMOTENC(categorical_features=cat_idx, random_state=42)
    
    X_res, y_res = smote.fit_resample(X, y)
    
    # ── SPLIT AFTER LEAKAGE ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    print("Training LightGBM on data with SMOTE leakage...")
    lgbm = LGBMClassifier(
        random_state=42, max_depth=4, learning_rate=0.03,
        n_estimators=200, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    # Evaluate
    test_probs = lgbm.predict_proba(X_test)[:, 1]
    preds = (test_probs >= 0.5).astype(int)
    
    print("\n" + "=" * 60)
    print("LEAKED TEST SET RESULTS (THE 0.80+ F1 ILLUSION)")
    print("=" * 60)
    print(f"  Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision: {precision_score(y_test, preds):.4f}")
    print(f"  Recall:    {recall_score(y_test, preds):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, preds):.4f}  <<< THE FAKE 0.80+ SCORE")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, test_probs):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
