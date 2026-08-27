import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def engineer_domain_features(df):
    df = df.copy()
    df['credit_to_age'] = df['credit_amount'] / df['age']
    df['payment_burden'] = df['credit_amount'] / df['duration']
    df['duration_to_age'] = df['duration'] / df['age']

    df['vulnerability_score'] = (
        (df['checking_status'] == 'no checking').astype(int) +
        (df['savings_status'] == 'no known savings').astype(int) +
        (df['property_magnitude'] == 'no known property').astype(int))

    df['total_exposure'] = df['existing_credits'] * df['installment_commitment']

    df['purpose_risk_tier'] = df['purpose'].map({
        'education': 'high_risk', 'other': 'high_risk', 'new car': 'high_risk',
        'repairs': 'medium_risk', 'business': 'medium_risk',
        'domestic appliance': 'medium_risk', 'furniture/equipment': 'medium_risk',
        'radio/tv': 'low_risk', 'used car': 'low_risk', 'retraining': 'low_risk'})

    df['financial_health'] = (
        df['checking_status'].astype(str) + '__' + df['savings_status'].astype(str))
    
    df['stability'] = df['employment'].astype(str) + '__' + df['housing'].astype(str)
    return df


def main():
    logger.info("=" * 70)
    logger.info("Phase 11: The Confidence Band Cascade")
    logger.info("=" * 70)

    # 1. Load Data
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    df = engineer_domain_features(df)
    
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'class']
    
    X = df.drop('class', axis=1)
    y = df['class']

    # 2. Train/Test Split (80/20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
        
    logger.info(f"Train: {X_train_full.shape[0]} | Test: {X_test.shape[0]}")
    cat_idx = [X_train_full.columns.get_loc(c) for c in cat_cols]

    # 3. 5-Fold CV to generate OOF probabilities
    logger.info("Running 5-fold CV to generate Out-Of-Fold probabilities...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_model_A = np.zeros(len(X_train_full))
    oof_model_B = np.zeros(len(X_train_full))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_ft = X_train_full.iloc[train_idx].copy()
        y_ft = y_train_full.iloc[train_idx].copy()
        X_fv = X_train_full.iloc[val_idx].copy()
        
        # --- MODEL A: LightGBM + SMOTE (The Extremes / Boundary Setter) ---
        X_ft_lgb = X_ft.copy()
        X_fv_lgb = X_fv.copy()
        for c in cat_cols:
            X_ft_lgb[c] = X_ft_lgb[c].astype('category')
            X_fv_lgb[c] = X_fv_lgb[c].astype('category')
            
        smote = SMOTENC(categorical_features=cat_idx, random_state=42)
        X_ft_lgb_res, y_ft_res = smote.fit_resample(X_ft_lgb, y_ft)
        
        model_a = LGBMClassifier(
            random_state=42+fold, max_depth=4, learning_rate=0.03,
            n_estimators=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=30,
            verbose=-1, importance_type='gain'
        )
        model_a.fit(X_ft_lgb_res, y_ft_res)
        oof_model_A[val_idx] = model_a.predict_proba(X_fv_lgb)[:, 1]
        
        # --- MODEL B: CatBoost + Cost Sensitive (The Nuance Judge) ---
        model_b = CatBoostClassifier(
            random_seed=42+fold, depth=4, learning_rate=0.03,
            iterations=200, l2_leaf_reg=5.0,
            scale_pos_weight=2.33,
            cat_features=cat_idx,
            verbose=0
        )
        model_b.fit(X_ft, y_ft)
        oof_model_B[val_idx] = model_b.predict_proba(X_fv)[:, 1]

    # 4. Tune the Confidence Band Thresholds
    logger.info("Tuning Confidence Band limits (T_low, T_high) on OOF data...")
    # We want to find T_low and T_high to maximize F1 score.
    # Logic: 
    # If prob_A < T_low -> predict 0
    # If prob_A > T_high -> predict 1
    # If T_low <= prob_A <= T_high -> predict using prob_B (using optimal T_mid)
    
    # First, let's find the best threshold for Model B on its own (T_mid)
    from sklearn.metrics import precision_recall_curve
    precs_B, recs_B, threshs_B = precision_recall_curve(y_train_full, oof_model_B)
    f1s_B = 2 * precs_B * recs_B / (precs_B + recs_B + 1e-10)
    T_mid = threshs_B[np.argmax(f1s_B)]
    
    best_f1 = 0
    best_t_low = 0
    best_t_high = 1.0
    
    # Scan T_low from 0.1 to 0.4
    # Scan T_high from 0.6 to 0.9
    for t_low in np.arange(0.1, 0.45, 0.05):
        for t_high in np.arange(0.55, 0.9, 0.05):
            preds = np.zeros(len(y_train_full))
            for i in range(len(y_train_full)):
                p_a = oof_model_A[i]
                p_b = oof_model_B[i]
                
                if p_a < t_low:
                    preds[i] = 0
                elif p_a > t_high:
                    preds[i] = 1
                else:
                    preds[i] = 1 if p_b >= T_mid else 0
                    
            f1 = f1_score(y_train_full, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t_low = t_low
                best_t_high = t_high
                
    logger.info(f"Optimal Band Found: T_low={best_t_low:.2f}, T_high={best_t_high:.2f}")
    logger.info(f"Model B optimal threshold: T_mid={T_mid:.4f}")
    logger.info(f"OOF F1 with this band: {best_f1:.4f}")
    
    # Let's see how many samples fall into the Gray Zone
    gray_zone = (oof_model_A >= best_t_low) & (oof_model_A <= best_t_high)
    logger.info(f"Samples in Gray Zone (Evaluated by Model B): {gray_zone.sum()} out of {len(X_train_full)}")

    # 5. Train Final Models
    logger.info("Training final Phase 11 models on full training data...")
    X_train_full_lgb = X_train_full.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_train_full_lgb[c] = X_train_full_lgb[c].astype('category')
        X_test_lgb[c] = X_test_lgb[c].astype('category')
        
    smote_final = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_final_res, y_final_res = smote_final.fit_resample(X_train_full_lgb, y_train_full)
    
    final_model_a = LGBMClassifier(
        random_state=42, max_depth=4, learning_rate=0.03,
        n_estimators=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=30,
        verbose=-1, importance_type='gain'
    )
    final_model_a.fit(X_final_res, y_final_res)
    
    final_model_b = CatBoostClassifier(
        random_seed=42, depth=4, learning_rate=0.03,
        iterations=200, l2_leaf_reg=5.0,
        scale_pos_weight=2.33,
        cat_features=cat_idx,
        verbose=0
    )
    final_model_b.fit(X_train_full, y_train_full)

    # 6. Final Evaluation
    test_probs_a = final_model_a.predict_proba(X_test_lgb)[:, 1]
    test_probs_b = final_model_b.predict_proba(X_test)[:, 1]
    
    final_preds = np.zeros(len(y_test))
    
    for i in range(len(y_test)):
        p_a = test_probs_a[i]
        p_b = test_probs_b[i]
        
        if p_a < best_t_low:
            final_preds[i] = 0
        elif p_a > best_t_high:
            final_preds[i] = 1
        else:
            final_preds[i] = 1 if p_b >= T_mid else 0
            
    results = {
        'Strategy': 'P (Confidence Band Cascade)',
        'Accuracy': accuracy_score(y_test, final_preds),
        'Precision': precision_score(y_test, final_preds),
        'Recall': recall_score(y_test, final_preds),
        'F1': f1_score(y_test, final_preds),
    }

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 11 FINAL RESULTS (Test Set)")
    logger.info("=" * 70)
    for k, v in results.items():
        if k == 'Strategy':
            logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {k:>12s}: {v:.4f}")

    logger.info(f"\n{classification_report(y_test, final_preds)}")

if __name__ == "__main__":
    main()
