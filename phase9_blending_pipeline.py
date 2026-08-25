import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve,
                             classification_report)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. DOMAIN FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

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
    
    # Adding one more logical interaction from our list: stability
    df['stability'] = df['employment'].astype(str) + '__' + df['housing'].astype(str)
    
    return df

# ─────────────────────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Phase 9: Blending, Cost-Sensitive Learning, Repeated CV")
    logger.info("=" * 70)

    # Load and engineer
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    df = engineer_domain_features(df)

    # Note: LightGBM needs 'category', CatBoost needs list of column names or index
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'class']
    
    # We will keep strings as object for CatBoost, and convert them to category inside LightGBM wrappers
    X = df.drop('class', axis=1)
    y = df['class']

    # ── CLEAN 80/20 SPLIT ──
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"Train: {X_train_full.shape[0]} | Test: {X_test.shape[0]}")
    
    # ── 5 REPEATS × 5-FOLD CV FOR ROBUST OUT-OF-FOLD THRESHOLD TUNING ──
    logger.info("Running Repeated Stratified K-Fold (5 repeats × 5 folds = 25 folds)...")
    
    n_splits = 5
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # We will store ALL out-of-fold predictions across all repeats.
    # Since each row is validated exactly `n_repeats` times, we will have 5 predictions per row.
    # We flatten them all into a massive array to compute the threshold.
    all_oof_probs = []
    all_oof_y = []

    # Model parameters
    scale_weight = 2.33 # Based on the class imbalance 700:300
    
    # Pre-identify categorical features for CatBoost
    cat_features = [X.columns.get_loc(c) for c in cat_cols]
    
    fold_idx = 1
    for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
        X_ft = X_train_full.iloc[train_idx].copy()
        y_ft = y_train_full.iloc[train_idx].copy()
        X_fv = X_train_full.iloc[val_idx].copy()
        
        # Train LightGBM (Needs pandas category type)
        X_ft_lgb = X_ft.copy()
        X_fv_lgb = X_fv.copy()
        for c in cat_cols:
            X_ft_lgb[c] = X_ft_lgb[c].astype('category')
            X_fv_lgb[c] = X_fv_lgb[c].astype('category')
            
        lgbm = LGBMClassifier(
            random_state=42+fold_idx, max_depth=4, learning_rate=0.03,
            n_estimators=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=30,
            scale_pos_weight=scale_weight,
            verbose=-1, importance_type='gain'
        )
        lgbm.fit(X_ft_lgb, y_ft)
        
        # Train CatBoost (Needs raw objects and cat_features list)
        catb = CatBoostClassifier(
            random_seed=42+fold_idx, depth=4, learning_rate=0.03,
            iterations=200, l2_leaf_reg=5.0,
            scale_pos_weight=scale_weight,
            cat_features=cat_features,
            verbose=0
        )
        catb.fit(X_ft, y_ft)
        
        # Blend probabilities (Average)
        lgbm_probs = lgbm.predict_proba(X_fv_lgb)[:, 1]
        catb_probs = catb.predict_proba(X_fv)[:, 1]
        blend_probs = (lgbm_probs + catb_probs) / 2.0
        
        all_oof_probs.extend(blend_probs)
        all_oof_y.extend(y_train_full.iloc[val_idx].values)
        
        if fold_idx % 5 == 0:
            logger.info(f"  Completed Repeat {fold_idx // 5}/{n_repeats}")
        fold_idx += 1

    # ── FIND OPTIMAL THRESHOLD ON BLENDED OOF (4,000 samples) ──
    all_oof_probs = np.array(all_oof_probs)
    all_oof_y = np.array(all_oof_y)
    
    precs, recs, threshs = precision_recall_curve(all_oof_y, all_oof_probs)
    min_precision = 0.54
    best_t, best_r = 0.5, 0.0
    for p, r, t in zip(precs, recs, threshs):
        if p >= min_precision and r > best_r:
            best_r = r
            best_t = t
            
    logger.info(f"Optimal OOF threshold (Precision >= {min_precision}): {best_t:.4f}")

    # ── TRAIN FINAL MODELS ON FULL TRAINING SET ──
    logger.info("Training final Blended Models on full training data...")
    
    X_train_full_lgb = X_train_full.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_train_full_lgb[c] = X_train_full_lgb[c].astype('category')
        X_test_lgb[c] = X_test_lgb[c].astype('category')
        
    final_lgbm = LGBMClassifier(
        random_state=42, max_depth=4, learning_rate=0.03,
        n_estimators=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=30,
        scale_pos_weight=scale_weight,
        verbose=-1, importance_type='gain'
    )
    final_lgbm.fit(X_train_full_lgb, y_train_full)
    
    final_catb = CatBoostClassifier(
        random_seed=42, depth=4, learning_rate=0.03,
        iterations=200, l2_leaf_reg=5.0,
        scale_pos_weight=scale_weight,
        cat_features=cat_features,
        verbose=0
    )
    final_catb.fit(X_train_full, y_train_full)
    
    # ── FINAL EVALUATION ON UNTOUCHED TEST SET ──
    lgbm_test_probs = final_lgbm.predict_proba(X_test_lgb)[:, 1]
    catb_test_probs = final_catb.predict_proba(X_test)[:, 1]
    test_probs = (lgbm_test_probs + catb_test_probs) / 2.0
    
    preds = (test_probs >= best_t).astype(int)

    results = {
        'Strategy': 'N (Blended LGBM+CatBoost + Cost-Sensitive + Repeated CV)',
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, test_probs),
    }

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 9 FINAL RESULTS (Test Set)")
    logger.info("=" * 70)
    for k, v in results.items():
        if k == 'Strategy':
            logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {k:>12s}: {v:.4f}")

    logger.info(f"\n{classification_report(y_test, preds)}")

if __name__ == "__main__":
    main()
