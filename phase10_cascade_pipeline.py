import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve,
                             classification_report)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC

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
    
    df['stability'] = df['employment'].astype(str) + '__' + df['housing'].astype(str)
    
    return df

# ─────────────────────────────────────────────────────────────
# 2. MAIN CASCADE PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Phase 10: Two-Stage Cascade Architecture")
    logger.info("=" * 70)

    # Load and engineer
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    df = engineer_domain_features(df)

    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'class']
    
    X = df.drop('class', axis=1)
    y = df['class']

    # ── CLEAN 80/20 SPLIT ──
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"Train: {X_train_full.shape[0]} | Test: {X_test.shape[0]}")
    
    cat_idx = [X_train_full.columns.get_loc(c) for c in cat_cols]
    
    # ── 5-FOLD CV FOR OUT-OF-FOLD THRESHOLD TUNING ──
    logger.info("Running 5-fold CV to tune thresholds for both stages...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store Out-of-fold predictions
    oof_stage1_probs = np.zeros(len(X_train_full))
    oof_stage2_probs = np.zeros(len(X_train_full))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_ft = X_train_full.iloc[train_idx].copy()
        y_ft = y_train_full.iloc[train_idx].copy()
        X_fv = X_train_full.iloc[val_idx].copy()
        
        # --- STAGE 1: LightGBM + SMOTENC (The Wide Net) ---
        X_ft_lgb = X_ft.copy()
        X_fv_lgb = X_fv.copy()
        for c in cat_cols:
            X_ft_lgb[c] = X_ft_lgb[c].astype('category')
            X_fv_lgb[c] = X_fv_lgb[c].astype('category')
            
        smote = SMOTENC(categorical_features=cat_idx, random_state=42)
        X_ft_lgb_res, y_ft_res = smote.fit_resample(X_ft_lgb, y_ft)
        
        stage1_model = LGBMClassifier(
            random_state=42+fold, max_depth=4, learning_rate=0.03,
            n_estimators=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=30,
            verbose=-1, importance_type='gain'
        )
        stage1_model.fit(X_ft_lgb_res, y_ft_res)
        oof_stage1_probs[val_idx] = stage1_model.predict_proba(X_fv_lgb)[:, 1]
        
        # --- STAGE 2: CatBoost + Cost-Sensitive (The Precise Judge) ---
        # NO SMOTE here! Pure data with scale_pos_weight
        stage2_model = CatBoostClassifier(
            random_seed=42+fold, depth=4, learning_rate=0.03,
            iterations=200, l2_leaf_reg=5.0,
            scale_pos_weight=2.33, # 700/300 class ratio
            cat_features=cat_idx,
            verbose=0
        )
        stage2_model.fit(X_ft, y_ft)
        oof_stage2_probs[val_idx] = stage2_model.predict_proba(X_fv)[:, 1]
        
        logger.info(f"  Completed Fold {fold+1}/5")

    # ── TUNE STAGE 1 (MAX RECALL) ──
    precs1, recs1, threshs1 = precision_recall_curve(y_train_full, oof_stage1_probs)
    target_recall = 0.90
    t1 = 0.5
    for p, r, t in zip(precs1, recs1, threshs1):
        if r >= target_recall:
            t1 = t # keep updating to get the highest threshold that still maintains 0.90 recall
    
    logger.info(f"Stage 1 Threshold (Recall >= {target_recall}): {t1:.4f}")
    
    # ── TUNE STAGE 2 (MAX PRECISION) ──
    # We only tune Stage 2 on the instances that Stage 1 ACTUALLY flagged as Bad.
    stage1_flags = oof_stage1_probs >= t1
    y_stage2_target = y_train_full[stage1_flags]
    stage2_probs_on_flagged = oof_stage2_probs[stage1_flags]
    
    precs2, recs2, threshs2 = precision_recall_curve(y_stage2_target, stage2_probs_on_flagged)
    target_prec = 0.65 # We want to be highly precise
    t2, max_r2 = 0.5, 0.0
    for p, r, t in zip(precs2, recs2, threshs2):
        if p >= target_prec and r > max_r2:
            max_r2 = r
            t2 = t
    
    if max_r2 == 0:
        logger.warning(f"Could not reach {target_prec} precision on Stage 2. Falling back to Max F1.")
        f1s2 = 2 * precs2 * recs2 / (precs2 + recs2 + 1e-10)
        t2 = threshs2[np.argmax(f1s2)]
        
    logger.info(f"Stage 2 Threshold (Precision >= {target_prec} on Flagged): {t2:.4f}")

    # ── TRAIN FINAL MODELS ON FULL TRAINING SET ──
    logger.info("Training final Stage 1 & 2 models on full training data...")
    
    X_train_full_lgb = X_train_full.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_train_full_lgb[c] = X_train_full_lgb[c].astype('category')
        X_test_lgb[c] = X_test_lgb[c].astype('category')
        
    smote_final = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_final_res, y_final_res = smote_final.fit_resample(X_train_full_lgb, y_train_full)
    
    final_stage1 = LGBMClassifier(
        random_state=42, max_depth=4, learning_rate=0.03,
        n_estimators=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=30,
        verbose=-1, importance_type='gain'
    )
    final_stage1.fit(X_final_res, y_final_res)
    
    final_stage2 = CatBoostClassifier(
        random_seed=42, depth=4, learning_rate=0.03,
        iterations=200, l2_leaf_reg=5.0,
        scale_pos_weight=2.33,
        cat_features=cat_idx,
        verbose=0
    )
    final_stage2.fit(X_train_full, y_train_full)
    
    # ── FINAL EVALUATION (THE CASCADE) ──
    test_probs_s1 = final_stage1.predict_proba(X_test_lgb)[:, 1]
    test_probs_s2 = final_stage2.predict_proba(X_test)[:, 1]
    
    final_preds = np.zeros(len(y_test))
    
    for i in range(len(y_test)):
        # Stage 1 prediction
        if test_probs_s1[i] >= t1:
            # Flagged by Stage 1 -> Sent to Stage 2 Judge
            if test_probs_s2[i] >= t2:
                final_preds[i] = 1 # Both agree it's Bad
            else:
                final_preds[i] = 0 # Stage 2 overrides
        else:
            final_preds[i] = 0 # Cleared by Stage 1 immediately

    results = {
        'Strategy': 'O (Two-Stage Cascade Classification)',
        'Accuracy': accuracy_score(y_test, final_preds),
        'Precision': precision_score(y_test, final_preds),
        'Recall': recall_score(y_test, final_preds),
        'F1': f1_score(y_test, final_preds),
    }

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 10 FINAL RESULTS (Test Set)")
    logger.info("=" * 70)
    for k, v in results.items():
        if k == 'Strategy':
            logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {k:>12s}: {v:.4f}")

    logger.info(f"\n{classification_report(y_test, final_preds)}")

if __name__ == "__main__":
    main()
