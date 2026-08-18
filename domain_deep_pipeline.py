import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve,
                             classification_report)
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. DOMAIN FEATURE ENGINEERING
#    Every transformation is a pure function of the row itself.
#    No cross-row statistics — safe to apply before splitting.
# ─────────────────────────────────────────────────────────────

def engineer_domain_features(df):
    df = df.copy()

    # Numeric ratios (Phase 4 originals)
    df['credit_to_age'] = df['credit_amount'] / df['age']
    df['payment_burden'] = df['credit_amount'] / df['duration']
    df['duration_to_age'] = df['duration'] / df['age']

    # Domain composites (Phase 8 additions)
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

    # Combined financial health: checking × savings
    df['financial_health'] = (
        df['checking_status'].astype(str) + '__' + df['savings_status'].astype(str))

    return df

# ─────────────────────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 8: Domain-Deep + CV Threshold Pipeline")
    logger.info("=" * 60)

    # Load and engineer
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    df = engineer_domain_features(df)

    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'class']
    for c in cat_cols:
        df[c] = df[c].astype('category')

    X = df.drop('class', axis=1)
    y = df['class']

    # ── CLEAN 80/20 SPLIT ──
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"Train: {X_train_full.shape[0]} | Test: {X_test.shape[0]}")
    cat_idx = [X_train_full.columns.get_loc(c) for c in cat_cols]

    # ── 5-FOLD CV FOR OUT-OF-FOLD THRESHOLD TUNING ──
    logger.info("Running 5-fold CV for out-of-fold threshold estimation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X_train_full))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_ft = X_train_full.iloc[train_idx]
        y_ft = y_train_full.iloc[train_idx]
        X_fv = X_train_full.iloc[val_idx]

        # SMOTENC inside each fold — zero leakage
        smote = SMOTENC(categorical_features=cat_idx, random_state=42)
        Xr, yr = smote.fit_resample(X_ft, y_ft)

        ests = [(f'lgbm_{i}', LGBMClassifier(
            random_state=42+i, max_depth=4, learning_rate=0.03,
            n_estimators=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=30,
            verbose=-1, importance_type='gain')) for i in range(5)]

        ens = VotingClassifier(estimators=ests, voting='soft')
        ens.fit(Xr, yr)
        oof_probs[val_idx] = ens.predict_proba(X_fv)[:, 1]
        logger.info(f"  Fold {fold+1}/5 complete.")

    # ── FIND OPTIMAL THRESHOLD ON OOF (800 samples) ──
    precs, recs, threshs = precision_recall_curve(y_train_full, oof_probs)
    min_precision = 0.54
    best_t, best_r = 0.5, 0.0
    for p, r, t in zip(precs, recs, threshs):
        if p >= min_precision and r > best_r:
            best_r = r
            best_t = t
    logger.info(f"Optimal OOF threshold (Precision >= {min_precision}): {best_t:.4f}")

    # ── TRAIN FINAL MODEL ON FULL TRAINING SET ──
    logger.info("Training final 5-model LightGBM ensemble on full training data...")
    smote_final = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_final_res, y_final_res = smote_final.fit_resample(X_train_full, y_train_full)

    final_ests = [(f'lgbm_{i}', LGBMClassifier(
        random_state=42+i, max_depth=4, learning_rate=0.03,
        n_estimators=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=30,
        verbose=-1, importance_type='gain')) for i in range(5)]

    final_ens = VotingClassifier(estimators=final_ests, voting='soft')
    final_ens.fit(X_final_res, y_final_res)

    # ── FINAL EVALUATION ON UNTOUCHED TEST SET ──
    test_probs = final_ens.predict_proba(X_test)[:, 1]
    preds = (test_probs >= best_t).astype(int)

    results = {
        'Strategy': 'M (Domain-Deep + CV Threshold + LightGBM Ensemble)',
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, test_probs),
    }

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 8 FINAL RESULTS (Test Set)")
    logger.info("=" * 60)
    for k, v in results.items():
        if k == 'Strategy':
            logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {k:>12s}: {v:.4f}")

    logger.info(f"\n{classification_report(y_test, preds)}")

    df_results = pd.DataFrame([results])
    logger.info(f"\n{df_results.to_markdown(index=False)}")

if __name__ == "__main__":
    main()
