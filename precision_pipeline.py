import pandas as pd
import numpy as np
import re
import warnings
import logging
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def clean_cols(df):
    regex = re.compile(r'[\[\]<>]')
    df.columns = [regex.sub('_', str(col)) for col in df.columns]
    return df

def feature_engineering(df):
    df['credit_to_age'] = df['credit_amount'] / df['age']
    df['payment_burden'] = df['credit_amount'] / df['duration']
    df['duration_to_age'] = df['duration'] / df['age']
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        
    return df, categorical_cols

def load_and_split():
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    df, cat_cols = feature_engineering(df)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    return X_train, y_train, X_val, y_val, X_test, y_test, cat_cols

def tune_threshold_precision(model, X_val, y_val, target_precision=0.52):
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    best_threshold = 0.5
    best_recall = 0.0
    
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= target_precision and r > best_recall:
            best_recall = r
            best_threshold = t
            
    # Fallback to max F1 if we simply cannot hit the target precision with any non-zero recall
    if best_recall == 0:
        logger.warning(f"Could not hit precision > {target_precision}. Falling back to standard F1 maximization.")
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
    return best_threshold

def evaluate(model, X_test, y_test, threshold):
    test_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred = (test_probs >= threshold).astype(int)
    return {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1': f1_score(y_test, y_test_pred),
        'ROC-AUC': roc_auc_score(y_test, test_probs)
    }

def main():
    logger.info("Starting Phase 4: Precision Target Pipeline...")
    X_train, y_train, X_val, y_val, X_test, y_test, cat_cols = load_and_split()
    
    # Identify indices for categorical columns for SMOTENC
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    
    logger.info("Applying SMOTENC for oversampling Native Categories...")
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_train_res, y_train_res = smote_nc.fit_resample(X_train, y_train)
    
    logger.info("Training Native LightGBM Bagging Ensemble...")
    # Train 5 LightGBM models with different seeds
    estimators = []
    for i in range(5):
        lgbm = LGBMClassifier(
            random_state=42+i,
            max_depth=4,
            learning_rate=0.03,
            n_estimators=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=30,
            verbose=-1,
            importance_type='gain'
        )
        estimators.append((f'lgbm_{i}', lgbm))
        
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train_res, y_train_res)
    
    target_prec = 0.52
    logger.info(f"Tuning Threshold specifically for Precision >= {target_prec}...")
    thresh = tune_threshold_precision(ensemble, X_val, y_val, target_precision=target_prec)
    logger.info(f"Selected Threshold: {thresh:.4f}")
    
    results = evaluate(ensemble, X_test, y_test, thresh)
    
    logger.info("\n--- Phase 4 Results (Test Set) ---")
    df_results = pd.DataFrame([{'Strategy': 'H & I (Domain + Native LGBM + Precision Tuning)', **results}])
    logger.info(f"\n{df_results.to_markdown(index=False)}")
    
    with open("precision_model_results.md", "w") as f:
        f.write("# Phase 4 Precision Target Results\n\n")
        f.write(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
