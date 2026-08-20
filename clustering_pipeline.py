import pandas as pd
import numpy as np
import re
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from category_encoders import LeaveOneOutEncoder

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
    return df

def load_and_split():
    df = pd.read_csv('German_Credit_Data.csv')
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    df = feature_engineering(df)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    return X_train, y_train, X_val, y_val, X_test, y_test

def tune_threshold_precision(model, X_val, y_val, target_precision=0.52):
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    best_threshold = 0.5
    best_recall = 0.0
    
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= target_precision and r > best_recall:
            best_recall = r
            best_threshold = t
            
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
    logger.info("Starting Phase 5: Unsupervised Clustering & LOOE Pipeline...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    # 1. Unsupervised Clustering Feature Injection
    logger.info("Fitting K-Means strictly on X_train numericals...")
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_val_num_scaled = scaler.transform(X_val[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    X_train['risk_cluster'] = kmeans.fit_predict(X_train_num_scaled).astype(str)
    
    # STRICT NO-LEAKAGE: Predict only on Val and Test
    X_val['risk_cluster'] = kmeans.predict(X_val_num_scaled).astype(str)
    X_test['risk_cluster'] = kmeans.predict(X_test_num_scaled).astype(str)
    
    cat_cols.append('risk_cluster')
    
    # 2. Leave-One-Out Encoding (LOOE)
    logger.info("Applying Leave-One-Out Target Encoding...")
    looe = LeaveOneOutEncoder(cols=cat_cols, random_state=42)
    X_train_enc = looe.fit_transform(X_train, y_train)
    X_val_enc = looe.transform(X_val)
    X_test_enc = looe.transform(X_test)
    
    # Scale all features (now fully numeric)
    final_scaler = StandardScaler()
    X_train_final = pd.DataFrame(final_scaler.fit_transform(X_train_enc), columns=X_train_enc.columns)
    X_val_final = pd.DataFrame(final_scaler.transform(X_val_enc), columns=X_val_enc.columns)
    X_test_final = pd.DataFrame(final_scaler.transform(X_test_enc), columns=X_test_enc.columns)
    
    logger.info("Applying standard SMOTE on clean, fully numeric data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)
    
    logger.info("Training XGBoost Classifier...")
    model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train_res, y_train_res)
    
    target_prec = 0.52
    logger.info(f"Tuning Threshold specifically for Precision >= {target_prec}...")
    thresh = tune_threshold_precision(model, X_val_final, y_val, target_precision=target_prec)
    logger.info(f"Selected Threshold: {thresh:.4f}")
    
    results = evaluate(model, X_test_final, y_test, thresh)
    
    logger.info("\n--- Phase 5 Results (Test Set) ---")
    df_results = pd.DataFrame([{'Strategy': 'J (K-Means Clustering + LOOE + XGBoost)', **results}])
    logger.info(f"\n{df_results.to_markdown(index=False)}")
    
    with open("clustering_model_results.md", "w") as f:
        f.write("# Phase 5 Unsupervised Clustering Results\n\n")
        f.write(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
