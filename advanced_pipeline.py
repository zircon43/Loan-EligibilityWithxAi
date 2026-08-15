import pandas as pd
import numpy as np
import re
import warnings
import logging
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import BorderlineSMOTE
from category_encoders import WOEEncoder
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def clean_cols(df):
    regex = re.compile(r'[\[\]<>]')
    df.columns = [regex.sub('_', str(col)) for col in df.columns]
    return df

def load_and_split():
    df = pd.read_csv('German_Credit_Data.csv')
    df['payment_burden'] = df['credit_amount'] / df['duration']
    bins = [0, 25, 45, 65, 100]
    labels = ['Youth', 'Adult', 'Middle-Age', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    return X_train_val, y_train_val, X_train, y_train, X_val, y_val, X_test, y_test

def get_base_preprocessed(X_train_val, X_train, X_val, X_test, y_train_val):
    categorical_cols = list(X_train_val.select_dtypes(include=['object', 'category']).columns)
    numerical_cols = list(X_train_val.select_dtypes(include=['int64', 'float64']).columns)
    
    dropped_cols = []
    for col in categorical_cols:
        crosstab = pd.crosstab(y_train_val, X_train_val[col])
        _, p, _, _ = chi2_contingency(crosstab)
        if p > 0.05: dropped_cols.append(col); categorical_cols.remove(col)
            
    for col in numerical_cols:
        group_0 = X_train_val[y_train_val == 0][col]
        group_1 = X_train_val[y_train_val == 1][col]
        _, p = ttest_ind(group_0, group_1, equal_var=False)
        if p > 0.05: dropped_cols.append(col); numerical_cols.remove(col)

    def prep(df): return df.drop(columns=dropped_cols)
    X_train_val, X_train, X_val, X_test = map(prep, [X_train_val, X_train, X_val, X_test])
    
    return X_train, X_val, X_test, categorical_cols, numerical_cols

def tune_threshold_f1(model, X_val, y_val):
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
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

def strategy_e(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols, numerical_cols):
    logger.info("Running Strategy E: Isolation Forest + BorderlineSMOTE...")
    def prep(df): return clean_cols(pd.get_dummies(df, columns=categorical_cols, drop_first=True))
    X_train_enc = prep(X_train)
    X_val_enc, _ = prep(X_val).align(X_train_enc, join='right', axis=1, fill_value=0)
    X_test_enc, _ = prep(X_test).align(X_train_enc, join='right', axis=1, fill_value=0)
    
    # Standard scaling
    scaler = StandardScaler()
    X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
    X_val_enc[numerical_cols] = scaler.transform(X_val_enc[numerical_cols])
    X_test_enc[numerical_cols] = scaler.transform(X_test_enc[numerical_cols])
    
    # 1. Outlier Removal on Train Set
    iso = IsolationForest(contamination=0.05, random_state=42)
    yhat = iso.fit_predict(X_train_enc)
    mask = yhat != -1
    X_train_clean, y_train_clean = X_train_enc[mask], y_train[mask]
    
    # 2. BorderlineSMOTE
    bsmote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    try:
        X_train_res, y_train_res = bsmote.fit_resample(X_train_clean, y_train_clean)
    except Exception as e:
        logger.warning(f"BorderlineSMOTE failed ({e}), falling back to standard SMOTE.")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train_clean)
        
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, max_depth=3, learning_rate=0.05)
    model.fit(X_train_res, y_train_res)
    
    thresh = tune_threshold_f1(model, X_val_enc, y_val)
    return evaluate(model, X_test_enc, y_test, thresh)

def strategy_f(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols, numerical_cols):
    logger.info("Running Strategy F: Weight of Evidence (WoE) Encoding + Logistic Regression...")
    X_train_woe = X_train.copy()
    X_val_woe = X_val.copy()
    X_test_woe = X_test.copy()
    
    # Discretize continuous variables into bins (WoE requires discrete/categorical)
    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    X_train_woe[numerical_cols] = est.fit_transform(X_train_woe[numerical_cols])
    X_val_woe[numerical_cols] = est.transform(X_val_woe[numerical_cols])
    X_test_woe[numerical_cols] = est.transform(X_test_woe[numerical_cols])
    
    # Convert numerical bins to string so WoE encoder treats them as categories
    X_train_woe[numerical_cols] = X_train_woe[numerical_cols].astype(str)
    X_val_woe[numerical_cols] = X_val_woe[numerical_cols].astype(str)
    X_test_woe[numerical_cols] = X_test_woe[numerical_cols].astype(str)
    
    all_features = categorical_cols + numerical_cols
    woe = WOEEncoder(cols=all_features)
    X_train_enc = woe.fit_transform(X_train_woe, y_train)
    X_val_enc = woe.transform(X_val_woe)
    X_test_enc = woe.transform(X_test_woe)
    
    # Scale WoE features
    scaler = StandardScaler()
    X_train_enc = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=X_train_enc.columns)
    X_val_enc = pd.DataFrame(scaler.transform(X_val_enc), columns=X_val_enc.columns)
    X_test_enc = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns)
    
    # Highly regularized logistic regression because WoE linearly perfectly encodes risks
    model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.5, random_state=42)
    model.fit(X_train_enc, y_train)
    
    thresh = tune_threshold_f1(model, X_val_enc, y_val)
    return evaluate(model, X_test_enc, y_test, thresh)

def strategy_g(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols, numerical_cols):
    logger.info("Running Strategy G: Stacking Meta-Model...")
    def prep(df): return clean_cols(pd.get_dummies(df, columns=categorical_cols, drop_first=True))
    X_train_enc = prep(X_train)
    X_val_enc, _ = prep(X_val).align(X_train_enc, join='right', axis=1, fill_value=0)
    X_test_enc, _ = prep(X_test).align(X_train_enc, join='right', axis=1, fill_value=0)
    
    # Transform numerics with PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    X_train_enc[numerical_cols] = pt.fit_transform(X_train_enc[numerical_cols])
    X_val_enc[numerical_cols] = pt.transform(X_val_enc[numerical_cols])
    X_test_enc[numerical_cols] = pt.transform(X_test_enc[numerical_cols])
    
    scaler = StandardScaler()
    X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
    X_val_enc[numerical_cols] = scaler.transform(X_val_enc[numerical_cols])
    X_test_enc[numerical_cols] = scaler.transform(X_test_enc[numerical_cols])
    
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)
    
    # Define Base Estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)),
        ('lgbm', LGBMClassifier(verbose=-1, random_state=42, max_depth=3)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42, depth=3))
    ]
    
    # Meta Model learns from predictions of Base Estimators
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    clf.fit(X_train_res, y_train_res)
    thresh = tune_threshold_f1(clf, X_val_enc, y_val)
    return evaluate(clf, X_test_enc, y_test, thresh)

def main():
    logger.info("Starting Advanced Phase 3 Execution...")
    X_train_val, y_train_val, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()
    X_train, X_val, X_test, cat_cols, num_cols = get_base_preprocessed(X_train_val, X_train, X_val, X_test, y_train_val)
    
    res_e = strategy_e(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, num_cols)
    res_f = strategy_f(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, num_cols)
    res_g = strategy_g(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, num_cols)
    
    results = [
        {'Strategy': 'E (Outlier Removal + BorderlineSMOTE)', **res_e},
        {'Strategy': 'F (Weight of Evidence + LR)', **res_f},
        {'Strategy': 'G (Stacking Meta-Model)', **res_g}
    ]
    
    df_results = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    logger.info(f"\n--- Advanced Strategy Leaderboard (Test Set) ---\n{df_results.to_markdown(index=False)}")
    
    with open("advanced_model_comparison.md", "w") as f:
        f.write("# Advanced Phase 3 F1 Optimization Leaderboard\n\n")
        f.write(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
