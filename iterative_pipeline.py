import pandas as pd
import numpy as np
import re
import warnings
import logging
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import ADASYN
from category_encoders import TargetEncoder
import optuna
import tabulate

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
    
    pt = PowerTransformer(method='yeo-johnson')
    X_train_val[numerical_cols] = pt.fit_transform(X_train_val[numerical_cols])
    X_train[numerical_cols] = pt.transform(X_train[numerical_cols])
    X_val[numerical_cols] = pt.transform(X_val[numerical_cols])
    X_test[numerical_cols] = pt.transform(X_test[numerical_cols])
    
    return X_train_val, X_train, X_val, X_test, categorical_cols, numerical_cols

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

def strategy_a(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols):
    logger.info("Running Strategy A: Class Weights + Validation F1 Tuning...")
    def prep(df): return clean_cols(pd.get_dummies(df, columns=categorical_cols, drop_first=True))
    X_train_enc = prep(X_train)
    X_val_enc = prep(X_val)
    X_test_enc = prep(X_test)
    X_val_enc, _ = X_val_enc.align(X_train_enc, join='right', axis=1, fill_value=0)
    X_test_enc, _ = X_test_enc.align(X_train_enc, join='right', axis=1, fill_value=0)
    
    scale_pos = sum(y_train == 0) / sum(y_train == 1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'scale_pos_weight': scale_pos,
            'verbose': -1, 'random_state': 42
        }
        model = LGBMClassifier(**params)
        model.fit(X_train_enc, y_train)
        probs = model.predict_proba(X_val_enc)[:, 1]
        pre, rec, thresh = precision_recall_curve(y_val, probs)
        f1s = 2*(pre*rec)/(pre+rec+1e-10)
        return np.max(f1s)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    
    best_params = study.best_params
    best_params.update({'scale_pos_weight': scale_pos, 'verbose': -1, 'random_state': 42})
    model = LGBMClassifier(**best_params)
    model.fit(X_train_enc, y_train)
    
    thresh = tune_threshold_f1(model, X_val_enc, y_val)
    return evaluate(model, X_test_enc, y_test, thresh)

def strategy_b(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols, numerical_cols):
    logger.info("Running Strategy B: Polynomial Interactions + LASSO...")
    def prep(df): return clean_cols(pd.get_dummies(df, columns=categorical_cols, drop_first=True))
    X_train_enc = prep(X_train)
    X_val_enc, _ = prep(X_val).align(X_train_enc, join='right', axis=1, fill_value=0)
    X_test_enc, _ = prep(X_test).align(X_train_enc, join='right', axis=1, fill_value=0)
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    def apply_poly(df):
        poly_features = poly.fit_transform(df[numerical_cols])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_cols), index=df.index)
        return clean_cols(pd.concat([df.drop(columns=numerical_cols), poly_df], axis=1))
        
    X_train_poly = apply_poly(X_train_enc)
    X_val_poly = apply_poly(X_val_enc)
    X_test_poly = apply_poly(X_test_enc)
    
    scaler = StandardScaler()
    X_train_poly = pd.DataFrame(scaler.fit_transform(X_train_poly), columns=X_train_poly.columns)
    X_val_poly = pd.DataFrame(scaler.transform(X_val_poly), columns=X_val_poly.columns)
    X_test_poly = pd.DataFrame(scaler.transform(X_test_poly), columns=X_test_poly.columns)
    
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_train_poly, y_train)
    selector = SelectFromModel(lasso, prefit=True)
    X_train_sel = selector.transform(X_train_poly)
    X_val_sel = selector.transform(X_val_poly)
    X_test_sel = selector.transform(X_test_poly)
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_sel, y_train)
    thresh = tune_threshold_f1(model, X_val_sel, y_val)
    return evaluate(model, X_test_sel, y_test, thresh)

def strategy_c(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols):
    logger.info("Running Strategy C: Target Encoding + ADASYN...")
    te = TargetEncoder(cols=categorical_cols)
    X_train_enc = clean_cols(te.fit_transform(X_train, y_train))
    X_val_enc = clean_cols(te.transform(X_val))
    X_test_enc = clean_cols(te.transform(X_test))
    
    ada = ADASYN(random_state=42)
    try:
        X_train_res, y_train_res = ada.fit_resample(X_train_enc, y_train)
    except Exception as e:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_enc, y_train)
        
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)
    thresh = tune_threshold_f1(model, X_val_enc, y_val)
    return evaluate(model, X_test_enc, y_test, thresh)

def strategy_d(X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols):
    logger.info("Running Strategy D: Soft Voting Meta-Ensemble...")
    def prep(df): return clean_cols(pd.get_dummies(df, columns=categorical_cols, drop_first=True))
    X_train_enc = prep(X_train)
    X_val_enc, _ = prep(X_val).align(X_train_enc, join='right', axis=1, fill_value=0)
    X_test_enc, _ = prep(X_test).align(X_train_enc, join='right', axis=1, fill_value=0)
    
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train_enc, y_train)
    
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf4 = LGBMClassifier(verbose=-1, random_state=42)
    clf5 = CatBoostClassifier(verbose=0, random_state=42)
    
    eclf = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('xgb', clf3), ('lgbm', clf4), ('cat', clf5)
    ], voting='soft')
    
    eclf.fit(X_train_res, y_train_res)
    thresh = tune_threshold_f1(eclf, X_val_enc, y_val)
    return evaluate(eclf, X_test_enc, y_test, thresh)

def main():
    logger.info("Starting Iterative Execution of 4 Strategies...")
    X_train_val, y_train_val, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()
    X_train_val, X_train, X_val, X_test, cat_cols, num_cols = get_base_preprocessed(X_train_val, X_train, X_val, X_test, y_train_val)
    
    res_a = strategy_a(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols)
    res_b = strategy_b(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, num_cols)
    res_c = strategy_c(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols)
    res_d = strategy_d(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols)
    
    results = [
        {'Strategy': 'A (Class Weights + F1 Tuning)', **res_a},
        {'Strategy': 'B (Polynomial + LASSO)', **res_b},
        {'Strategy': 'C (Target Enc + ADASYN)', **res_c},
        {'Strategy': 'D (Meta-Ensemble)', **res_d}
    ]
    
    df_results = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    logger.info(f"\n--- Iterative Strategy Leaderboard (Test Set) ---\n{df_results.to_markdown(index=False)}")
    
    with open("model_comparison.md", "w") as f:
        f.write("# Iterative F1 Optimization Leaderboard\n\n")
        f.write(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
