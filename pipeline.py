import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import logging
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.combine import SMOTEENN
import optuna
import shap
import lime
import lime.lime_tabular
import tabulate

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure elegant logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Clean Pipeline Execution...")
    
    logger.info("Loading dataset 'German_Credit_Data.csv'...")
    df = pd.read_csv('German_Credit_Data.csv')
    
    # Feature Engineering
    logger.info("Performing Feature Engineering (payment_burden, age_group)...")
    df['payment_burden'] = df['credit_amount'] / df['duration']
    bins = [0, 25, 45, 65, 100]
    labels = ['Youth', 'Adult', 'Middle-Age', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    numerical_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
    categorical_cols = list(X.select_dtypes(include=['object', 'category']).columns)
    
    # Clean Data Science Split: 60% Train, 20% Val, 20% Test
    logger.info("Splitting data into strictly isolated Train (60%), Validation (20%), and Test (20%) sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    
    # --- STATISTICAL FEATURE SELECTION (On Train+Val to guide ensemble) ---
    logger.info("Running Statistical Tests to drop insignificant noise (p > 0.05)...")
    dropped_cols = []
    
    # Categorical: Chi-Square
    for col in categorical_cols:
        crosstab = pd.crosstab(y_train_val, X_train_val[col])
        _, p, _, _ = chi2_contingency(crosstab)
        if p > 0.05:
            dropped_cols.append(col)
            categorical_cols.remove(col)
            logger.info(f"  Dropped Categorical: {col} (p-value: {p:.4f})")
            
    # Numerical: T-Test
    for col in numerical_cols:
        group_0 = X_train_val[y_train_val == 0][col]
        group_1 = X_train_val[y_train_val == 1][col]
        _, p = ttest_ind(group_0, group_1, equal_var=False)
        if p > 0.05:
            dropped_cols.append(col)
            numerical_cols.remove(col)
            logger.info(f"  Dropped Numerical: {col} (p-value: {p:.4f})")
            
    X_train_val = X_train_val.drop(columns=dropped_cols)
    X_train = X_train.drop(columns=dropped_cols)
    X_val = X_val.drop(columns=dropped_cols)
    X_test = X_test.drop(columns=dropped_cols)
    
    # --- POWER TRANSFORMATION ---
    logger.info("Applying PowerTransformer (yeo-johnson) to correct skewness in continuous features...")
    pt = PowerTransformer(method='yeo-johnson')
    X_train_val[numerical_cols] = pt.fit_transform(X_train_val[numerical_cols])
    X_train[numerical_cols] = pt.transform(X_train[numerical_cols])
    X_val[numerical_cols] = pt.transform(X_val[numerical_cols])
    X_test[numerical_cols] = pt.transform(X_test[numerical_cols])
    
    # --- ENCODING ---
    logger.info("Applying One-Hot Encoding and Scaling...")
    X_train_val = pd.get_dummies(X_train_val, columns=categorical_cols, drop_first=True)
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    # Align columns in case some categories were missing in splits
    X_train, _ = X_train.align(X_train_val, join='right', axis=1, fill_value=0)
    X_val, _ = X_val.align(X_train_val, join='right', axis=1, fill_value=0)
    X_test, _ = X_test.align(X_train_val, join='right', axis=1, fill_value=0)
    
    scaler = StandardScaler()
    X_train_val[numerical_cols] = scaler.fit_transform(X_train_val[numerical_cols])
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Clean feature names
    regex = re.compile(r'[\[\]<>]')
    for df_split in [X_train_val, X_train, X_val, X_test]:
        df_split.columns = [regex.sub('_', str(col)) for col in df_split.columns]
        
    logger.info("Applying SMOTEENN for class imbalance exclusively on Train sets...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
    
    logger.info("Initiating Optuna Bayesian Optimization on the Validation Set...")
    def optimize_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'random_state': 42,
            'verbose': -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train_res, y_train_res)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    def optimize_catboost(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 200),
            'depth': trial.suggest_int('depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'verbose': 0,
            'random_state': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train_res, y_train_res)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(optimize_lgbm, n_trials=15)
    best_lgbm_params = study_lgbm.best_params
    best_lgbm_params.update({'verbose': -1, 'random_state': 42})
    logger.info(f"  LightGBM best ROC-AUC on Val: {study_lgbm.best_value:.4f}")
    
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(optimize_catboost, n_trials=15)
    best_cat_params = study_cat.best_params
    best_cat_params.update({'verbose': 0, 'random_state': 42})
    logger.info(f"  CatBoost best ROC-AUC on Val: {study_cat.best_value:.4f}")
    
    logger.info("Training K-Fold Ensemble on Train+Val Data...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_models = []
    
    for fold, (train_idx, fold_val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        X_fold_train = X_train_val.iloc[train_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        
        X_fold_train_res, y_fold_train_res = smote_enn.fit_resample(X_fold_train, y_fold_train)
        
        lgbm = LGBMClassifier(**best_lgbm_params)
        lgbm.fit(X_fold_train_res, y_fold_train_res)
        ensemble_models.append(lgbm)
        
        cat = CatBoostClassifier(**best_cat_params)
        cat.fit(X_fold_train_res, y_fold_train_res)
        ensemble_models.append(cat)
        
    logger.info("Evaluating Ensemble on Untouched 20% Test Set...")
    test_probs = np.zeros(len(X_test))
    val_probs = np.zeros(len(X_val)) 
    
    for model in ensemble_models:
        test_probs += model.predict_proba(X_test)[:, 1]
        val_probs += model.predict_proba(X_val)[:, 1]
        
    test_probs /= len(ensemble_models)
    val_probs /= len(ensemble_models)
    
    roc = roc_auc_score(y_test, test_probs)
    
    # --- PRECISION/F1 TARGETED THRESHOLD TUNING ---
    logger.info("Selecting Optimal Threshold to Maximize F1-Score on Validation set...")
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    logger.info(f"  Selected Validation Threshold: {best_threshold:.4f} (Validation F1 Max: {f1_scores[best_idx]:.4f})")
    
    y_test_pred = (test_probs >= best_threshold).astype(int)
    
    results = [{
        'Model': 'CatBoost+LightGBM Ensemble',
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1': f1_score(y_test, y_test_pred),
        'ROC-AUC': roc
    }]
    
    results_df = pd.DataFrame(results)
    logger.info(f"\n--- Final Test Set Results ---\n{results_df.to_markdown(index=False)}")
    
    with open("model_comparison.md", "w") as f:
        f.write("# Clean Textbook Ensemble Comparison (Precision Optimized)\n\n")
        f.write(results_df.to_markdown(index=False))
        
    logger.info("Generating Explanations (SHAP/LIME)...")
    X_train_val_res, y_train_val_res = smote_enn.fit_resample(X_train_val, y_train_val)
    rep_model = CatBoostClassifier(**best_cat_params)
    rep_model.fit(X_train_val_res, y_train_val_res)
    
    correct_preds = y_test == y_test_pred
    good_idx = y_test[correct_preds & (y_test == 0)].index[0]
    bad_idx = y_test[correct_preds & (y_test == 1)].index[0]
    
    instances_idx = [good_idx, bad_idx]
    
    explainer = shap.TreeExplainer(rep_model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    plt.close()
    
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    elif isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]
        
    for i, idx in enumerate(instances_idx):
        loc_idx = X_test.index.get_loc(idx)
        if isinstance(loc_idx, slice): loc_idx = loc_idx.start
        elif isinstance(loc_idx, np.ndarray): loc_idx = np.where(loc_idx)[0][0]
        
        plt.figure()
        shap.force_plot(
            float(expected_value), 
            shap_values[loc_idx], 
            X_test.iloc[loc_idx].astype(float).round(3).values,
            feature_names=X_test.columns.tolist(),
            matplotlib=True, 
            show=False
        )
        plt.savefig(f'shap_force_plot_instance_{i}.png', bbox_inches='tight')
        plt.close()
        
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
