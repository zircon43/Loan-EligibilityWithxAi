import pandas as pd
import numpy as np
import re
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

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

def mRMR_feature_selection(X_train, y_train, correlation_threshold=0.7, top_k=15):
    # 1. Relevance (Mutual Information)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    
    # Select top initial candidates based purely on relevance
    candidates = mi_series.head(top_k).index.tolist()
    logger.info(f"Top {top_k} Candidates by Mutual Information: {candidates}")
    
    # 2. Redundancy (Correlation Drop)
    corr_matrix = X_train[candidates].corr().abs()
    
    features_to_keep = []
    for feature in candidates:
        is_redundant = False
        for kept_feature in features_to_keep:
            if corr_matrix.loc[feature, kept_feature] > correlation_threshold:
                is_redundant = True
                logger.info(f"Dropping {feature} because it is highly correlated ({corr_matrix.loc[feature, kept_feature]:.2f}) with {kept_feature}")
                break
        if not is_redundant:
            features_to_keep.append(feature)
            
    logger.info(f"Final Selected Features after dropping Redundancy: {features_to_keep}")
    return features_to_keep

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
    logger.info("Starting Phase 6: The True Tree Lord Pipeline...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Decision Trees prefer ordinal variables to high-cardinality OHE
    logger.info("Applying Ordinal Encoding for pure Decision Tree splits...")
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = oe.transform(X_val[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])
    
    logger.info("Applying mRMR Feature Selection...")
    selected_features = mRMR_feature_selection(X_train, y_train, correlation_threshold=0.7, top_k=15)
    
    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]
    
    # No SMOTE. We want the True Tree to learn the raw, unmanipulated real-world probabilities.
    # Class weights will handle the imbalance perfectly inside the tree.
    
    logger.info("Extracting Cost-Complexity Pruning (CCP) Path on Training Set...")
    base_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    path = base_tree.cost_complexity_pruning_path(X_train_sel, y_train)
    ccp_alphas = path.ccp_alphas
    
    logger.info(f"Testing {len(ccp_alphas)} possible pruning thresholds on Validation Set...")
    best_alpha = 0
    best_f1 = -1
    
    for alpha in ccp_alphas:
        # Don't prune the whole tree away
        if alpha < 0: continue
            
        tree = DecisionTreeClassifier(random_state=42, class_weight='balanced', ccp_alpha=alpha)
        tree.fit(X_train_sel, y_train)
        
        # We can just use default 0.5 threshold for this internal search because we want the tree's natural structure
        val_preds = tree.predict(X_val_sel)
        f1 = f1_score(y_val, val_preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            
    logger.info(f"Optimal Pruning Alpha found: {best_alpha:.6f} (Validation F1: {best_f1:.4f})")
    
    logger.info("Training The True Tree Lord with optimal alpha...")
    final_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced', ccp_alpha=best_alpha)
    final_tree.fit(X_train_sel, y_train)
    
    logger.info("Tuning Final Threshold for Precision Constraint...")
    thresh = tune_threshold_precision(final_tree, X_val_sel, y_val, target_precision=0.52)
    logger.info(f"Selected Threshold: {thresh:.4f}")
    
    results = evaluate(final_tree, X_test_sel, y_test, thresh)
    
    logger.info(f"True Tree Lord Statistics:")
    logger.info(f"Tree Depth: {final_tree.get_depth()}")
    logger.info(f"Number of Leaves: {final_tree.get_n_leaves()}")
    
    logger.info("\n--- Phase 6 Results (Test Set) ---")
    df_results = pd.DataFrame([{'Strategy': 'K (True Tree Lord: mRMR + Pruned Tree)', **results}])
    logger.info(f"\n{df_results.to_markdown(index=False)}")
    
    with open("true_tree_results.md", "w") as f:
        f.write("# Phase 6 True Tree Lord Results\n\n")
        f.write(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
