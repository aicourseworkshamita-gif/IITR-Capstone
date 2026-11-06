import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.shipment_prep import FEATURES

def run_part_c():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df['is_delayed']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=180/1200, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=180/1020, random_state=42, stratify=y_train_val)
    
    selector = SelectKBest(score_func=f_classif, k=7)
    selector.fit(X_train, y_train)
    scores_df = pd.DataFrame({'feature': FEATURES, 'score': selector.scores_, 'p_value': selector.pvalues_}).sort_values(by='score', ascending=False)
    scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    selected_features = list(scores_df['feature'].head(7))
    pd.DataFrame({'selected_features': selected_features}).to_csv("outputs/part_c_selectk.csv", index=False)
    
    X_train_k = X_train[selected_features]
    X_val_k = X_val[selected_features]
    X_test_k = X_test[selected_features]
    
    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }
    
    model_rows = []
    for name, m in models.items():
        m.fit(X_train_k, y_train)
        pred = m.predict(X_val_k)
        proba = m.predict_proba(X_val_k)[:, 1]
        model_rows.append({
            "model": name,
            "precision": round(precision_score(y_val, pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, pred, zero_division=0), 4),
            "f1": round(f1_score(y_val, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_val, proba), 4)
        })
    pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 10, None]}, cv=5, scoring='recall')
    rf_grid.fit(X_train_k, y_train)
    
    gb_grid = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5]}, n_iter=4, cv=5, random_state=42, scoring='recall')
    gb_grid.fit(X_train_k, y_train)
    
    tuning_rows = [
        {"model": "random_forest", "best_params": str(rf_grid.best_params_), "best_cv_recall": round(rf_grid.best_score_, 4)},
        {"model": "gradient_boosting", "best_params": str(gb_grid.best_params_), "best_cv_recall": round(gb_grid.best_score_, 4)}
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
    
    best_clf = rf_grid.best_estimator_
    val_probs = best_clf.predict_proba(X_val_k)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 99)
    thresh_rows = []
    valid_thresh = []
    for t in thresholds:
        p = (val_probs >= t).astype(int)
        rec = recall_score(y_val, p, zero_division=0)
        prec = precision_score(y_val, p, zero_division=0)
        f1 = f1_score(y_val, p, zero_division=0)
        thresh_rows.append({"threshold": round(t, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
        if rec >= 0.80:
            valid_thresh.append((t, prec, rec))
            
    pd.DataFrame(thresh_rows).to_csv("outputs/part_c_thresholds.csv", index=False)
    
    chosen_threshold = max(valid_thresh, key=lambda x: x[1])[0] if valid_thresh else 0.1
    pd.DataFrame([{"threshold": round(chosen_threshold, 4)}], index=[0]).to_csv("outputs/part_c_final_config.csv", index=False)
    
    test_probs = best_clf.predict_proba(X_test_k)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    pd.DataFrame([{
        "test_precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, test_preds, zero_division=0), 4),
        "test_f1": round(f1_score(y_test, test_preds, zero_division=0), 4),
        "test_roc_auc": round(roc_auc_score(y_test, test_probs), 4)
    }]).to_csv("outputs/part_c_final_test.csv", index=False)
    print("Part C completed successfully.")

if __name__ == "__main__":
    run_part_c()
