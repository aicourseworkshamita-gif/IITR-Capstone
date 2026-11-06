import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments

def run_packaging():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    
    raw_df = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw_df)
    pd.DataFrame(list(medians.items()), columns=['column', 'median']).to_csv("models/impute_values.csv", index=False)
    
    X = df[FEATURES]
    y = df['is_delayed']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=180/1200, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=180/1020, random_state=42, stratify=y_train_val)
    
    selector = SelectKBest(score_func=f_classif, k=7)
    selector.fit(X_train, y_train)
    selected_features = list(pd.DataFrame({'feature': FEATURES, 'score': selector.scores_}).sort_values(by='score', ascending=False)['feature'].head(7))
    
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_classif, k=7)),
        ('model', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train[FEATURES], y_train)
    
    joblib.dump(pipeline, "models/delay_model.pkl")
    pd.DataFrame({'feature': FEATURES}).to_csv("models/feature_order.csv", index=False)
    
    config = pd.read_csv("outputs/part_c_final_config.csv")
    threshold = float(config['threshold'].iloc[0])
    pd.DataFrame({'threshold': [threshold]}).to_csv("models/delay_threshold.csv", index=False)
    
    loaded_model = joblib.load("models/delay_model.pkl")
    test_probs_mem = pipeline.predict_proba(X_test[FEATURES])[:, 1]
    test_probs_load = loaded_model.predict_proba(X_test[FEATURES])[:, 1]
    max_diff = float(abs(test_probs_mem - test_probs_load).max())
    
    reload_check = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", max_diff),
        ("threshold", threshold)
    ]
    pd.DataFrame(reload_check, columns=['metric', 'value']).to_csv("outputs/part_d_reload_check.csv", index=False)
    print("Model packaging completed successfully.")

if __name__ == "__main__":
    run_packaging()
