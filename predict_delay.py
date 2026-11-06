import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

def run_predictions():
    new_shipments = pd.read_csv("data/new_shipments.csv")
    hubs_df = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")
    
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df['column'], impute_df['median']))
    
    feature_order_df = pd.read_csv("models/feature_order.csv")
    feature_cols = list(feature_order_df['feature'])
    
    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df['threshold'].iloc[0])
    
    clean_new, _ = clean_shipments(new_shipments, medians=medians)
    featured_new = build_features(clean_new, hubs_df)
    
    X_new = featured_new[feature_cols]
    model = joblib.load("models/delay_model.pkl")
    
    probs = model.predict_proba(X_new)[:, 1]
    featured_new['delay_probability'] = [round(p, 4) for p in probs]
    featured_new['flagged'] = (featured_new['delay_probability'] >= threshold).astype(int)
    
    merged = featured_new.merge(hub_segments[['hub_id', 'segment']], on='hub_id', how='left')
    
    actions = []
    for _, row in merged.iterrows():
        if row['flagged'] == 1:
            if row['segment'] == 'Escalate':
                act = "Escalate to hub manager"
            else:
                act = "Proactive customer call"
        else:
            act = "No action"
        actions.append(act)
        
    merged['action'] = actions
    
    output_df = merged[['shipment_id', 'hub_id', 'segment', 'delay_probability', 'flagged', 'action']]
    os.makedirs("outputs", exist_ok=True)
    output_df.to_csv("outputs/new_shipment_actions.csv", index=False)
    print("New shipment predictions completed successfully.")

if __name__ == "__main__":
    run_predictions()
