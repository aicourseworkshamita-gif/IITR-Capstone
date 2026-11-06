import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

def run_part_a():
    raw_df = pd.read_csv("data/raw_shipments.csv")
    hubs_df = pd.read_csv("data/hubs.csv")
    
    duplicates_count = raw_df.duplicated().sum()
    service_type_before = raw_df['service_type'].drop_duplicates().nunique()
    weight_non_sentinel = raw_df['parcel_weight_kg'][raw_df['parcel_weight_kg'] <= 100]
    missing_wt_before = weight_non_sentinel.isna().sum()
    missing_dist_before = raw_df['distance_km'].isna().sum()
    
    clean_df, medians = clean_shipments(raw_df)
    service_type_after = clean_df['service_type'].drop_duplicates().nunique()
    
    modelling_df = build_features(clean_df, hubs_df)
    modelling_df['is_delayed'] = (modelling_df['delivery_hours'] > modelling_df['promised_hours']).astype(int)
    modelling_df.to_csv("data/modelling_table.csv", index=False)
    
    metrics = [
        ("rows_raw", len(raw_df)),
        ("rows_clean", len(modelling_df)),
        ("duplicates_removed", duplicates_count),
        ("service_type_spellings_before", service_type_before),
        ("service_type_spellings_after", service_type_after),
        ("missing_distance_before", missing_dist_before),
        ("missing_parcel_weight_before", missing_wt_before),
        ("nulls_in_features_and_target", modelling_df[FEATURES + ['delivery_hours', 'is_delayed']].isna().sum().sum()),
        ("declared_value_mean_inr", round(modelling_df['declared_value_inr'].mean(), 4)),
        ("declared_value_median_inr", round(modelling_df['declared_value_inr'].median(), 4)),
        ("log_declared_value_mean", round(modelling_df['log_declared_value'].mean(), 4)),
        ("log_declared_value_median", round(modelling_df['log_declared_value'].median(), 4)),
    ]
    pd.DataFrame(metrics, columns=['metric', 'value']).to_csv("outputs/part_a_checks.csv", index=False)
    
    # Delay by group
    df_st = modelling_df.groupby('service_type').agg(shipments=('shipment_id', 'count'), delayed=('is_delayed', 'sum'), mean_hours=('delivery_hours', 'mean')).reset_index()
    df_st.insert(0, 'dimension', 'service_type')
    df_st = df_st.rename(columns={'service_type': 'group'})
    
    df_ht = modelling_df.groupby('hub_tier').agg(shipments=('shipment_id', 'count'), delayed=('is_delayed', 'sum'), mean_hours=('delivery_hours', 'mean')).reset_index()
    df_ht['group'] = 'Tier ' + df_ht['hub_tier'].astype(str)
    df_ht = df_ht.drop(columns=['hub_tier'])
    df_ht.insert(0, 'dimension', 'hub_tier')
    
    bins = [0, 200, 400, 600, 2000]
    labels = ['0-200', '201-400', '401-600', '601+']
    modelling_df['distance_band'] = pd.cut(modelling_df['distance_km'], bins=bins, labels=labels, right=True, include_lowest=True)
    df_db = modelling_df.groupby('distance_band', observed=False).agg(shipments=('shipment_id', 'count'), delayed=('is_delayed', 'sum'), mean_hours=('delivery_hours', 'mean')).reset_index()
    df_db.insert(0, 'dimension', 'distance_band')
    df_db = df_db.rename(columns={'distance_band': 'group'})
    
    delay_group_df = pd.concat([df_st, df_ht, df_db], ignore_index=True)
    delay_group_df['delay_rate'] = round(delay_group_df['delayed'] / delay_group_df['shipments'], 4)
    delay_group_df['mean_hours'] = round(delay_group_df['mean_hours'], 4)
    delay_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns
    modelling_df[numeric_cols].corr().round(4).to_csv("outputs/part_a_correlations.csv", index=True)
    print("Part A completed successfully.")

if __name__ == "__main__":
    run_part_a()
