import numpy as np
import pandas as pd

FEATURES = [
    'distance_km', 'parcel_weight_kg', 'n_stops', 'rain_flag', 'sale_event_flag',
    'promised_hours', 'is_cod', 'is_weekend', 'dispatch_hour', 'log_declared_value',
    'hub_tier', 'hub_daily_capacity', 'hub_encoded_region', 'hub_encoded_city'
]

def clean_shipments(raw_df, medians=None):
    df = raw_df.copy()
    df = df.drop_duplicates()
    df['service_type'] = df['service_type'].astype(str).str.strip().str.title()
    df['payment_mode'] = df['payment_mode'].fillna("Unknown")
    
    if 'declared_value' in df.columns:
        clean_val = df['declared_value'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df['declared_value_inr'] = pd.to_numeric(clean_val, errors='coerce')
        df = df.drop(columns=['declared_value'])
        
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
    df.loc[df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan
    
    if medians is None:
        medians = {
            'distance_km': float(df['distance_km'].median()),
            'parcel_weight_kg': float(df['parcel_weight_kg'].median()),
            'declared_value_inr': float(df['declared_value_inr'].median())
        }
        
    df['distance_km'] = df['distance_km'].fillna(medians['distance_km'])
    df['parcel_weight_kg'] = df['parcel_weight_kg'].fillna(medians['parcel_weight_kg'])
    df['declared_value_inr'] = df['declared_value_inr'].fillna(medians['declared_value_inr'])
    
    return df, medians

def build_features(clean_df, hubs_df):
    df = clean_df.merge(hubs_df, on='hub_id', how='left')
    if 'region' in df.columns and 'hub_encoded_region' not in df.columns:
        df['hub_encoded_region'] = df['region'].astype('category').cat.codes
    if 'hub_city' in df.columns and 'hub_encoded_city' not in df.columns:
        df['hub_encoded_city'] = df['hub_city'].astype('category').cat.codes
        
    promise_map = {'Express': 36, 'Standard': 60, 'Economy': 96}
    df['promised_hours'] = df['service_type'].map(promise_map).fillna(60).astype(int)
    df['is_cod'] = (df['payment_mode'] == 'COD').astype(int)
    df['is_weekend'] = df['dispatch_time'].dt.dayofweek.isin([5, 6]).astype(int)
    df['dispatch_hour'] = df['dispatch_time'].dt.hour.astype(int)
    df['log_declared_value'] = np.log(df['declared_value_inr'].clip(lower=1))
    
    return df
