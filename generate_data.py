"""Synthetic shipment data for the Delay Intelligence capstone.
Run once: python generate_data.py (writes data/hubs.csv,
data/raw_shipments.csv, data/new_shipments.csv)
Do not edit any constant: every number in the brief was measured on this
exact output."""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

# 1. Hubs
n_hubs = 24
hub_ids = [f"HUB{i:02d}" for i in range(1, n_hubs + 1)]
cities = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai",
    "Kolkata", "Surat", "Pune", "Jaipur", "Lucknow", "Kanpur",
    "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna",
    "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad"
]
regions = np.random.choice(["North", "South", "East", "West", "Central"], size=n_hubs)
hub_tiers = np.random.choice([1, 2, 3], size=n_hubs, p=[0.3, 0.5, 0.2])
hub_capacities = np.random.randint(5000, 25000, size=n_hubs)

hubs_df = pd.DataFrame({
    "hub_id": hub_ids,
    "hub_city": cities,
    "region": regions,
    "hub_tier": hub_tiers,
    "hub_daily_capacity": hub_capacities
})
os.makedirs("data", exist_ok=True)
hubs_df.to_csv("data/hubs.csv", index=False)

# 2. Raw Shipments (Historical)
n_shipments = 1200
shipment_ids = [f"SHP{i:05d}" for i in range(1, n_shipments + 1)]
hub_assigned = np.random.choice(hub_ids, size=n_shipments)
service_types = np.random.choice(["Express", "Standard", "Economy", " Express ", "Standard  ", "Economy "], size=n_shipments, p=[0.3, 0.4, 0.2, 0.03, 0.04, 0.03])
payment_modes = np.random.choice(["COD", "Prepaid", "UPI", np.nan], size=n_shipments, p=[0.5, 0.3, 0.15, 0.05])
distances = np.random.exponential(scale=300, size=n_shipments) + 50
weights = np.random.exponential(scale=5, size=n_shipments) + 0.5
fault_mask = np.random.rand(n_shipments) < 0.03
weights[fault_mask] = np.random.uniform(105, 250, size=fault_mask.sum())

declared_values = np.random.exponential(scale=2000, size=n_shipments) + 500
declared_str = []
for v in declared_values:
    if np.random.rand() < 0.4:
        declared_str.append(f"Rs. {v:,.2f}")
    elif np.random.rand() < 0.7:
        declared_str.append(f"₹{v:,.1f}")
    else:
        declared_str.append(str(round(v, 2)))

start_date = pd.Timestamp("2026-09-01")
dispatch_times = [start_date + pd.Timedelta(hours=int(np.random.randint(0, 24*25)), minutes=int(np.random.randint(0, 60))) for _ in range(n_shipments)]

raw_df = pd.DataFrame({
    "shipment_id": shipment_ids,
    "hub_id": hub_assigned,
    "service_type": service_types,
    "payment_mode": payment_modes,
    "declared_value": declared_str,
    "dispatch_time": dispatch_times,
    "parcel_weight_kg": weights,
    "distance_km": distances,
    "n_stops": np.random.randint(1, 6, size=n_shipments),
    "rain_flag": np.random.choice([0, 1], size=n_shipments, p=[0.85, 0.15]),
    "sale_event_flag": np.random.choice([0, 1], size=n_shipments, p=[0.8, 0.2]),
})

dups = raw_df.sample(n=24, random_state=42)
raw_df = pd.concat([raw_df, dups], ignore_index=True)

missing_dist = np.random.choice(raw_df.index, size=30, replace=False)
raw_df.loc[missing_dist, "distance_km"] = np.nan
missing_wt = np.random.choice(raw_df.index, size=25, replace=False)
raw_df.loc[missing_wt, "parcel_weight_kg"] = np.nan

promise_map = {"Express": 36, "Standard": 60, "Economy": 96}
st_clean = raw_df["service_type"].astype(str).str.strip().str.title()
promised = st_clean.map(promise_map).fillna(60)

base_hours = promised * 0.7 + raw_df["distance_km"].fillna(300) * 0.04 + raw_df["parcel_weight_kg"].fillna(5) * 0.5 + np.random.normal(0, 8, size=len(raw_df))
delivery_hours = np.maximum(12, base_hours)
raw_df["delivery_hours"] = delivery_hours

raw_df.to_csv("data/raw_shipments.csv", index=False)

n_new = 48
new_ids = [f"NEW{i:05d}" for i in range(1, n_new + 1)]
new_hub = np.random.choice(hub_ids, size=n_new)
new_service = np.random.choice(["Express", "Standard", "Economy"], size=n_new, p=[0.3, 0.4, 0.3])
new_payment = np.random.choice(["COD", "Prepaid", "UPI", np.nan], size=n_new, p=[0.5, 0.3, 0.15, 0.05])
new_dist = np.random.exponential(scale=300, size=n_new) + 50
new_wt = np.random.exponential(scale=5, size=n_new) + 0.5
new_val = np.random.exponential(scale=2000, size=n_new) + 500
new_val_str = [f"Rs. {v:,.2f}" for v in new_val]
new_dispatch = [pd.Timestamp("2026-10-01") + pd.Timedelta(hours=int(np.random.randint(0, 48))) for _ in range(n_new)]

new_df = pd.DataFrame({
    "shipment_id": new_ids,
    "hub_id": new_hub,
    "service_type": new_service,
    "payment_mode": new_payment,
    "declared_value": new_val_str,
    "dispatch_time": new_dispatch,
    "parcel_weight_kg": new_wt,
    "distance_km": new_dist,
    "n_stops": np.random.randint(1, 6, size=n_new),
    "rain_flag": np.random.choice([0, 1], size=n_new, p=[0.85, 0.15]),
    "sale_event_flag": np.random.choice([0, 1], size=n_new, p=[0.8, 0.2]),
})
new_df.to_csv("data/new_shipments.csv", index=False)

print(f"{n_hubs} {len(raw_df)} {n_new}")
