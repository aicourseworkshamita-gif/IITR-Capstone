import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

def run_part_d():
    modelling_df = pd.read_csv("data/modelling_table.csv")
    hubs_df = pd.read_csv("data/hubs.csv")
    
    hub_profile = modelling_df.groupby('hub_id').agg(
        shipments=('shipment_id', 'count'),
        delayed=('is_delayed', 'sum'),
        mean_delivery_hours=('delivery_hours', 'mean'),
        mean_distance_km=('distance_km', 'mean'),
        mean_n_stops=('n_stops', 'mean')
    ).reset_index()
    hub_profile['delay_rate'] = hub_profile['delayed'] / hub_profile['shipments']
    
    hubs_full = hubs_df.merge(hub_profile, on='hub_id', how='left').sort_values('hub_id').reset_index(drop=True)
    hubs_full.to_csv("outputs/hub_profile.csv", index=False)
    
    features_for_clustering = ['hub_tier', 'hub_daily_capacity', 'delay_rate', 'mean_delivery_hours']
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hubs_full[features_for_clustering])
    
    separation_rows = []
    for K in [2, 3, 4, 5]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        
        dist_matrix = squareform(pdist(scaled_matrix, metric='euclidean'))
        same_cluster_dists, diff_cluster_dists = [], []
        
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    same_cluster_dists.append(dist_matrix[i, j])
                else:
                    diff_cluster_dists.append(dist_matrix[i, j])
                    
        across_mean = np.mean(diff_cluster_dists) if diff_cluster_dists else 0
        within_mean = np.mean(same_cluster_dists) if same_cluster_dists else 1e-6
        ratio = across_mean / within_mean
        
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).values
        sizes_str = "-".join(map(str, counts))
        separation_rows.append({"K": K, "across_over_within": round(ratio, 4), "cluster_sizes": sizes_str})
        
    pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)
    
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels3 = km3.fit_predict(scaled_matrix)
    hubs_full['cluster'] = labels3
    
    cluster_means = hubs_full.groupby('cluster')['delay_rate'].mean().sort_values()
    cluster_mapping = {}
    names = ['Standard', 'Watch', 'Escalate']
    for idx, c_id in enumerate(cluster_means.index):
        cluster_mapping[c_id] = names[idx]
        
    hubs_full['segment'] = hubs_full['cluster'].map(cluster_mapping)
    
    hub_segments = hubs_full[['hub_id', 'segment', 'hub_city', 'hub_tier']]
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)
    
    segment_profile = hubs_full.groupby('segment').agg(
        hubs=('hub_id', 'count'),
        mean_delivery_hours=('mean_delivery_hours', 'mean'),
        delay_rate=('delay_rate', 'mean'),
        mean_distance_km=('mean_distance_km', 'mean'),
        mean_n_stops=('mean_n_stops', 'mean')
    ).reset_index()
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)
    
    tier_map = {1: 'Standard', 2: 'Watch', 3: 'Escalate'}
    hubs_full['implied_segment'] = hubs_full['hub_tier'].map(tier_map)
    mismatches = hubs_full[hubs_full['segment'] != hubs_full['implied_segment']]
    mismatches[['hub_id', 'segment', 'hub_city', 'hub_tier']].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    print("Part D segments completed successfully.")

if __name__ == "__main__":
    run_part_d()
