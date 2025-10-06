import os
import pandas as pd
import numpy as np
import webbrowser
import folium
from folium.plugins import HeatMap
from xgboost import XGBRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from libInternal import getConnection, setFileLocation, cleanYear


conn = getConnection()
df_all = pd.read_sql("SELECT * FROM emmisions;", conn)
df_all = df_all.dropna(subset=['latitude', 'longitude', 'report_year'])
df_all['latitude'] = df_all['latitude'].astype(float)
df_all['longitude'] = df_all['longitude'].astype(float)
df_all['report_year'] = df_all['report_year'].apply(cleanYear)

# run dbscan
agg_scan = (
    df_all.groupby(['report_year', 'latitude', 'longitude'])
    .size()
    .reset_index(name='count')
)
agg_scan['year_num'] = agg_scan['report_year'].astype(int)
print(f"\aggregated data: {len(agg_scan)} row")

db_features = agg_scan[['latitude', 'longitude', 'count']]
scaler = StandardScaler()
db_scaled = scaler.fit_transform(db_features)
db_scan = DBSCAN(eps=0.5, min_samples=5)
agg_scan['dbscan_label'] = db_scan.fit_predict(db_scaled)
n_clusters = len(set(agg_scan['dbscan_label'])) - (1 if -1 in agg_scan['dbscan_label'] else 0)
n_noise = list(agg_scan['dbscan_label']).count(-1)
noise_pct = 100 * n_noise / len(agg_scan)
print(f"DBSCAN Report clusters data: {n_clusters}, noise point: {n_noise} ({noise_pct:.2f}%)")
agg_clean = agg_scan[agg_scan['dbscan_label'] != -1].copy()
print(f"forecasting data setelah hapus noise: {len(agg_clean)} row")