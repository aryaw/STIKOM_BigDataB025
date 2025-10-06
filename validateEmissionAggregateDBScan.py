import os
import pandas as pd
import numpy as np
import webbrowser
import folium
from folium.plugins import HeatMap
from xgboost import XGBRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from libInternal import getConnection, setFileLocation, cleanYear

fileTimeStamp, output_dir = setFileLocation()
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
print(f"\n aggregated data: {len(agg_scan)} row")

db_features = agg_scan[['latitude', 'longitude', 'count']]
scaler = StandardScaler()
db_scaled = scaler.fit_transform(db_features)
db_scan = DBSCAN(eps=0.5, min_samples=5)
agg_scan['dbscan_label'] = db_scan.fit_predict(db_scaled)
n_clusters = len(set(agg_scan['dbscan_label'])) - (1 if -1 in agg_scan['dbscan_label'] else 0)
n_noise = list(agg_scan['dbscan_label']).count(-1)
noise_pct = 100 * n_noise / len(agg_scan)
print(f"\n DBSCAN Report clusters data: {n_clusters}, noise point: {n_noise} ({noise_pct:.2f}%)")

cluster_summary = (
    agg_scan.groupby('dbscan_label')
    .size()
    .reset_index(name='total_records')
    .sort_values(by='total_records', ascending=False)
)
print("\n DBSCAN cluster summ:")
print(cluster_summary)

# cek cluster
for cluster_id in sorted(agg_scan['dbscan_label'].unique()):
    cluster_data = agg_scan[agg_scan['dbscan_label'] == cluster_id]
    label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"\n{label} → Total records: {len(cluster_data)}")
    print(cluster_data.head())


agg_clean = agg_scan[agg_scan['dbscan_label'] != -1].copy()
print(f"data setelah hapus noise: {len(agg_clean)} row")

agg_scan['cluster_label'] = agg_scan['dbscan_label'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

fig = px.scatter_3d(
    agg_scan,
    x='longitude',
    y='latitude',
    z='count',
    color='cluster_label',
    hover_data=['report_year', 'count'],
    title=f"DBSCAN Clusters ({n_clusters} clusters + noise)",
    width=900, height=700
)

file_html = os.path.join(output_dir, f"dbscan_clusters_{fileTimeStamp}.html")
fig.write_html(file_html)
print(f"\n✅ DBSCAN cluster visualization saved → {file_html}\n")
webbrowser.open(file_html)