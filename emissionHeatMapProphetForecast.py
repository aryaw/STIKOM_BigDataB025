import os
import pandas as pd
import webbrowser
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from libInternal import variableDump, getConnection, setFileLocation, cleanYear

# set var
future_years = 15
forecast_target_year = 2031
forecast_data = []

fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
df_forecast = pd.read_sql("SELECT * FROM emmisions;", conn)
df_forecast['latitude'] = df_forecast['latitude'].astype(float)
df_forecast['longitude'] = df_forecast['longitude'].astype(float)
df_forecast['report_year'] = df_forecast['report_year'].apply(cleanYear)

# group data frame
agg_scan = (
    df_forecast.groupby(['report_year', 'latitude', 'longitude'])
    .size()
    .reset_index(name='count')
)
# set field
db_features = agg_scan[['latitude', 'longitude', 'count']]
scaler = StandardScaler()
db_scaled = scaler.fit_transform(db_features)

# apply db scan
db = DBSCAN(eps=0.5, min_samples=50)
# fit the data
agg_scan['dbscan_label'] = db.fit_predict(db_scaled)

n_clusters = len(set(agg_scan['dbscan_label'])) - (1 if -1 in agg_scan['dbscan_label'] else 0)
n_noise = (agg_scan['dbscan_label'] == -1).sum()
noise_pct = 100 * n_noise / len(agg_scan)
print(f"DBSCAN {n_clusters} cluster, {n_noise} noise ({noise_pct:.2f}%)")

# create summary
cluster_summary = (
    agg_scan.groupby('dbscan_label')
    .size()
    .reset_index(name='total_records')
    .sort_values(by='total_records', ascending=False)
)
print("\n cluster summary:")
print(cluster_summary)

# hapus noise
agg_clean = agg_scan[agg_scan['dbscan_label'] != -1].copy()
print(f" data bersih tanpa noise: {len(agg_clean)} row")

# load data to plotly
# fig_cluster = px.scatter_mapbox(
#     agg_clean,
#     lat='latitude',
#     lon='longitude',
#     color='dbscan_label',
#     size='count',
#     hover_name='report_year',
#     title=f"cluster map ({n_clusters} clusters, {n_noise} noise)",
#     mapbox_style='carto-positron',
#     zoom=4,
#     height=600
# )

# ganti ke 3d scatterplot
fig_cluster = px.scatter_3d(
    agg_clean,
    x='longitude',
    y='latitude',
    z='count',
    color='dbscan_label',
    hover_data=['report_year', 'count'],
    title=f"clusters ({n_clusters} clusters, {n_noise} noise)",
    width=1200, height=1000
)
fileClusterHTML = os.path.join(output_dir, f"dbscan_cluster_report_{fileTimeStamp}.html")
fig_cluster.write_html(fileClusterHTML)
print(f" Cluster report save → {fileClusterHTML}")
webbrowser.open(fileClusterHTML)

# set forecast
for (lat, lon), group in df_forecast.groupby(['latitude','longitude']):
    if len(group['report_year'].unique()) >= 4:

        # count record per year
        ts = group.groupby("report_year").size().reset_index(name="count")
        ts = ts.rename(columns={"report_year":"ds", "count":"y"})
        ts['ds'] = pd.to_datetime(ts['ds'], format='%Y')
        
        # fit model
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model.fit(ts)

        # buat future frame
        future = model.make_future_dataframe(periods=future_years, freq='YE')
        forecast = model.predict(future)

        # prediksi untuk target year
        forecast['year'] = forecast['ds'].dt.year
        pred_val = forecast.loc[forecast['year']==forecast_target_year, 'yhat']
        
        if not pred_val.empty:
            y_pred = float(pred_val.values[0])
            forecast_data.append([lat, lon, max(y_pred,0)])  # no negative
            

heatMapForecast = folium.Map(location=[-26.853388, 133.275154], zoom_start=5)

HeatMap(
    forecast_data,
    min_opacity=0.3,
    radius=12,
    blur=18,
    max_zoom=1,
).add_to(heatMapForecast)

fileMapName = os.path.join(output_dir, f"forecastHeatmap_{forecast_target_year}_{fileTimeStamp}.html")
heatMapForecast.save(fileMapName)

print(f"\n Forecast Heatmap {forecast_target_year} → {fileMapName}\n")
webbrowser.open(fileMapName)