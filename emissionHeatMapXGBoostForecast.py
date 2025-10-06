import os
import pandas as pd
import numpy as np
import webbrowser
import folium
from folium.plugins import HeatMap
from xgboost import XGBRegressor
from libInternal import getConnection, setFileLocation, cleanYear


forecast_target_year = 2031
print(f"\n✅ Forecast target year → {forecast_target_year}")

fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()

df_all = pd.read_sql("SELECT * FROM emmisions;", conn)
df_all = df_all.dropna(subset=['latitude', 'longitude', 'report_year'])
df_all['latitude'] = df_all['latitude'].astype(float)
df_all['longitude'] = df_all['longitude'].astype(float)
df_all['report_year'] = df_all['report_year'].apply(cleanYear)

agg = (
    df_all.groupby(['report_year', 'latitude', 'longitude'])
    .size()
    .reset_index(name='count')
)
agg['year_num'] = agg['report_year'].astype(int)

X = agg[['year_num', 'latitude', 'longitude']]
y = agg['count']

# train model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X, y)

forecast_year = forecast_target_year
lat_lon_unique = agg[['latitude', 'longitude']].drop_duplicates()
future_X = lat_lon_unique.copy()
future_X['year_num'] = forecast_year

forecast_pred = model.predict(future_X)
forecast_pred = np.maximum(forecast_pred, 0)

forecast_data = list(zip(
    future_X['latitude'],
    future_X['longitude'],
    forecast_pred
))

print(f"✅ Forecasted {len(forecast_data)} locations for {forecast_year}")

heatMapForecast = folium.Map(location=[-26.853388, 133.275154], zoom_start=5)
HeatMap(
    forecast_data,
    min_opacity=0.3,
    radius=12,
    blur=18,
    max_zoom=1,
    gradient={0.4:'blue', 0.6:'lime', 0.8:'red'}
).add_to(heatMapForecast)

fileMapName = os.path.join(output_dir, f"forecastHeatmapXGB_{forecast_year}_{fileTimeStamp}.html")
heatMapForecast.save(fileMapName)

print(f"\n✅ Forecast Heatmap (XGBoost {forecast_year}) saved → {fileMapName}\n")
webbrowser.open(fileMapName)
