import os
import pandas as pd
import webbrowser
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
from prophet import Prophet

from libInternal import variableDump, getConnection, setFileLocation


fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
df = pd.read_sql("SELECT * FROM emmisions;", conn)

# ------------- heat map distribution
df = df.dropna(subset=['latitude','longitude','report_year'])
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

heatMapRender = folium.Map(location=[-6.200000, 106.816666], zoom_start=4)

heatData = df[['latitude', 'longitude']].dropna().values.tolist()
HeatMap(
    heatData,
    min_opacity=0.3,
    radius=10,
    blur=15,
    max_zoom=1,
).add_to(heatMapRender)

fileMapName = os.path.join(output_dir, f"heatMapRender_{fileTimeStamp}.html")
heatMapRender.save(fileMapName)

print(f"\n✅ Finish Map Render → {fileMapName}\n")
webbrowser.open(fileMapName)


# heat forecast
def clean_year(val):
    if pd.isna(val):
        return None
    val = str(val)
    if "/" in val:
        return int(val.split("/")[1])
    return int(val)

df['report_year'] = df['report_year'].apply(clean_year)

future_years = 10
forecast_target_year = 2030
forecast_data = []

for (lat, lon), group in df.groupby(['latitude','longitude']):
    if len(group['report_year'].unique()) >= 4:

        # count record per year
        ts = group.groupby("report_year").size().reset_index(name="count")
        ts = ts.rename(columns={"report_year":"ds", "count":"y"})
        ts['ds'] = pd.to_datetime(ts['ds'], format='%Y')
        
        # fit model
        model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
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
            

heatMapForecast = folium.Map(location=[-6.200000, 106.816666], zoom_start=5)

HeatMap(
    forecast_data,
    min_opacity=0.3,
    radius=12,
    blur=18,
    max_zoom=1,
).add_to(heatMapForecast)

fileMapName = os.path.join(output_dir, f"forecastHeatmap_{forecast_target_year}_{fileTimeStamp}.html")
heatMapForecast.save(fileMapName)

print(f"\n✅ Forecast Heatmap {forecast_target_year} → {fileMapName}\n")
webbrowser.open(fileMapName)