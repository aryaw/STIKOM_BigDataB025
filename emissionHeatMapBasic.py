import os
import pandas as pd
import webbrowser
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
from libInternal import variableDump, getConnection, setFileLocation, cleanYear


fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
df_2024 = pd.read_sql("SELECT * FROM emmisions WHERE report_year = '2023/2024';", conn)

# ------------- heat map distribution
df_2024 = df_2024.dropna(subset=['latitude','longitude','report_year'])
df_2024['latitude'] = df_2024['latitude'].astype(float)
df_2024['longitude'] = df_2024['longitude'].astype(float)

heatMapRender = folium.Map(location=[-26.853388, 133.275154], zoom_start=4)

heatData = df_2024[['latitude', 'longitude']].dropna().values.tolist()
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