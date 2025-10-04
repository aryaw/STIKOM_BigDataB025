import os
import pandas as pd
import webbrowser
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px

from libInternal import variableDump, getConnection, setFileLocation


fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
# df = pd.read_sql("SELECT * FROM emmisions ORDER BY RAND() LIMIT 300000;", conn)
df = pd.read_sql("SELECT * FROM emmisions;", conn)

# ------------- map distribution
df = df.dropna(subset=['latitude','longitude','report_year'])
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

mapRender = folium.Map(location=[-6.200000, 106.816666], zoom_start=6)
mapCluster = MarkerCluster().add_to(mapRender)

for row in df.itertuples(index=False):
    htmlPopup = f"""
    <b>{row.facility_name}</b><br>
    Business name: {row.registered_business_name}<br>
    Facilty type: {row.primary_anzsic_class_name}<br>
    Substance name: {row.substance_name}<br>
    Air emission (kg): {row.air_total_emission_kg}<br>
    Water emission (kg): {row.water_emission_kg}<br>
    Report Year: {row.report_year}<br>
    Land emission (kg): {row.land_emission_kg if pd.notna(row.land_emission_kg) else "N/A"}<br>
    Location): {row.suburb} {row.state} {row.postcode}<br>
    """

    # log console
    # print(f"""
    # Facility: {row.facility_name}
    # Business: {row.registered_business_name}
    # Type    : {row.primary_anzsic_class_name}
    # Substance: {row.substance_name}
    # Air (kg): {row.air_total_emission_kg}
    # Water(kg): {row.water_emission_kg}
    # Land (kg): {row.land_emission_kg if pd.notna(row.land_emission_kg) else "N/A"}
    # Location : {row.suburb}, {row.state} {row.postcode}
    # Lat/Lon  : {row.latitude}, {row.longitude}
    # ----------------------------
    # """)
    
    folium.Marker(
        location=[row.latitude, row.longitude],
        tooltip=row.facility_name,
        popup=folium.Popup(htmlPopup, max_width=250),
        icon=folium.Icon(color="green"),
    ).add_to(mapCluster)

fileMapName = os.path.join(output_dir, f"mapRender_{fileTimeStamp}.html")
mapRender.save(fileMapName)

print(f"\n✅ Finish Map Render → {fileMapName}\n")
webbrowser.open(fileMapName)