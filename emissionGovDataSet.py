import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

import webbrowser
from datetime import datetime
import pandas as pd

import folium
from folium.plugins import MarkerCluster

import plotly.express as px

load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_pass = quote_plus(os.getenv("DB_PASS"))
db_name = os.getenv("DB_NAME")

fileTimeStamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = "/home/arya/Documents/Pasca Stikom/BigData/Repo/DataMap/mapRender/assets"
os.makedirs(output_dir, exist_ok=True) 

conn = create_engine(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}")

df = pd.read_sql("SELECT * FROM emmisions;", conn)

# distribution
fig = px.histogram(
    df, 
    x="substance_name", 
    nbins=40, 
    title="Substance Name Distribution",
    # color_discrete_sequence=["skyblue"],
    color="substance_name", 
    category_orders={
        "substance_name": df["substance_name"].value_counts().index.tolist()
    }
)
fig.update_traces(marker_line_color="black", marker_line_width=1)
fig.update_layout(bargap=0.1)

fileDistributionName = os.path.join(output_dir, f"distributionRender_{fileTimeStamp}.html")
fig.write_html(fileDistributionName, include_plotlyjs="cdn")

print(f"\n✅ Finish Distribution Render → {fileDistributionName}\n")
webbrowser.open(fileDistributionName)

# scatter plot
# fig_scatter = px.scatter(
#     df,
#     x="substance_name",
#     y="report_year",
#     color="substance_name",
#     hover_data=["facility_name", "substance_name", 'report_year', "state"],
#     title="Substance Count by Report Year",
# )

df_grouped = df.groupby(
    ["report_year", "substance_name"]
).size().reset_index(name="count")
fig_scatter = px.scatter(
    df_grouped,
    x="substance_name",
    y="report_year",
    size="count",
    color="substance_name",
    hover_data=["count", "report_year", "substance_name"],
    title="Substance Count by Report Year"
)

fig_scatter.update_layout(
    xaxis_title="Substance",
    yaxis_title="Year",
    legend_title="Report Year",
    template="plotly_white",
    yaxis=dict(
        tickmode="linear",
        dtick=1,
        categoryorder="category ascending"
    )
)
fig_scatter.update_traces(
    marker=dict(
        line=dict(width=2, color="white")
    )
)

fileScatterName = os.path.join(output_dir, f"scatterRender_{fileTimeStamp}.html")
fig_scatter.write_html(fileScatterName, include_plotlyjs="cdn")

print(f"\n✅ Finish Scatter Render → {fileScatterName}\n")
webbrowser.open(fileScatterName)

# map distribution
df = df.dropna(subset=['latitude','longitude'])
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

mapRender = folium.Map(location=[-6.200000, 106.816666], zoom_start=6)
marker_cluster = MarkerCluster().add_to(mapRender)

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
    ).add_to(marker_cluster)

fileMapName = os.path.join(output_dir, f"mapRender_{fileTimeStamp}.html")
mapRender.save(fileMapName)

print(f"\n✅ Finish Map Render → {fileMapName}\n")
webbrowser.open(fileMapName)