import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime

import webbrowser
import pandas as pd

import folium
from folium.plugins import MarkerCluster

import plotly.express as px

#import statsmodels.api as sm
#import statsmodels.formula.api as smf

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_pass = quote_plus(os.getenv("DB_PASS"))
db_name = os.getenv("DB_NAME")

fileTimeStamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = "/home/arya/Documents/Pasca Stikom/BigData/Repo/DataMap/tugas1BigData/assets"
os.makedirs(output_dir, exist_ok=True) 

conn = create_engine(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}")

df = pd.read_sql("SELECT * FROM emmisions ORDER BY RAND() LIMIT 300000;", conn)

# ------------- distribution
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

# ------------- group substance by year report
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
    xaxis_title="Substance Name",
    yaxis_title="Year",
    legend_title="Substance Name",
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

# ------------- regret
# drop empty rows
df_reg = df.dropna(subset=['report_year', 'substance_name', 'primary_anzsic_class_name'])

print(f"\n✅ Start cast data \n")
# cast all data as a string
df['report_year'] = df['report_year'].astype(str)
df['substance_name'] = df['substance_name'].astype(str)
df['primary_anzsic_class_name'] = df['primary_anzsic_class_name'].astype(str)

X = pd.get_dummies(df['report_year'])

print(f"\n✅ Start probs substance \n")
le_substance = LabelEncoder()
y_substance = le_substance.fit_transform(df['substance_name'])
model_substance = LogisticRegression(solver='lbfgs', max_iter=500)
model_substance.fit(X, y_substance)
probs_substance = model_substance.predict_proba(X)
probs_df_substance = pd.DataFrame(probs_substance, columns=le_substance.classes_)
probs_df_substance['report_year'] = df['report_year'].values

print(f"\n✅ Start probs industry \n")
le_industry = LabelEncoder()
y_industry = le_industry.fit_transform(df['primary_anzsic_class_name'])
model_industry = LogisticRegression(solver='lbfgs', max_iter=500)
model_industry.fit(X, y_industry)
probs_industry = model_industry.predict_proba(X)
probs_df_industry = pd.DataFrame(probs_industry, columns=le_industry.classes_)
probs_df_industry['report_year'] = df['report_year'].values

print(f"\n✅ Start concat \n")
probs_df = pd.concat([probs_df_substance.drop(columns='report_year'),
                      probs_df_industry.drop(columns='report_year')],
                     axis=1)
probs_df['report_year'] = df['report_year'].values

# sortby year of report
df = df.sort_values(by='report_year', ascending=True)

# get top 10 substance
top10_substance = df['substance_name'].value_counts().nlargest(10).index
for sub in top10_substance:
    df[f'prob_{sub}'] = probs_df[sub]

# get top 10 industry
top10_industry = df['primary_anzsic_class_name'].value_counts().nlargest(10).index
for sub in top10_industry:
    df[f'prob_{sub}'] = probs_df[sub]

print(f"\n✅ Get Top 10 industry & substance \n")
y_columns = [f'prob_{s}' for s in top10_substance] + [f'prob_{i}' for i in top10_industry]

fig_combined_substance_industry = px.line(
    df,
    x='report_year',
    y=y_columns,
    title='Predicted Probability of Top 10 Substances & Top 10 Industries Over Years',
    labels={'value':'Predicted Probability', 'report_year':'Year'}
)

file_combined_substance_industry_name = os.path.join(output_dir, f"regrt_combined_substance_industry_{fileTimeStamp}.html")
fig_combined_substance_industry.write_html(file_combined_substance_industry_name, include_plotlyjs='cdn', full_html=True)

print(f"\n✅ Finish Pobrality Render → {file_combined_substance_industry_name}\n")
webbrowser.open(file_combined_substance_industry_name)


# ------------- map distribution
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