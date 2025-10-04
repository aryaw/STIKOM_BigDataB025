import os
import pandas as pd
import webbrowser
import plotly.express as px

from libInternal import variableDump, getConnection, setFileLocation


fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
df = pd.read_sql("SELECT * FROM emmisions ORDER BY RAND() LIMIT 300000;", conn)

# ------------- distribution
fig = px.histogram(
    df, 
    x="substance_name", 
    nbins=40, 
    title="Substance Name Distribution",
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