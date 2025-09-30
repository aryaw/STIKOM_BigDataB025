import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import webbrowser

from libInternal import variableDump, getConnection, setFileLocation

forecast_year = "2026/2027"
print(f"\n✅ year set → {forecast_year}")

fileTimeStamp, output_dir = setFileLocation()
file_chart_svg = os.path.join(output_dir, f"forecast_industry_{fileTimeStamp}.html")

conn = getConnection()
query = "SELECT * FROM emmisions;"
df = pd.read_sql(query, conn)

df = df.dropna(subset=["report_year", "substance_name", "primary_anzsic_class_name"])
df["report_year"] = df["report_year"].astype(str)
df["primary_anzsic_class_name"] = df["primary_anzsic_class_name"].astype(str)

all_years = sorted(df["report_year"].unique().tolist() + [forecast_year])
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=[all_years])

X_array = enc.fit_transform(df[["report_year"]])
X = pd.DataFrame(X_array, columns=enc.get_feature_names_out(["report_year"]))

X_forecast = pd.DataFrame(
    enc.transform([[forecast_year]]),
    columns=enc.get_feature_names_out(["report_year"])
)

le_industry = LabelEncoder()
y = le_industry.fit_transform(df["primary_anzsic_class_name"])

model = LogisticRegression(solver="lbfgs", max_iter=500)
model.fit(X, y)

X_all = pd.DataFrame(
    enc.transform([[y] for y in all_years]),
    columns=enc.get_feature_names_out(["report_year"])
)

probs = model.predict_proba(X_all)
probs_df = pd.DataFrame(probs, columns=le_industry.classes_)
probs_df["report_year"] = all_years

top10_industry = df["primary_anzsic_class_name"].value_counts().nlargest(10).index

traces = []
for s in top10_industry:
    if s in probs_df.columns:
        traces.append(go.Scatter(
            x=probs_df["report_year"],
            y=probs_df[s],
            mode="lines+markers",
            name=s
        ))

fig_svg = go.Figure(traces)
fig_svg.update_layout(
    title=f"Predicted Probability of Top 10 Industry Over Years (Including Forecast {forecast_year})",
    xaxis_title="Year",
    yaxis_title="Predicted Probability",
    xaxis=dict(tickangle=-45)
)

fig_svg.write_html(file_chart_svg, include_plotlyjs=True, full_html=True)

print(f"\n✅ Chart saved → {file_chart_svg}")
webbrowser.open(file_chart_svg)