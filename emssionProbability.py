import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime

import webbrowser
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

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
# query = "SELECT * FROM emmisions LIMIT 100000;"
query = "SELECT * FROM emmisions;"
print(f"\n✅ Get SQL data")
df = pd.read_sql(query, conn)

df = df.dropna(subset=['report_year', 'substance_name', 'primary_anzsic_class_name'])
df['report_year'] = df['report_year'].astype(str)
df['substance_name'] = df['substance_name'].astype(str)
df['primary_anzsic_class_name'] = df['primary_anzsic_class_name'].astype(str)

le = LabelEncoder()
X = pd.get_dummies(df['report_year'])

file_substance_name = os.path.join(output_dir, f"regrt_top10_substances_{fileTimeStamp}.html")

y_substance = le.fit_transform(df['substance_name'])

model_substance = LogisticRegression(solver='lbfgs', max_iter=500)
model_substance.fit(X, y_substance)

probs_substance = model_substance.predict_proba(X)
probs_substance_df = pd.DataFrame(probs_substance, columns=le.classes_)
probs_substance_df['report_year'] = df['report_year'].values

top10_substance = df['substance_name'].value_counts().nlargest(10).index
for substance_item in top10_substance:
    df[f'prob_substance_{substance_item}'] = probs_substance_df[substance_item]

fig_substance = px.line(
    df,
    x='report_year',
    y=[f'prob_substance_{s}' for s in top10_substance],
    title='Predicted Probability of Top 10 Substances Over Years',
    labels={'value':'Predicted Probability', 'report_year':'Year'}
)
# fig_substance.write_html(file_substance_name, include_plotlyjs=True, full_html=True)

# svg format
traces = []
for s in top10_substance:
    traces.append(go.Scatter(
        x=df['report_year'],
        y=df[f'prob_substance_{s}'],
        mode='lines',
        name=s
    ))

fig_substance_svg = go.Figure(traces)
fig_substance_svg.update_layout(
    title="Predicted Probability of Top 10 Substances Over Years",
    xaxis_title="Year",
    yaxis_title="Predicted Probability"
)
fig_substance_svg.write_html(file_substance_name, include_plotlyjs=True, full_html=True)

print(f"\n✅ Substance chart saved → {file_substance_name}")
webbrowser.open(file_substance_name)
