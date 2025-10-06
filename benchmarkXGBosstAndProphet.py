import os
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time
from libInternal import variableDump, getConnection, setFileLocation, cleanYear

warnings.filterwarnings("ignore")

# setup & data loading
fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
df = pd.read_sql("SELECT * FROM emmisions;", conn)

df = df.dropna(subset=['latitude', 'longitude', 'report_year'])
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
df['year'] = df['report_year'].apply(cleanYear).astype(int)

if 'emission_count' not in df.columns:
    print("No emission_count column found — using record count per location/year.")
    df = df.groupby(['year', 'latitude', 'longitude']).size().reset_index(name='emission_count')
else:
    df = df.groupby(['year', 'latitude', 'longitude'])['emission_count'].sum().reset_index()

print(f"\n Data prepared: {len(df)} records, {df['year'].min()}–{df['year'].max()} range\n")

# train test split
train_cutoff = df['year'].quantile(0.8)
train = df[df['year'] <= train_cutoff]
test = df[df['year'] > train_cutoff]

print(f"Train years: {train['year'].min()}–{train['year'].max()}")
print(f"Test years : {test['year'].min()}–{test['year'].max()}\n")

# prophet per location
prophet_forecasts = []
print("Training Prophet per location...")
start_prophet = time.time()

for (lat, lon), group in train.groupby(['latitude', 'longitude']):
    if len(group['year'].unique()) < 3:
        continue
    
    ts = group[['year', 'emission_count']].sort_values('year')
    ts = ts.rename(columns={'year': 'ds', 'emission_count': 'y'})
    ts['ds'] = pd.to_datetime(ts['ds'], format='%Y')

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(ts)

    future_years = sorted(test['year'].unique())
    future = pd.DataFrame({'ds': pd.to_datetime(future_years, format='%Y')})
    forecast = model.predict(future)
    forecast['year'] = forecast['ds'].dt.year
    forecast['latitude'] = lat
    forecast['longitude'] = lon
    prophet_forecasts.append(forecast[['latitude', 'longitude', 'year', 'yhat']])

prophet_time = time.time() - start_prophet
print(f"✅ Prophet training + forecasting completed in {prophet_time:.2f} seconds.")

if len(prophet_forecasts) == 0:
    raise ValueError("No Prophet forecasts generated — check your dataset.")

prophet_pred = pd.concat(prophet_forecasts, ignore_index=True)
merged_prophet = pd.merge(test, prophet_pred, on=['latitude', 'longitude', 'year'], how='inner')

# XGBoost
print("\n Training XGBoost...")
start_xgb = time.time()
X_train = train[['year', 'latitude', 'longitude']]
y_train = train['emission_count']
X_test = test[['year', 'latitude', 'longitude']]
y_test = test['emission_count']

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_time = time.time() - start_xgb

print(f"XGBoost training + forecasting completed in {xgb_time:.2f} seconds.\n")

merged_xgb = test.copy()
merged_xgb['yhat'] = y_pred_xgb

# metrics calculation
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

mae_prophet = mean_absolute_error(merged_prophet['emission_count'], merged_prophet['yhat'])
rmse_prophet = rmse(merged_prophet['emission_count'], merged_prophet['yhat'])

mae_xgb = mean_absolute_error(merged_xgb['emission_count'], merged_xgb['yhat'])
rmse_xgb = rmse(merged_xgb['emission_count'], merged_xgb['yhat'])

# --- FIXED F1-score section ---
def categorize_emission_dynamic(y):
    """Quantile-based emission category (ensures balanced bins)."""
    q = y.quantile([0.33, 0.66]).values
    bins = [0, q[0], q[1], np.inf]
    labels = ["Low", "Medium", "High"]
    return pd.cut(y, bins=bins, labels=labels, include_lowest=True)

def safe_f1(y_true, y_pred):
    try:
        y_true_cat = categorize_emission_dynamic(pd.Series(y_true))
        y_pred_cat = categorize_emission_dynamic(pd.Series(y_pred))
        return f1_score(
            y_true_cat, y_pred_cat,
            labels=["Low", "Medium", "High"],
            average="weighted",
            zero_division=0
        )
    except Exception as e:
        print(f"⚠️ F1 warning: {e}")
        return np.nan

f1_prophet = safe_f1(merged_prophet['emission_count'], merged_prophet['yhat'])
f1_xgb = safe_f1(merged_xgb['emission_count'], merged_xgb['yhat'])

# results
print("\n MODEL PERFORMANCE COMPARISON")
print(f"Prophet → MAE={mae_prophet:.3f}, RMSE={rmse_prophet:.3f}, F1={f1_prophet:.3f}")
print(f"XGBoost → MAE={mae_xgb:.3f}, RMSE={rmse_xgb:.3f}, F1={f1_xgb:.3f}")

# visualization
sample_loc = df.groupby(['latitude', 'longitude']).size().idxmax()
lat_sample, lon_sample = sample_loc
actual_s = df[(df['latitude'] == lat_sample) & (df['longitude'] == lon_sample)]
prophet_s = merged_prophet[(merged_prophet['latitude'] == lat_sample) & (merged_prophet['longitude'] == lon_sample)]
xgb_s = merged_xgb[(merged_xgb['latitude'] == lat_sample) & (merged_xgb['longitude'] == lon_sample)]

fig = px.line(title=f"Forecast Comparison (Lat={lat_sample}, Lon={lon_sample})")
fig.add_scatter(x=actual_s['year'], y=actual_s['emission_count'], name="Actual", mode='lines+markers')
fig.add_scatter(x=prophet_s['year'], y=prophet_s['yhat'], name="Prophet Forecast", mode='lines+markers')
fig.add_scatter(x=xgb_s['year'], y=xgb_s['yhat'], name="XGBoost Forecast", mode='lines+markers')
fig.update_layout(xaxis_title="Year", yaxis_title="Emission Count")
file_fig = os.path.join(output_dir, f"forecast_comparison_{fileTimeStamp}.html")
fig.write_html(file_fig)

# metrics visualization
metrics_df = pd.DataFrame({
    "Model": ["Prophet", "XGBoost"],
    "MAE": [mae_prophet, mae_xgb],
    "RMSE": [rmse_prophet, rmse_xgb],
    "F1-score": [f1_prophet, f1_xgb],
    "Training Time (s)": [prophet_time, xgb_time]
}).round(3)

# chart
metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")

fig_bar = go.Figure()
for model in metrics_df["Model"]:
    subset = metrics_melted[metrics_melted["Model"] == model]
    fig_bar.add_trace(go.Bar(
        x=subset["Metric"], y=subset["Value"], name=model,
        text=subset["Value"], textposition="outside"
    ))

fig_bar.update_layout(
    title="⚖️ Prophet vs XGBoost Performance Metrics",
    xaxis_title="Metric", yaxis_title="Value",
    barmode="group", template="plotly_white", legend=dict(title="Model")
)

file_bar = os.path.join(output_dir, f"model_performance_chart_{fileTimeStamp}.html")
fig_bar.write_html(file_bar)
print(f"\nVisualization saved as interactive HTML: {file_bar}\n")

# table
fig_table = go.Figure(
    data=[go.Table(
        header=dict(
            values=list(metrics_df.columns),
            fill_color="#2C3E50",
            align="center",
            font=dict(color="white", size=14)
        ),
        cells=dict(
            values=[metrics_df[col] for col in metrics_df.columns],
            fill_color=[["#F9F9F9", "#FFFFFF"] * (len(metrics_df)//2 + 1)],
            align="center",
            font=dict(size=12)
        )
    )]
)
fig_table.update_layout(
    title_text=" Model Performance Table: Prophet vs XGBoost",
    title_x=0.5,
    margin=dict(l=20, r=20, t=60, b=20)
)
file_table = os.path.join(output_dir, f"model_performance_table_{fileTimeStamp}.html")
fig_table.write_html(file_table)
print(f"\nVisualization saved as interactive HTML: {file_table}\n")

print(f"\n Visualization exported:")
print(f"- Performance Table → {file_table}")
print(f"- Bar Chart → {file_bar}")
print(f"- Forecast Comparison → {file_fig}")
print("\n Prophet vs XGBoost benchmark complete.\n")
