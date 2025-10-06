import os
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import plotly.express as px
import warnings
import time
from libInternal import variableDump, getConnection, setFileLocation, cleanYear

warnings.filterwarnings("ignore")

fileTimeStamp, output_dir = setFileLocation()
conn = getConnection()
df = pd.read_sql("SELECT * FROM emmisions;", conn)

df = df.dropna(subset=['latitude', 'longitude', 'report_year'])
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
df['year'] = df['report_year'].apply(cleanYear).astype(int)

if 'emission_count' not in df.columns:
    print("No emission_count column found — using record count per location/year.")
    df = df.groupby(['year','latitude','longitude']).size().reset_index(name='emission_count')
else:
    df = df.groupby(['year','latitude','longitude'])['emission_count'].sum().reset_index()

print(f"\n Data prepared: {len(df)} records, {df['year'].min()}–{df['year'].max()} range\n")

train_cutoff = df['year'].quantile(0.8)
train = df[df['year'] <= train_cutoff]
test  = df[df['year'] > train_cutoff]

print(f"Train years: {train['year'].min()}–{train['year'].max()}")
print(f"Test years : {test['year'].min()}–{test['year'].max()}\n")

prophet_forecasts = []

print("Training Prophet per location...")
start_prophet = time.time()
for (lat, lon), group in train.groupby(['latitude','longitude']):
    if len(group['year'].unique()) < 3:
        continue  # skip short series
    
    ts = group[['year','emission_count']].sort_values('year')
    ts = ts.rename(columns={'year':'ds','emission_count':'y'})
    ts['ds'] = pd.to_datetime(ts['ds'], format='%Y')

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(ts)

    future_years = sorted(test['year'].unique())
    future = pd.DataFrame({'ds': pd.to_datetime(future_years, format='%Y')})
    forecast = model.predict(future)
    forecast['year'] = forecast['ds'].dt.year
    forecast['latitude'] = lat
    forecast['longitude'] = lon
    prophet_forecasts.append(forecast[['latitude','longitude','year','yhat']])

prophet_time = time.time() - start_prophet
print(f"✅ Prophet training + forecasting completed in {prophet_time:.2f} seconds.")

if len(prophet_forecasts) == 0:
    raise ValueError("No Prophet forecasts generated — check your dataset.")

prophet_pred = pd.concat(prophet_forecasts, ignore_index=True)

merged_prophet = pd.merge(
    test, prophet_pred,
    on=['latitude','longitude','year'], how='inner'
)

# XGBoost
print("\n⚡ Training XGBoost...")
start_xgb = time.time()
X_train = train[['year','latitude','longitude']]
y_train = train['emission_count']

X_test = test[['year','latitude','longitude']]
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
print(f"✅ XGBoost training + forecasting completed in {xgb_time:.2f} seconds.\n")

xgb_pred = test.copy()
xgb_pred['yhat'] = y_pred_xgb

merged_xgb = xgb_pred.copy()

# Evaluate models
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

mae_prophet = mean_absolute_error(merged_prophet['emission_count'], merged_prophet['yhat'])
rmse_prophet = rmse(merged_prophet['emission_count'], merged_prophet['yhat'])

mae_xgb = mean_absolute_error(merged_xgb['emission_count'], merged_xgb['yhat'])
rmse_xgb = rmse(merged_xgb['emission_count'], merged_xgb['yhat'])

print("\n MODEL PERFORMANCE COMPARISON")
print(f"Prophet → MAE={mae_prophet:.3f}, RMSE={rmse_prophet:.3f}")
print(f"XGBoost → MAE={mae_xgb:.3f}, RMSE={rmse_xgb:.3f}")

sample_loc = df.groupby(['latitude','longitude']).size().idxmax()
lat_sample, lon_sample = sample_loc

actual_s = df[(df['latitude']==lat_sample) & (df['longitude']==lon_sample)]
prophet_s = merged_prophet[(merged_prophet['latitude']==lat_sample) & (merged_prophet['longitude']==lon_sample)]
xgb_s = merged_xgb[(merged_xgb['latitude']==lat_sample) & (merged_xgb['longitude']==lon_sample)]

fig = px.line(title=f"Forecast Comparison (Lat={lat_sample}, Lon={lon_sample})")
fig.add_scatter(x=actual_s['year'], y=actual_s['emission_count'], name="Actual", mode='lines+markers')
fig.add_scatter(x=prophet_s['year'], y=prophet_s['yhat'], name="Prophet Forecast", mode='lines+markers')
fig.add_scatter(x=xgb_s['year'], y=xgb_s['yhat'], name="XGBoost Forecast", mode='lines+markers')
fig.update_layout(xaxis_title="Year", yaxis_title="Emission Count")
file_fig = os.path.join(output_dir, f"forecast_comparison_{fileTimeStamp}.html")
fig.write_html(file_fig)

print(f"\nVisualization saved → {file_fig}")
print(f"Prophet vs XGBoost benchmark complete.\n")
