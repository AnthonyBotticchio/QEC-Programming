from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import lightgbm as lgb
import data

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent  # QEC-Programming/ root git dir
DATA_DIR = BASE_DIR / "data"

# Target and features
TARGET_COL = "occ_ratio"

NUMERIC_FEATURES = [
    "hour",
    "dayofweek",
    "is_weekend",
    "month",
    "count",
    "fast_count",
    "slow_count",
    "area",
    "lon",
    "la",
    "CBD",
    "dynamic_pricing",
    # Lag features from the data.py
    "occ_lag_1",
    "occ_lag_12",
    "occ_lag_288",
]

CATEGORICAL_FEATURES = ["grid"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def train_ev_grid_model_LGB(data_dir, model_path="ev_model.pkl"):

    df = data.load_ev_grid_data(data_dir).sort_values("datetime")

    # time split
    timeSplit = int(len(df) * 0.8)
    train_df = df.iloc[:timeSplit].copy()
    val_df = df.iloc[timeSplit:].copy()

    # dtype downcast
    for col in NUMERIC_FEATURES:
        train_df[col] = train_df[col].astype("float32")
        val_df[col] = val_df[col].astype("float32")

    train_df["grid"] = train_df["grid"].astype("int32")
    val_df["grid"] = val_df["grid"].astype("int32")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)])

    preds = model.predict(X_val)

    print("MAE:", mean_absolute_error(y_val, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_val, preds)))

    joblib.dump(model, model_path)

    return model

def test_ev_grid_model(model_path: str, data_dir: Path):
    """
    Load a saved model and evaluate it on the last 20% of the time-ordered data.
    Returns the test DataFrame plus y_true and y_pred for plotting.
    """
    model = joblib.load(model_path)
    df = data.load_ev_grid_data(data_dir)

    df_sorted = df.sort_values("datetime")
    split_idx = int(len(df_sorted) * 0.8)
    test_df = df_sorted.iloc[split_idx:]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test R²   : {r2:.4f}")

    return test_df, y_test, y_pred

def forecast_future_occ_ratio(
    model_path: str,
    data_dir: Path,
    horizon_steps: int = 288,
    freq_minutes: int = 5,
):
    """
    Forecast future occ_ratio for all grids for the next `horizon_steps`
    time steps (default: 288 * 5 minutes ≈ 24 hours).

    Returns a DataFrame with one row per (grid, future datetime).
    """

    # Load trained model and full historical data
    model = joblib.load(model_path)
    df_hist = data.load_ev_grid_data(data_dir)

    # Make sure history is time-ordered
    df_hist = df_hist.sort_values("datetime")

    # Last known datetime in the dataset
    last_time = df_hist["datetime"].max()
    freq = pd.Timedelta(minutes=freq_minutes)

    #    Keep only recent history needed for lags (last 288 steps is enough
    #    for a 24h ahead forecast with 5-min data)
    window_start = last_time - 288 * freq
    df_recent = df_hist[df_hist["datetime"] >= window_start].copy()

    # Dictionary: (grid, datetime) -> occ_ratio (actuals +, later, predictions)
    history = {
        (int(row.grid), row.datetime): float(row.occ_ratio)
        for row in df_recent.itertuples()
    }

    # Static grid features (take last row per grid)
    static_cols = [
        "grid",
        "count",
        "fast_count",
        "slow_count",
        "area",
        "lon",
        "la",
        "CBD",
        "dynamic_pricing",
    ]
    grid_static = (
        df_hist.sort_values("datetime")
        .groupby("grid", as_index=False)[static_cols]
        .last()
    )

    # Recursive forecasting loop
    all_forecasts = []

    for step in range(1, horizon_steps + 1):
        future_time = last_time + step * freq

        rows = []
        for row in grid_static.itertuples():
            g = int(row.grid)

            # Compute lag times
            t_lag_1 = future_time - 1 * freq
            t_lag_12 = future_time - 12 * freq
            t_lag_288 = future_time - 288 * freq

            # Look up lags in history (which we keep updating with predictions)
            # If a lag isn't available, skip this grid for this step
            try:
                occ_lag_1 = history[(g, t_lag_1)]
                occ_lag_12 = history[(g, t_lag_12)]
                occ_lag_288 = history[(g, t_lag_288)]
            except KeyError:
                # Not enough history yet for this grid at this horizon
                continue

            # Time features from future_time
            future_month = future_time.month
            future_hour = future_time.hour
            future_dayofweek = future_time.dayofweek
            future_is_weekend = 1 if future_dayofweek >= 5 else 0

            rows.append({
                "grid": g,
                "month": future_month,
                "hour": future_hour,
                "dayofweek": future_dayofweek,
                "is_weekend": future_is_weekend,
                "count": row.count,
                "fast_count": row.fast_count,
                "slow_count": row.slow_count,
                "area": row.area,
                "lon": row.lon,
                "la": row.la,
                "CBD": row.CBD,
                "dynamic_pricing": row.dynamic_pricing,
                "occ_lag_1": occ_lag_1,
                "occ_lag_12": occ_lag_12,
                "occ_lag_288": occ_lag_288,
                "datetime": future_time,
                "step_ahead": step,
            })

        if not rows:
            # No rows to predict for this step (shouldn't normally happen)
            continue

        future_df = pd.DataFrame(rows)

        # Predict occ_ratio for this future step
        X_future = future_df[FEATURE_COLS]
        y_hat = model.predict(X_future)

        future_df["occ_ratio_pred"] = y_hat
        all_forecasts.append(future_df)

        # Update history with predictions so later steps can use them as lags
        for g, t, y in zip(
            future_df["grid"], future_df["datetime"], future_df["occ_ratio_pred"]
        ):
            history[(int(g), t)] = float(y)

    if not all_forecasts:
        return pd.DataFrame()

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    return forecast_df

# Train model using LightGBM
model = train_ev_grid_model_LGB(data_dir=DATA_DIR, model_path="ev_grid_occ_ratio_model.joblib")

# Evaluate model on hold-out period
test_df, y_test, y_pred = test_ev_grid_model(model_path="ev_grid_occ_ratio_model.joblib", data_dir=DATA_DIR)

# Plot for one grid over a selected date range
data.plot_grid_timeseries(
    test_df,
    y_test,
    y_pred,
    grid_id=102,
    start_date="2022-07-13",
    end_date="2022-07-19",
)

forecast_df = forecast_future_occ_ratio(
        "ev_grid_occ_ratio_model.joblib",
        DATA_DIR,
        horizon_steps=500,
        freq_minutes=5,
    )

print(forecast_df.head())

grid_102_forecast = forecast_df[forecast_df["grid"] == 102] # Forecast grid 102
grid_102_forecast.plot(x="datetime", y="occ_ratio_pred")

data.plot_forecast_timeseries(forecast_df, grid_id=102)
