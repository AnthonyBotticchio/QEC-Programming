from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import data

def train_ev_grid_model(data_dir: str, model_path: str = "ev_grid_occ_ratio_model.joblib"):
    # 1) Build full data table
    df = data.load_ev_grid_data(data_dir)

    # 2) Choose target and feature columns
    target_col = "occ_ratio"

    numeric_features = [
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
    ]

    categorical_features = [
        "grid",  # treat grid ID as a category
    ]

    feature_cols = numeric_features + categorical_features

    # 3) Sort by time and do a time-based split (e.g., last 20% as validation)
    df_sorted = df.sort_values("datetime")
    split_idx = int(len(df_sorted) * 0.8)

    train_df = df_sorted.iloc[:split_idx]
    val_df   = df_sorted.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val   = val_df[feature_cols]
    y_val   = val_df[target_col]

    # 4) Preprocessing + model
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # 5) Train
    pipe.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE (occ_ratio): {mae:.4f}")

    # 7) Save model
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")

    return pipe, mae

def test_ev_grid_model(model_path: str, data_dir: str):
    # 1) Load model and data
    model = joblib.load(model_path)
    df = data.load_ev_grid_data(data_dir)

    target_col = "occ_ratio"

    numeric_features = [
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
    ]
    categorical_features = ["grid"]
    feature_cols = numeric_features + categorical_features

    # 2) Sort by time and create a hold-out split (e.g., last 20% as test)
    df_sorted = df.sort_values("datetime")
    split_idx = int(len(df_sorted) * 0.8)

    test_df = df_sorted.iloc[split_idx:]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 3) Predict
    y_pred = model.predict(X_test)

    # 4) Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test R²   : {r2:.4f}")

    return test_df, y_test, y_pred


pipe, mae_val = train_ev_grid_model("../data", model_path="ev_grid_occ_ratio_model.joblib")
test_df, y_test, y_pred = test_ev_grid_model("ev_grid_occ_ratio_model.joblib", "data")
data.plot_grid_timeseries(test_df, y_test, y_pred, grid_id=102,
                     start_date="2022-07-01", end_date="2022-07-07")
