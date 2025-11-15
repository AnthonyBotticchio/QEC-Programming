import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_ev_usage_model(data_dir, target="occupancy"):
    data_dir = Path(data_dir)

    # 1) Load CSVs
    stations = pd.read_csv(data_dir / "stations.csv")          # station_id, lat, lon, power, ...
    info     = pd.read_csv(data_dir / "information.csv")       # station_id, extra metadata
    dist     = pd.read_csv(data_dir / "distance.csv")          # station_id, distance_to_cbd, ...
    occ      = pd.read_csv(data_dir / "occupancy.csv")         # station_id, timestamp, occupancy
    vol      = pd.read_csv(data_dir / "volume.csv")            # station_id, timestamp, sessions/kWh
    # If time.csv is just derived features, you may not need it; you can build your own.

    # 2) Merge time-series (occ + vol) into one table
    ts = occ.merge(vol, on=["station_id", "timestamp"], how="left")

    # 3) Attach station-level data
    df = (ts
          .merge(stations,   on="station_id", how="left")
          .merge(info,       on="station_id", how="left")
          .merge(dist,       on="station_id", how="left"))

    # 4) Parse timestamp and engineer time features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 5) Choose target variable
    if target == "occupancy":
        y = df["occupancy"]  # adjust to match your column name
    elif target == "volume":
        y = df["sessions"]   # or kWh; adjust to your column name
    else:
        raise ValueError("Unknown target")

    # 6) Select features (you’ll need to adjust these to match your actual columns)
    numeric_features = [
        "hour",
        "dayofweek",
        "is_weekend",
        "lat",
        "lon",
        "power_kw",          # from stations.csv
        "num_connectors",
        "distance_to_cbd",   # from distance.csv
        # add more numeric features...
    ]

    categorical_features = [
        "station_type",      # e.g., Level2/DCFC
        "neighbourhood",     # if present
        # add more categorical fields...
    ]

    # Drop rows with missing target
    mask = y.notna()
    df = df.loc[mask]
    y = y.loc[mask]

    X = df[numeric_features + categorical_features]

    # 7) Build preprocessing + model pipeline
    numeric_transformer = StandardScaler()
    # Simple one-hot on categoricals; for large cardinality, you might want something more advanced
    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # 8) Time-based or simple split (better: split by timestamp; as a first pass we use random split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 9) Train
    clf.fit(X_train, y_train)

    # 10) Evaluate
    y_pred = clf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.3f} ({target})")

    # 11) Save model
    joblib.dump(clf, data_dir / f"ev_usage_model_{target}.joblib")

    return clf, mae
