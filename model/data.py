import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_ev_grid_data(data_dir: str) -> pd.DataFrame:
    data_dir = Path(data_dir)

    # 1) Load core tables
    occ = pd.read_csv(data_dir / "occupancy.csv")      # timestamp + grid columns
    vol = pd.read_csv(data_dir / "volume.csv")         # timestamp + grid columns
    info = pd.read_csv(data_dir / "information.csv")   # one row per grid
    tdf = pd.read_csv(data_dir / "time.csv")           # one row per timestamp

    # 2) Add matching timestamp index to time table (1..8640)
    tdf = tdf.copy()
    tdf["timestamp"] = range(1, len(tdf) + 1)

    # 3) Convert wide -> long for occupancy and volume
    occ_long = occ.melt(id_vars="timestamp",
                        var_name="grid",
                        value_name="occupancy")
    vol_long = vol.melt(id_vars="timestamp",
                        var_name="grid",
                        value_name="volume")

    # grid IDs come in as strings from column names; convert to int to match information.csv
    occ_long["grid"] = occ_long["grid"].astype(int)
    vol_long["grid"] = vol_long["grid"].astype(int)

    # 4) Merge occupancy + volume
    ts = occ_long.merge(vol_long, on=["timestamp", "grid"], how="left")

    # 5) Merge time info
    df = ts.merge(tdf, on="timestamp", how="left")

    # 6) Merge static grid attributes
    df = df.merge(info, on="grid", how="left")

    # 7) Build a proper datetime and extra time features
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]]
    )
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 8) Target: occupancy ratio (bounded 0–1) using connector count per grid
    # 'count' here comes from information.csv (total connectors in the grid)
    df["occ_ratio"] = df["occupancy"] / df["count"].replace(0, pd.NA)
    df = df.dropna(subset=["occ_ratio"])

    return df

def plot_grid_timeseries(test_df, y_test, y_pred, grid_id, start_date=None, end_date=None):
    df_plot = test_df.copy()
    df_plot = df_plot.assign(
        y_true=y_test.values,
        y_pred=y_pred
    )

    # Filter by grid
    df_plot = df_plot[df_plot["grid"] == grid_id]

    # Optional: filter by date range
    if start_date is not None:
        df_plot = df_plot[df_plot["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot["datetime"] <= pd.to_datetime(end_date)]

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["datetime"], df_plot["y_true"], label="Actual occ_ratio")
    plt.plot(df_plot["datetime"], df_plot["y_pred"], label="Predicted occ_ratio", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Occupancy ratio")
    plt.title(f"Grid {grid_id}: actual vs predicted occupancy ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()
