from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

def load_ev_grid_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    data_dir = Path(data_dir)

    # Core CSV fiels
    occ = pd.read_csv(data_dir / "occupancy.csv")      # timestamp + grid columns
    vol = pd.read_csv(data_dir / "volume.csv")         # timestamp + grid columns
    info = pd.read_csv(data_dir / "information.csv")   # one row per grid
    tdf = pd.read_csv(data_dir / "time.csv")           # one row per timestamp

    # Ensure time table has a timestamp column to join on
    if "timestamp" not in tdf.columns:
        tdf = tdf.copy()
        tdf["timestamp"] = range(1, len(tdf) + 1)

    # Wide -> long for occupancy and volume
    occ_long = occ.melt(
        id_vars="timestamp",
        var_name="grid",
        value_name="occupancy",
    )
    vol_long = vol.melt(
        id_vars="timestamp",
        var_name="grid",
        value_name="volume",
    )

    # Grid IDs as integers to match information.csv
    occ_long["grid"] = occ_long["grid"].astype(int)
    vol_long["grid"] = vol_long["grid"].astype(int)

    # Merge occupancy + volume
    ts = occ_long.merge(vol_long, on=["timestamp", "grid"], how="left")

    # Merge time info
    df = ts.merge(tdf, on="timestamp", how="left")

    # Merge static grid attributes
    df = df.merge(info, on="grid", how="left")

    # Datetime and time-derived features
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]]
    )
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Target: occupancy ratio (per connector) – avoid divide-by-zero
    df["occ_ratio"] = df["occupancy"] / df["count"].replace(0, pd.NA)
    df = df.dropna(subset=["occ_ratio"])

    # Add lag features
    df = add_lags(df)

    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["grid", "datetime"]).copy()

    grouped = df.groupby("grid", group_keys=False)

    df["occ_lag_1"] = grouped["occ_ratio"].shift(1)
    df["occ_lag_12"] = grouped["occ_ratio"].shift(12)
    df["occ_lag_288"] = grouped["occ_ratio"].shift(288)

    df = df.dropna(subset=["occ_lag_1", "occ_lag_12", "occ_lag_288"])
    return df

# This function is to plot the timeseries of the actual data vs the predicted data
def plot_grid_timeseries(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    grid_id: int,
    start_date: str = None,
    end_date: str = None,) -> None:
    """
    Plot actual vs predicted occ_ratio over time for a single grid.
    """
    df_plot = test_df.copy()
    df_plot = df_plot.assign(
        y_true=y_true.values,
        y_pred=y_pred,
    )

    # Filter by grid
    df_plot = df_plot[df_plot["grid"] == grid_id]

    # Optional date range
    if start_date is not None:
        df_plot = df_plot[df_plot["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot["datetime"] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["datetime"], df_plot["y_true"], label="Actual occ_ratio")
    plt.plot(df_plot["datetime"], df_plot["y_pred"], label="Predicted occ_ratio", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Occupancy ratio")
    plt.title(f"Grid {grid_id}: actual vs predicted occupancy ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_forecast_timeseries(forecast_df, grid_id, start_date=None, end_date=None):
    df_plot = forecast_df.copy()
    df_plot = df_plot[df_plot["grid"] == grid_id]

    if start_date is not None:
        df_plot = df_plot[df_plot["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot[df_plot["datetime"] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["datetime"], df_plot["occ_ratio_pred"], label="Forecast occ_ratio")
    plt.xlabel("Time")
    plt.ylabel("Occupancy ratio")
    plt.title(f"Grid {grid_id}: forecast occupancy ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()
