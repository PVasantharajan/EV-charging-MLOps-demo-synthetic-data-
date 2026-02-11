import pandas as pd
import numpy as np


def compute_soc_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soc_delta"] = df.groupby("asset_id")["soc"].diff()

    # Mask delta across long gaps
    df.loc[df["gap_flag"] == True, "soc_delta"] = np.nan
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def add_lag_features(df: pd.DataFrame, lags=[1, 2, 3]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"soc_lag_{lag}"] = df.groupby("asset_id")["soc"].shift(lag)
    return df


def create_targets(df: pd.DataFrame, horizon_steps=4) -> pd.DataFrame:
    df = df.copy()

    df["plug_future"] = (
        df.groupby("asset_id")["is_plugged_in"]
        .shift(-horizon_steps)
    )

    df["soc_future"] = (
        df.groupby("asset_id")["soc"]
        .shift(-horizon_steps)
    )

    df["energy_to_full_future"] = 100 - df["soc_future"]

    return df
