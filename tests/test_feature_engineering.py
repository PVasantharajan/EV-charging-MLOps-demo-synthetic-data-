import pandas as pd
import numpy as np

from src.feature_engineering import (
    compute_soc_delta,
    add_time_features,
    add_lag_features,
    create_targets,
)


def make_small_df():
    """
    synthetic EV trace for one asset with a long gap on last row.
    """
    return pd.DataFrame({
        "asset_id": [1, 1, 1, 1],
        "timestamp": pd.to_datetime([
            "2025-06-01 00:00:00",
            "2025-06-01 00:15:00",
            "2025-06-01 00:30:00",
            "2025-06-01 03:30:00",   # big gap
        ]),
        "soc": [50.0, 52.0, 55.0, 70.0],
        "is_plugged_in": [1, 1, 0, 0],
        "gap_flag": [False, False, False, True],
    })


##################
# 1) SOC DELTA
##################
def test_soc_delta_basic():
    df = make_small_df()
    out = compute_soc_delta(df)

    assert "soc_delta" in out.columns
    assert np.isnan(out.loc[0, "soc_delta"])
    assert out.loc[1, "soc_delta"] == 2.0
    assert out.loc[2, "soc_delta"] == 3.0
    assert np.isnan(out.loc[3, "soc_delta"])  # masked due to gap


##################
# 2) TIME FEATURES
##################
def test_time_features():
    df = make_small_df()
    out = add_time_features(df)

    assert "hour" in out.columns
    assert "day_of_week" in out.columns

    assert out.loc[0, "hour"] == 0
    assert set(out["day_of_week"].unique()) == {6}  # Sunday


###################
# 3) LAG FEATURES
###################
def test_lag_features():
    df = make_small_df()
    df = compute_soc_delta(df)
    out = add_lag_features(df, lags=[1])

    assert "soc_lag_1" in out.columns

    assert np.isnan(out.loc[0, "soc_lag_1"])
    assert out.loc[1, "soc_lag_1"] == df.loc[0, "soc"]


###################
# 4) TARGET CREATION
###################
def test_create_targets():
    df = make_small_df()
    out = create_targets(df, horizon_steps=1)

    assert "plug_future" in out.columns
    assert "energy_to_full_future" in out.columns

    # plug_future shift check
    assert out.loc[0, "plug_future"] == df.loc[1, "is_plugged_in"]

    # last row has no future → NaN
    assert np.isnan(out.loc[len(out) - 1, "plug_future"])

    # energy_to_full_future = 100 - future soc
    expected = 100 - df.loc[1, "soc"]
    assert out.loc[0, "energy_to_full_future"] == expected
