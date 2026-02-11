import pandas as pd
import numpy as np

from src.preprocessing import (
    parse_and_sort,
    compute_time_diff,
    flag_gaps,
    filter_short_assets,
)


def make_df():
    return pd.DataFrame({
        "asset_id": [1, 1, 1, 2],
        "timestamp": [
            "2025-06-01 00:15:00",
            "2025-06-01 00:00:00",   # out of order
            "2025-06-01 05:00:00",   # long gap
            "2025-06-01 00:00:00",
        ],
        "soc": [52, 50, 70, 80],
        "is_plugged_in": [1, 1, 0, 1],
    })


############################
# 1) PARSE + SORT
############################
def test_parse_and_sort():
    df = make_df()
    out = parse_and_sort(df)

    assert out.iloc[0]["timestamp"] < out.iloc[1]["timestamp"]
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


############################
# 2) TIME DIFF
############################
def test_compute_time_diff():
    df = make_df()
    df = parse_and_sort(df)
    out = compute_time_diff(df)

    assert "time_diff_minutes" in out.columns

    # first row per asset should be NaN
    first_asset_rows = out[out["asset_id"] == 1]
    assert np.isnan(first_asset_rows.iloc[0]["time_diff_minutes"])


############################
# 3) GAP FLAGGING
############################
def test_flag_gaps():
    df = make_df()
    df = parse_and_sort(df)
    df = compute_time_diff(df)
    out = flag_gaps(df, gap_threshold=240)  # 4 hours

    assert "gap_flag" in out.columns

    # There should be at least one True (5 hour gap)
    assert out["gap_flag"].sum() >= 1


############################
# 4) FILTER SHORT ASSETS
############################
def test_filter_short_assets():
    df = pd.DataFrame({
        "asset_id": [1]*25 + [2]*5,
        "timestamp": pd.date_range("2025-06-01", periods=30, freq="H"),
        "soc": [50]*30,
        "is_plugged_in": [1]*30,
    })

    out = filter_short_assets(df, min_points=10)

    # asset 2 should be removed
    assert 2 not in out["asset_id"].unique()
    assert 1 in out["asset_id"].unique()
