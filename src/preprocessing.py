import pandas as pd


def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="ISO8601",
    utc=True,
)

    df = df.sort_values(["asset_id", "timestamp"])
    return df


def compute_time_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_diff_minutes"] = (
        df.groupby("asset_id")["timestamp"]
        .diff()
        .dt.total_seconds() / 60
    )
    return df


def flag_gaps(df: pd.DataFrame, gap_threshold=240) -> pd.DataFrame:
    df = df.copy()
    df["gap_flag"] = df["time_diff_minutes"] > gap_threshold
    return df


def filter_short_assets(df: pd.DataFrame, min_points=20) -> pd.DataFrame:
    counts = df["asset_id"].value_counts()
    valid_ids = counts[counts >= min_points].index
    return df[df["asset_id"].isin(valid_ids)].copy()
