import pandas as pd

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "exam_period",
    "temperature_c",
    "is_weekend",
    "special_event",
    "is_holiday",
    "sports_or_challenge",
    "is_new_term",
    "previous_day_occupancy",
]

TARGET_COL = "occupancy_percentage"


def preprocess_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for the gym_footfall table.

    - Keeps only the required feature + target columns.
    - Enforces numeric dtypes.
    - Drops rows with missing values.
    - Clips occupancy_percentage to [0, 100].
    """
    df = raw_df.copy()

    cols = FEATURE_COLS + [TARGET_COL]
    df = df[cols]

    # enforce numeric types
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # drop rows with missing values
    df = df.dropna(subset=cols)

    # clip target
    df[TARGET_COL] = df[TARGET_COL].clip(0, 100)

    return df
