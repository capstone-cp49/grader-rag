from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame, stratify_cols: list[str], random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train/val/test while preserving the distribution of stratify_cols.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        stratify_cols (list[str]): Columns to stratify by (e.g., ["problem_id", "type"]).
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    df["_stratify_group"] = df[stratify_cols].astype(str).agg("_".join, axis=1)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["_stratify_group"], random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["_stratify_group"],
        random_state=random_state,
    )

    for d in (train_df, val_df, test_df):
        d.drop(columns="_stratify_group", inplace=True)

    return train_df, val_df, test_df
