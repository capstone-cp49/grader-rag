from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into 80% train, 10% validation, and 10% test.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple containing (train_df, val_df, test_df).
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=["type"], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=["type"], random_state=random_state
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(val_df) == len(test_df)

    return train_df, val_df, test_df
