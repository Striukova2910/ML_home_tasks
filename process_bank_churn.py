"""
Data preprocessing module for Bank Churn dataset.

This module contains functions for:
- Splitting raw data
- Encoding categorical features
- Scaling numerical features
- Preparing data for model training
- Processing new unseen data
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into train and validation sets.

    Returns:
        X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_categorical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Apply One-Hot Encoding to categorical columns.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    encoder.fit(X_train[categorical_cols])

    train_encoded = encoder.transform(X_train[categorical_cols])
    val_encoded = encoder.transform(X_val[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_cols, index=X_val.index)

    X_train = X_train.drop(columns=categorical_cols)
    X_val = X_val.drop(columns=categorical_cols)

    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_val = pd.concat([X_val, val_encoded_df], axis=1)

    return X_train, X_val, encoder


def scale_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()

    scaler.fit(X_train[numeric_cols])

    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, X_val, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Full preprocessing pipeline for model training.

    Returns:
        X_train,
        y_train,
        X_val,
        y_val,
        input_cols,
        scaler,
        encoder
    """

    # Remove unnecessary column
    if "Surname" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Surname"])

    # Define categorical & numeric columns
    categorical_cols = raw_df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]

    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target_col)

    # Split
    X_train, X_val, y_train, y_val = split_data(raw_df, target_col)

    # Encode
    X_train, X_val, encoder = encode_categorical(
        X_train, X_val, categorical_cols
    )

    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric(
            X_train, X_val, numeric_cols
        )

    input_cols = X_train.columns.tolist()

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[StandardScaler] = None
) -> pd.DataFrame:
    """
    Preprocess new unseen data using fitted encoder and scaler.
    """

    if "Surname" in new_df.columns:
        new_df = new_df.drop(columns=["Surname"])

    categorical_cols = new_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns.tolist()

    # Encode
    encoded = encoder.transform(new_df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=new_df.index)

    new_df = new_df.drop(columns=categorical_cols)
    new_df = pd.concat([new_df, encoded_df], axis=1)

    # Scale
    if scaler is not None:
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

    # Align columns
    new_df = new_df.reindex(columns=input_cols, fill_value=0)

    return new_df